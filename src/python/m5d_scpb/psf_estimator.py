import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def extract_luminance(img_lin):
    if img_lin.ndim == 3 and img_lin.shape[2] == 3:
        # Standard luminance weights
        return np.dot(img_lin, [0.2126, 0.7152, 0.0722]).astype(np.float32)
    elif img_lin.ndim == 3 and img_lin.shape[2] == 1:
        return img_lin[:, :, 0].astype(np.float32)
    return img_lin.astype(np.float32)

def detect_edges(luma, low_thresh=0.05, high_thresh=0.15):
    """
    Detects candidate edge pixels using gradient magnitude.
    Returns: gradient magnitude, gradient x, gradient y, candidate mask
    """
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    # NMS (simplified) or simple thresholding
    candidates = mag > high_thresh
    return mag, gx, gy, candidates

def sample_edge_profile(luma, x, y, gx, gy, length=7, num_samples=15):
    """
    Sample an edge profile perpendicular to the edge direction.
    """
    # Normal vector
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    nx = gx / mag
    ny = gy / mag
    
    # Sample points
    t = np.linspace(-length/2, length/2, num_samples)
    px = x + nx * t
    py = y + ny * t
    
    # Check boundaries
    h, w = luma.shape
    if np.any(px < 0) or np.any(px > w-1) or np.any(py < 0) or np.any(py > h-1):
        return None, None
        
    # Interpolate
    map_x = px.astype(np.float32)
    map_y = py.astype(np.float32)
    
    profile = cv2.remap(luma, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).flatten()
    return t, profile

def error_function(x):
    from scipy.special import erf
    return erf(x)

def gaussian_step(x, amplitude, center, sigma, offset):
    # Integral of Gaussian is Error Function
    # Step edge modeled as erf
    from scipy.special import erf
    return amplitude * (0.5 * (1 + erf((x - center) / (sigma * np.sqrt(2))))) + offset

def fit_edge_profile(t, profile):
    """
    Fit the sampled edge profile to a Gaussian step function.
    """
    # Normalize profile to [0, 1] approximately for robust fitting
    p_min, p_max = np.min(profile), np.max(profile)
    amp_guess = p_max - p_min
    if amp_guess < 0.05:
        return None
        
    offset_guess = p_min
    center_guess = 0.0 # because we centered at the gradient peak
    sigma_guess = 1.0
    
    p0 = [amp_guess, center_guess, sigma_guess, offset_guess]
    bounds = (
        [0.01, -2.0, 0.35, -0.5],
        [2.0, 2.0, 3.0, 1.5]
    )
    
    try:
        popt, pcov = curve_fit(gaussian_step, t, profile, p0=p0, bounds=bounds, maxfev=100)
        amplitude, center, sigma, offset = popt
        
        # Calculate fit residual
        fit = gaussian_step(t, amplitude, center, sigma, offset)
        rmse = np.sqrt(np.mean((profile - fit)**2))
        
        return {
            'sigma': sigma,
            'rmse': rmse,
            'amplitude': amplitude,
            'center': center,
            'offset': offset,
            't': t,
            'profile': profile,
            'fit': fit
        }
    except Exception:
        return None

def estimate_psf(img_lin, min_sigma=0.35, max_sigma=1.5):
    """
    Full pipeline to estimate a Gaussian PSF sigma from an image.
    """
    luma = extract_luminance(img_lin)
    mag, gx, gy, candidates = detect_edges(luma)
    
    # Find local maxima to thin edges
    # For simplicity, we just take top N pixels in terms of magnitude
    y_idx, x_idx = np.where(candidates)
    
    if len(y_idx) == 0:
        return {'confidence': 0.0, 'sigma': 1.0, 'reason': 'No edges found'}
        
    # Sort by magnitude descending
    mags = mag[y_idx, x_idx]
    order = np.argsort(mags)[::-1]
    
    # Take top candidates, ensuring spatial isolation (simplified)
    # Just sample a subset to avoid excessive computation
    num_to_test = min(1000, len(order))
    
    accepted = []
    
    for i in range(num_to_test):
        idx = order[i]
        x, y = x_idx[idx], y_idx[idx]
        
        # Avoid saturated edges
        if luma[y, x] > 0.95 or luma[y, x] < 0.02:
            continue
            
        t, profile = sample_edge_profile(luma, x, y, gx[y, x], gy[y, x])
        if profile is None:
            continue
            
        fit_res = fit_edge_profile(t, profile)
        if fit_res is None:
            continue
            
        if fit_res['rmse'] < 0.03 and min_sigma <= fit_res['sigma'] <= max_sigma:
            accepted.append(fit_res)
            
    if len(accepted) < 10:
        return {
            'confidence': 0.0, 
            'sigma': 1.0, 
            'reason': 'Insufficient accepted edges',
            'accepted_count': len(accepted)
        }
        
    sigmas = [r['sigma'] for r in accepted]
    median_sigma = np.median(sigmas)
    mad = np.median(np.abs(sigmas - median_sigma))
    
    confidence = min(1.0, len(accepted) / 50.0)
    if mad > 0.2:
        confidence *= 0.5 # lower confidence if estimates disagree heavily
        
    # Create normalized kernel
    k_size = int(np.ceil(median_sigma * 3) * 2 + 1)
    if k_size < 3: k_size = 3
    
    kernel = cv2.getGaussianKernel(k_size, median_sigma)
    kernel_2d = np.outer(kernel, kernel)
    
    return {
        'sigma': median_sigma,
        'mad': mad,
        'confidence': confidence,
        'kernel': kernel_2d,
        'accepted_count': len(accepted),
        'accepted_edges': accepted,
        'candidate_mask': candidates,
        'reason': 'Success'
    }
