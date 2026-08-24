import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.special import erf

def extract_luminance(img_lin):
    if img_lin.ndim == 3 and img_lin.shape[2] == 3:
        return np.dot(img_lin, [0.2126, 0.7152, 0.0722]).astype(np.float32)
    elif img_lin.ndim == 3 and img_lin.shape[2] == 1:
        return img_lin[:, :, 0].astype(np.float32)
    return img_lin.astype(np.float32)

def gaussian_step(x, amplitude, center, sigma, offset):
    return amplitude * (0.5 * (1 + erf((x - center) / (sigma * np.sqrt(2))))) + offset

def estimate_psf(img_lin, min_sigma=0.35, max_sigma=2.0, patch_size=9):
    luma = extract_luminance(img_lin)
    h, w = luma.shape
    
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    # 1. Non-maximum suppression and thresholding
    mag_pad = cv2.copyMakeBorder(mag, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    is_max_x = (mag >= mag_pad[1:-1, 0:-2]) & (mag >= mag_pad[1:-1, 2:])
    is_max_y = (mag >= mag_pad[0:-2, 1:-1]) & (mag >= mag_pad[2:, 1:-1])
    is_max = is_max_x | is_max_y
    candidates = (mag > 0.1) & is_max
    
    # 2. Border rejection
    border = patch_size // 2 + 2
    candidates[:border, :] = False
    candidates[-border:, :] = False
    candidates[:, :border] = False
    candidates[:, -border:] = False
    
    y_idx, x_idx = np.where(candidates)
    mags = mag[y_idx, x_idx]
    order = np.argsort(mags)[::-1]
    
    accepted_edges = []
    rejected_edges = []
    
    # Grid for spatial isolation
    grid_size = 15
    grid_w = (w + grid_size - 1) // grid_size
    grid_h = (h + grid_size - 1) // grid_size
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    
    for idx in order:
        x, y = x_idx[idx], y_idx[idx]
        
        # Spatial isolation
        gx_idx, gy_idx = x // grid_size, y // grid_size
        if occupied[gy_idx, gx_idx]:
            rejected_edges.append({'x': int(x), 'y': int(y), 'reason': 'not_isolated'})
            continue
            
        # Extract patch
        patch = luma[y-border:y+border+1, x-border:x+border+1]
        
        # Saturation rejection
        if np.any(patch >= 1.0) or np.any(patch <= 0.0):
            rejected_edges.append({'x': int(x), 'y': int(y), 'reason': 'saturated'})
            continue
            
        # Straight-edge / curve / texture rejection
        patch_gx = gx[y-border:y+border+1, x-border:x+border+1]
        patch_gy = gy[y-border:y+border+1, x-border:x+border+1]
        patch_mag = mag[y-border:y+border+1, x-border:x+border+1]
        
        # Orientation consistency
        angles = np.arctan2(patch_gy, patch_gx)
        center_angle = angles[border, border]
        angle_diff = np.abs(np.arctan2(np.sin(angles - center_angle), np.cos(angles - center_angle)))
        
        strong_pixels = patch_mag > 0.05
        if np.sum(strong_pixels) < 5:
            rejected_edges.append({'x': int(x), 'y': int(y), 'reason': 'texture'})
            continue
            
        if np.median(angle_diff[strong_pixels]) > 0.3:
            rejected_edges.append({'x': int(x), 'y': int(y), 'reason': 'curved'})
            continue
            
        # Subpixel edge profile sampling
        nx = patch_gx[border, border] / patch_mag[border, border]
        ny = patch_gy[border, border] / patch_mag[border, border]
        
        length = 7.0
        num_samples = 15
        t = np.linspace(-length/2, length/2, num_samples)
        px = x + nx * t
        py = y + ny * t
        
        map_x = px.astype(np.float32)
        map_y = py.astype(np.float32)
        profile = cv2.remap(luma, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).flatten()
        
        # Fit Gaussian step
        p_min, p_max = np.min(profile), np.max(profile)
        amp_guess = p_max - p_min
        if amp_guess < 0.1:
            rejected_edges.append({'x': int(x), 'y': int(y), 'reason': 'weak_amplitude'})
            continue
            
        p0 = [amp_guess, 0.0, 1.0, p_min]
        bounds = ([0.05, -2.0, min_sigma, -0.5], [2.0, 2.0, max_sigma, 1.5])
        
        try:
            popt, pcov = curve_fit(gaussian_step, t, profile, p0=p0, bounds=bounds, maxfev=100)
            amplitude, center, sigma, offset = popt
            fit = gaussian_step(t, amplitude, center, sigma, offset)
            rmse = np.sqrt(np.mean((profile - fit)**2))
            
            if rmse > 0.03:
                rejected_edges.append({'x': int(x), 'y': int(y), 'reason': 'poor_fit'})
                continue
                
            accepted_edges.append({
                'x': int(x), 'y': int(y),
                'sigma': float(sigma), 'rmse': float(rmse),
                'amplitude': float(amplitude), 'center': float(center), 'offset': float(offset),
                't': t.tolist(), 'profile': profile.tolist(), 'fit': fit.tolist(),
                'reason': 'accepted'
            })
            occupied[gy_idx, gx_idx] = True
            
        except Exception:
            rejected_edges.append({'x': int(x), 'y': int(y), 'reason': 'fit_failed'})
            
    if len(accepted_edges) < 5:
        return {
            'confidence': 0.0,
            'sigma': 1.0,
            'reason': 'Insufficient accepted edges',
            'accepted_count': len(accepted_edges),
            'accepted_edges': accepted_edges,
            'rejected_edges': rejected_edges
        }
        
    sigmas = np.array([e['sigma'] for e in accepted_edges])
    median_sigma = float(np.median(sigmas))
    mad = float(np.median(np.abs(sigmas - median_sigma)))
    
    # Confidence heuristics
    confidence = min(1.0, len(accepted_edges) / 15.0)
    if mad > 0.15: confidence *= np.exp(-(mad - 0.15) * 5)
    
    return {
        'sigma': median_sigma,
        'mad': mad,
        'confidence': float(confidence),
        'reason': 'Success',
        'accepted_count': len(accepted_edges),
        'accepted_edges': accepted_edges,
        'rejected_edges': rejected_edges
    }
