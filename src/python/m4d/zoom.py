import cv2
import numpy as np
from PIL import Image
from .color import srgb_to_linear, linear_to_srgb, get_luminance
import scipy.ndimage as ndimage

def zoom_nearest(img_srgb: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbour interpolation. target_shape is (w, h)."""
    return cv2.resize(img_srgb, target_shape, interpolation=cv2.INTER_NEAREST)

def zoom_bicubic(img_srgb: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Bicubic interpolation."""
    return cv2.resize(img_srgb, target_shape, interpolation=cv2.INTER_CUBIC)

def zoom_lanczos(img_srgb: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Lanczos interpolation using PIL."""
    pil_img = Image.fromarray(img_srgb)
    pil_resized = pil_img.resize(target_shape, resample=Image.Resampling.LANCZOS)
    return np.array(pil_resized)

def structure_tensor_edges(luma: np.ndarray):
    """Compute structure tensor to find edge confidence and orientation."""
    dx = cv2.Scharr(luma, cv2.CV_64F, 1, 0)
    dy = cv2.Scharr(luma, cv2.CV_64F, 0, 1)
    
    # Elements of the structure tensor
    Ixx = dx**2
    Iyy = dy**2
    Ixy = dx * dy
    
    # Smooth them to aggregate neighborhood
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), 1.0)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), 1.0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 1.0)
    
    # Eigenvalues
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy**2
    diff = np.sqrt(np.maximum((Sxx - Syy)**2 + 4 * Sxy**2, 0))
    
    lambda1 = (trace + diff) / 2
    lambda2 = (trace - diff) / 2
    
    # Confidence: difference between eigenvalues indicates strong edge vs flat/corner
    coherence = np.where(lambda1 + lambda2 > 1e-5, (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-5), 0)
    
    return coherence, dx, dy

def zoom_hybrid(img_srgb: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Original Classical Edge-Aware Hybrid:
    - sRGB to linear
    - Lanczos upscale
    - Luminance extraction & Structure tensor edge detection
    - Conservative residual sharpening with local min/max anti-ringing
    - Linear to sRGB
    """
    # 1. Linearize
    lin_img = srgb_to_linear(img_srgb)
    
    # 2. Base upscale (Lanczos)
    h_target, w_target = target_shape[1], target_shape[0]
    
    # Upscale channels individually or via PIL (PIL needs uint8 or handles float poorly, 
    # so we'll use OpenCV's Lanczos4 for float data).
    base_up = cv2.resize(lin_img, target_shape, interpolation=cv2.INTER_LANCZOS4)
    
    # 3. Luminance & Edges
    luma_up = get_luminance(base_up)
    coherence, dx, dy = structure_tensor_edges(luma_up)
    
    # 4. Sharpening: we use an unsharp mask modulated by edge coherence
    blur = cv2.GaussianBlur(base_up, (5, 5), 1.5)
    high_freq = base_up - blur
    
    # Apply sharpening proportional to coherence
    coherence_expanded = np.expand_dims(coherence, axis=-1)
    sharpened = base_up + high_freq * (coherence_expanded * 1.5) # Conservative sharpening factor
    
    # 5. Anti-ringing clamp
    # Find local min/max from the base upscaled image (which has some ringing, but less than sharpened)
    # Actually, a better anti-ringing is clamping to the local min/max of the *original* pixels mapped to upscaled space,
    # but using morphological operations on base_up is standard.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local_min = cv2.erode(base_up, kernel)
    local_max = cv2.dilate(base_up, kernel)
    
    clamped = np.clip(sharpened, local_min, local_max)
    clamped = np.clip(clamped, 0.0, 1.0)
    
    # 6. Convert back
    final_srgb = linear_to_srgb(clamped, to_uint8=True)
    return final_srgb
