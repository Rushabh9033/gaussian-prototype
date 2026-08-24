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

def zoom_hybrid(img_srgb: np.ndarray, target_shape: tuple, strength: float = 1.0) -> np.ndarray:
    """
    Original Classical Edge-Aware Hybrid:
    - sRGB to linear
    - Lanczos upscale
    - Luminance extraction & Structure tensor edge detection
    - Direction-aware sharpening (sharpening across edges)
    - Conservative residual sharpening with local min/max anti-ringing
    - Linear to sRGB
    """
    # 1. Linearize
    lin_img = srgb_to_linear(img_srgb)
    
    # 2. Base upscale (Lanczos)
    base_up = cv2.resize(lin_img, target_shape, interpolation=cv2.INTER_LANCZOS4)
    
    # 3. Luminance & Edges
    luma_up = get_luminance(base_up)
    
    # First derivatives (Scharr is more rotationally symmetric than Sobel)
    dx = cv2.Scharr(luma_up, cv2.CV_64F, 1, 0)
    dy = cv2.Scharr(luma_up, cv2.CV_64F, 0, 1)
    
    # Gradient magnitude
    mag = np.sqrt(dx**2 + dy**2)
    mag_max = mag.max() + 1e-5
    
    # Normalized gradient direction vectors (nx, ny) point across the edge
    nx = dx / (mag + 1e-5)
    ny = dy / (mag + 1e-5)
    
    # Second derivatives
    dxx = cv2.Scharr(dx, cv2.CV_64F, 1, 0)
    dyy = cv2.Scharr(dy, cv2.CV_64F, 0, 1)
    dxy = cv2.Scharr(dx, cv2.CV_64F, 0, 1)
    
    # Directional second derivative across the edge
    D_nn = nx**2 * dxx + 2 * nx * ny * dxy + ny**2 * dyy
    
    # 4. Sharpening: subtract D_nn proportional to edge strength
    edge_weight = mag / mag_max
    # Normalize D_nn to roughly match image scale
    sharpen_term = -D_nn * edge_weight * strength * 0.05
    
    sharpen_term_rgb = np.expand_dims(sharpen_term, axis=-1)
    sharpened = base_up + sharpen_term_rgb
    
    # 5. Anti-ringing clamp
    # Use morphology on the base upscaled image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local_min = cv2.erode(base_up, kernel)
    local_max = cv2.dilate(base_up, kernel)
    
    clamped = np.clip(sharpened, local_min, local_max)
    clamped = np.clip(clamped, 0.0, 1.0)
    
    # 6. Convert back
    final_srgb = linear_to_srgb(clamped, to_uint8=True)
    return final_srgb
