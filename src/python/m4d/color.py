import numpy as np

def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB numpy array (0-255 uint8 or 0-1 float) to linear light (0-1 float)."""
    if img_srgb.dtype == np.uint8:
        img_srgb = img_srgb.astype(np.float32) / 255.0
    
    img_linear = np.where(img_srgb <= 0.04045, 
                          img_srgb / 12.92, 
                          ((img_srgb + 0.055) / 1.055) ** 2.4)
    return img_linear

def linear_to_srgb(img_linear: np.ndarray, to_uint8: bool = True) -> np.ndarray:
    """Convert linear light (0-1 float) to sRGB."""
    img_linear = np.clip(img_linear, 0.0, 1.0)
    img_srgb = np.where(img_linear <= 0.0031308, 
                        img_linear * 12.92, 
                        1.055 * (img_linear ** (1 / 2.4)) - 0.055)
    
    if to_uint8:
        return (img_srgb * 255.0).round().astype(np.uint8)
    return img_srgb

def get_luminance(img_linear: np.ndarray) -> np.ndarray:
    """Get relative luminance (Y) from linear RGB."""
    return 0.2126 * img_linear[..., 0] + 0.7152 * img_linear[..., 1] + 0.0722 * img_linear[..., 2]
