import cv2
import numpy as np

def extract_color_field(img_linear: np.ndarray, d: int = 5, sigma_color: float = 0.1, sigma_space: float = 2.0) -> np.ndarray:
    """
    Creates a smooth, continuous base color field that prevents bleeding 
    across strong structural boundaries using a deterministic bilateral filter.
    Operates on linear RGB space.
    """
    # Bilateral filter works best in float32 for linear data
    img_f32 = img_linear.astype(np.float32)
    
    # We might need multiple passes for a very smooth field
    filtered = cv2.bilateralFilter(img_f32, d, sigma_color, sigma_space)
    filtered = cv2.bilateralFilter(filtered, d, sigma_color, sigma_space)
    
    return filtered
