import cv2
import numpy as np

def build_laplacian_pyramid(image: np.ndarray, levels: int = 3) -> list:
    """
    Builds a true signed Laplacian pyramid.
    Level 0 is the finest high-frequency residual.
    The last level is the low-frequency base.
    """
    pyramid = []
    current = image.copy().astype(np.float32)
    
    for i in range(levels - 1):
        h, w = current.shape[:2]
        down = cv2.resize(current, (w//2, h//2), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Signed residual
        residual = current - up
        pyramid.append(residual)
        
        current = down
        
    pyramid.append(current) # Base level
    return pyramid

def reconstruct_laplacian_pyramid(pyramid: list) -> np.ndarray:
    """
    Deterministically decodes the Laplacian pyramid.
    """
    current = pyramid[-1]
    
    for i in range(len(pyramid) - 2, -1, -1):
        residual = pyramid[i]
        h, w = residual.shape[:2]
        
        up = cv2.resize(current, (w, h), interpolation=cv2.INTER_CUBIC)
        current = up + residual
        
    return current

def extract_source_residuals(source_linear: np.ndarray, base_1x_linear: np.ndarray) -> np.ndarray:
    """
    Extracts the source-supported residual as a signed float32 array.
    """
    # Simply the difference for this implementation, which acts as a 1-level residual
    # over the vector-continuous base.
    residual = source_linear.astype(np.float32) - base_1x_linear.astype(np.float32)
    return residual

def compute_residual_energy(residual: np.ndarray) -> float:
    """
    Returns the energy (mean squared value) of a residual level.
    """
    return float(np.mean(residual**2))
