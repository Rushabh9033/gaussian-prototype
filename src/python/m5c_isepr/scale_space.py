import cv2
import numpy as np

def create_scale_space(source_lin: np.ndarray, scales=[1.0, 0.8, 0.67, 0.5, 0.4, 0.33, 0.25]):
    """
    Creates a deterministic scale-space from the input photograph in linear light.
    """
    h, w = source_lin.shape[:2]
    pyramid = {}
    for s in scales:
        if s == 1.0:
            pyramid[s] = source_lin.copy()
        else:
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            # Anti-aliased area downsampling
            pyramid[s] = cv2.resize(source_lin, (nw, nh), interpolation=cv2.INTER_AREA)
    return pyramid

def create_low_surrogate(img: np.ndarray, downscale_factor: float = 2.0):
    """
    Creates the low-frequency surrogate by simulating downsampling and then restoring to the same grid.
    This mimics the baseline scaling process (e.g., Lanczos).
    """
    h, w = img.shape[:2]
    nw, nh = max(1, int(w / downscale_factor)), max(1, int(h / downscale_factor))
    
    # Simulate low-res capture
    low_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Restore to grid using the same baseline we will use for enlargement (Lanczos)
    surrogate = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return surrogate
