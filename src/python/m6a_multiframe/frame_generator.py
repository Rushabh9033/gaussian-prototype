import numpy as np
from .forward_model import forward_model

def generate_frames(hr_img: np.ndarray, scale_factor: int, num_frames: int, base_sigma: float, seed: int = 42):
    """
    Generate independent frames with sub-pixel translations.
    Returns:
        frames: list of generated LR frames
        true_translations: list of (dx, dy) applied in HR space
    """
    rng = np.random.RandomState(seed)
    frames = []
    true_translations = []
    
    # hr_sigma matches what a realistic camera would have.
    # The PSF standard deviation is typically around 0.5 * scale_factor
    hr_sigma = base_sigma * scale_factor
    
    for i in range(num_frames):
        if i == 0:
            # First frame is strictly the reference without shift
            dx, dy = 0.0, 0.0
        else:
            # Shift uniformly in [-scale_factor, scale_factor] in HR space
            # This ensures at least -1 to +1 pixel shift in LR space
            dx = rng.uniform(-scale_factor, scale_factor)
            dy = rng.uniform(-scale_factor, scale_factor)
            
        true_translations.append((dx, dy))
        
        lr_frame = forward_model(hr_img, scale_factor, hr_sigma, translation=(dx, dy))
        
        # Add a tiny bit of quantization noise to simulate realistic sensor
        # We assume hr_img is in [0, 1] linear. So noise is extremely small.
        # But for exact controlled validation, we'll keep it deterministic and clean.
        # noise = rng.normal(0, 0.001, lr_frame.shape)
        # lr_frame = np.clip(lr_frame + noise, 0, 1)
        
        frames.append(lr_frame)
        
    return frames, true_translations
