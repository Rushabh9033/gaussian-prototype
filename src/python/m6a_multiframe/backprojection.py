import numpy as np
import cv2
from .forward_model import forward_model


def iterative_backprojection(
    hr_estimate, frames, translations, scale_factor, sigma,
    max_iters=20, initial_step=0.5, min_step=0.01, backtrack_factor=0.5,
    reg_weight=0.01
):
    """
    Iterative back-projection from multiple frames.
    
    hr_estimate: initial HR image (H, W, C) float32
    frames: list of LR observations
    translations: list of (dx, dy) in LR space (estimated)
    scale_factor: int
    sigma: PSF sigma for forward model
    max_iters: maximum iterations
    initial_step: learning rate
    """
    hr = hr_estimate.copy()
    lr_h, lr_w = frames[0].shape[:2]
    hr_h, hr_w = hr.shape[:2]
    channels = hr.shape[2] if hr.ndim == 3 else 1
    n_frames = len(frames)
    hr_sigma = sigma * scale_factor
    
    history = []
    prev_obj = float('inf')
    step = initial_step
    
    for it in range(max_iters):
        # Compute residuals from each frame
        total_gradient = np.zeros_like(hr, dtype=np.float64)
        total_obj = 0.0
        
        for frame, (dx, dy) in zip(frames, translations):
            # Convert LR translation to HR translation
            hr_dx = dx * scale_factor
            hr_dy = dy * scale_factor
            
            # Forward: translate, blur, downsample
            simulated = forward_model(hr, scale_factor, hr_sigma, translation=(hr_dx, hr_dy))
            
            # Crop to match frame size
            fh, fw = frame.shape[:2]
            sh, sw = simulated.shape[:2]
            min_h, min_w = min(fh, sh), min(fw, sw)
            
            residual = frame[:min_h, :min_w] - simulated[:min_h, :min_w]
            total_obj += float(np.sum(residual ** 2))
            
            # Back-project: upsample residual and inverse-warp
            if residual.ndim == 2:
                residual = residual[:, :, np.newaxis]
            
            bp = cv2.resize(residual, (hr_w, hr_h), interpolation=cv2.INTER_LANCZOS4)
            if bp.ndim == 2:
                bp = bp[:, :, np.newaxis]
                
            # Inverse warp (negate the translation)
            M_inv = np.float32([[1, 0, -hr_dx], [0, 1, -hr_dy]])
            for c in range(channels):
                bp_c = cv2.warpAffine(
                    bp[:, :, c], M_inv, (hr_w, hr_h),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
                )
                total_gradient[:, :, c] += bp_c
                
        total_gradient /= n_frames
        total_obj /= n_frames
        
        # Regularization: Laplacian smoothness
        if reg_weight > 0:
            for c in range(channels):
                lap = cv2.Laplacian(hr[:, :, c].astype(np.float32), cv2.CV_32F)
                total_gradient[:, :, c] -= reg_weight * lap
        
        # Backtracking: if objective increased, halve step
        if total_obj > prev_obj and it > 0:
            step *= backtrack_factor
            if step < min_step:
                step = min_step
        
        history.append({'iteration': it, 'objective': total_obj, 'step': step})
        prev_obj = total_obj
        
        # Update
        hr = hr + step * total_gradient
        hr = np.clip(hr, 0, 1).astype(np.float32)
    
    return hr, history
