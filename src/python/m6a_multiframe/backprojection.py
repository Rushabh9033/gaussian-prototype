import numpy as np
import cv2
from .forward_model import forward_model


def compute_objective(hr_img, frames, translations, scale_factor, hr_sigma, reg_weight):
    """
    Computes the complete objective function:
    Objective = Data Fidelity (MSE of residuals) + Regularization (Laplacian energy)
    """
    n_frames = len(frames)
    total_data_obj = 0.0
    
    # 1. Data fidelity term
    for frame, (dx, dy) in zip(frames, translations):
        hr_dx = dx * scale_factor
        hr_dy = dy * scale_factor
        simulated = forward_model(hr_img, scale_factor, hr_sigma, translation=(hr_dx, hr_dy))
        
        fh, fw = frame.shape[:2]
        sh, sw = simulated.shape[:2]
        min_h, min_w = min(fh, sh), min(fw, sw)
        
        residual = frame[:min_h, :min_w] - simulated[:min_h, :min_w]
        total_data_obj += float(np.sum(residual ** 2))
        
    data_fidelity = total_data_obj / n_frames
    
    # 2. Regularization term
    reg_penalty = 0.0
    if reg_weight > 0:
        channels = hr_img.shape[2] if hr_img.ndim == 3 else 1
        for c in range(channels):
            # Gradient energy (sum of squared gradients) corresponds to Laplacian smoothing
            # E_reg = sum( (dx I)^2 + (dy I)^2 )
            img_c = hr_img[:, :, c].astype(np.float32)
            gx = cv2.Sobel(img_c, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_c, cv2.CV_32F, 0, 1, ksize=3)
            reg_penalty += float(np.sum(gx ** 2 + gy ** 2))
            
    total_objective = data_fidelity + reg_weight * reg_penalty
    return total_objective, data_fidelity, reg_weight * reg_penalty


def compute_gradient(hr_img, frames, translations, scale_factor, hr_sigma, reg_weight):
    hr_h, hr_w = hr_img.shape[:2]
    channels = hr_img.shape[2] if hr_img.ndim == 3 else 1
    n_frames = len(frames)
    
    total_gradient = np.zeros_like(hr_img, dtype=np.float64)
    
    for frame, (dx, dy) in zip(frames, translations):
        hr_dx = dx * scale_factor
        hr_dy = dy * scale_factor
        
        simulated = forward_model(hr_img, scale_factor, hr_sigma, translation=(hr_dx, hr_dy))
        
        fh, fw = frame.shape[:2]
        sh, sw = simulated.shape[:2]
        min_h, min_w = min(fh, sh), min(fw, sw)
        
        residual = frame[:min_h, :min_w] - simulated[:min_h, :min_w]
        
        if residual.ndim == 2:
            residual = residual[:, :, np.newaxis]
            
        bp = cv2.resize(residual, (hr_w, hr_h), interpolation=cv2.INTER_LANCZOS4)
        if bp.ndim == 2:
            bp = bp[:, :, np.newaxis]
            
        M_inv = np.float32([[1, 0, -hr_dx], [0, 1, -hr_dy]])
        for c in range(channels):
            bp_c = cv2.warpAffine(
                bp[:, :, c], M_inv, (hr_w, hr_h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
            )
            total_gradient[:, :, c] += bp_c
            
    total_gradient /= n_frames
    
    if reg_weight > 0:
        for c in range(channels):
            # Gradient of gradient energy is -2 * Laplacian
            # So the update direction for -gradient(E) is to add Laplacian
            # cv2.Laplacian applies a discrete laplacian operator
            lap = cv2.Laplacian(hr_img[:, :, c].astype(np.float32), cv2.CV_32F)
            total_gradient[:, :, c] += reg_weight * lap
            
    return total_gradient


def iterative_backprojection(
    hr_estimate, frames, translations, scale_factor, sigma,
    max_iters=5, initial_step=0.1, min_step=0.001, backtrack_factor=0.5,
    reg_weight=0.15
):
    """
    Iterative back-projection from multiple frames with correct backtracking.
    """
    hr = hr_estimate.copy()
    hr_sigma = sigma * scale_factor
    
    history = []
    step = initial_step
    
    # Compute initial objective
    current_obj, data_fid, reg_pen = compute_objective(
        hr, frames, translations, scale_factor, hr_sigma, reg_weight
    )
    
    history.append({
        'iteration': 0,
        'objective': current_obj,
        'data_fidelity': data_fid,
        'regularization': reg_pen,
        'step': step,
        'accepted': True
    })
    
    for it in range(1, max_iters + 1):
        # 1. Compute update direction based on current hr
        gradient = compute_gradient(
            hr, frames, translations, scale_factor, hr_sigma, reg_weight
        )
        
        # 2. Generate proposed candidate
        proposed_hr = hr + step * gradient
        proposed_hr = np.clip(proposed_hr, 0, 1).astype(np.float32)
        
        # 3. Calculate objective of proposed candidate
        prop_obj, prop_data_fid, prop_reg_pen = compute_objective(
            proposed_hr, frames, translations, scale_factor, hr_sigma, reg_weight
        )
        
        # 4. Accept or reject
        if prop_obj <= current_obj:
            # Accept
            hr = proposed_hr
            current_obj = prop_obj
            accepted = True
        else:
            # Reject and backtrack
            step *= backtrack_factor
            if step < min_step:
                step = min_step
            accepted = False
            
        history.append({
            'iteration': it,
            'objective': prop_obj, # Record the objective evaluated
            'data_fidelity': prop_data_fid,
            'regularization': prop_reg_pen,
            'step': step,
            'accepted': accepted
        })
        
    return hr, history
