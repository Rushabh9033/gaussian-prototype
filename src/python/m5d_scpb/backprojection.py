import numpy as np
import cv2
from .forward_model import forward_model
from .regularization import apply_regularization, compute_regularization_energy

def compute_objective(candidate, source, scale_factor, base_sigma, config_multiplier, reg_weight, overshoot_weight):
    """
    Computes the complete objective function.
    """
    # 1. Data-fidelity loss (MSE in source space)
    simulated = forward_model(candidate, source.shape, scale_factor, base_sigma, config_multiplier)
    fidelity_error = np.mean((source - simulated)**2)
    
    # 2. Regularization energy (TV or Huber)
    reg_energy = compute_regularization_energy(candidate) * reg_weight
    
    # 3. Overshoot penalty (penalize values < 0 or > 1)
    overshoot = np.maximum(0, -candidate)**2 + np.maximum(0, candidate - 1.0)**2
    overshoot_energy = np.mean(overshoot) * overshoot_weight
    
    total_objective = fidelity_error + reg_energy + overshoot_energy
    
    return total_objective, fidelity_error, reg_energy, overshoot_energy

def backproject_residual(residual, hr_shape, hr_sigma):
    """
    Mathematical transpose of the forward model.
    1. Transpose of INTER_AREA (downsample) is flat distribution (nearest-neighbor scaling).
    2. Transpose of symmetric Gaussian blur is the same Gaussian blur.
    """
    # Nearest neighbor distributes the error evenly
    up_res = cv2.resize(residual, (hr_shape[1], hr_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Blur
    k_size = int(np.ceil(hr_sigma * 3) * 2 + 1)
    if k_size < 3: k_size = 3
    if k_size % 2 == 0: k_size += 1
    
    kernel = cv2.getGaussianKernel(k_size, hr_sigma)
    kernel_2d = np.outer(kernel, kernel)
    
    bp_res = cv2.filter2D(up_res, -1, kernel_2d, borderType=cv2.BORDER_REPLICATE)
    
    return bp_res

def edge_aware_damping(hr_luma, bp_res):
    """
    Reduces correction flow across strong edges to prevent ringing and halo amplification.
    """
    gx = cv2.Sobel(hr_luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hr_luma, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    # Simple damping function: 1 / (1 + alpha * mag)
    damping = 1.0 / (1.0 + 5.0 * mag)
    
    # Apply damping mainly to the highest frequency components of the residual? 
    # For simplicity, damp the entire residual where edges are strong.
    # Actually, standard edge-aware correction dampens flow across edges.
    return bp_res * damping[:, :, np.newaxis] if bp_res.ndim == 3 else bp_res * damping

def run_scpb_optimization(source, candidate_init, scale_factor, base_sigma, config, max_iters, initial_step):
    """
    Source-consistent back-projection with backtracking.
    """
    sh, sw = source.shape[:2]
    ch, cw = candidate_init.shape[:2]
    
    # Pad source and candidate to exact multiples to ensure uniform integer scaling
    req_ch = max(ch, sh * scale_factor)
    if req_ch % scale_factor != 0:
        req_ch += scale_factor - (req_ch % scale_factor)
        
    req_cw = max(cw, sw * scale_factor)
    if req_cw % scale_factor != 0:
        req_cw += scale_factor - (req_cw % scale_factor)
        
    req_sh = req_ch // scale_factor
    req_sw = req_cw // scale_factor
    
    # Pad source
    pad_sh = req_sh - sh
    pad_sw = req_sw - sw
    if pad_sh > 0 or pad_sw > 0:
        src_padded = cv2.copyMakeBorder(source, 0, pad_sh, 0, pad_sw, cv2.BORDER_REPLICATE)
    else:
        src_padded = source.copy()
        
    # Pad candidate
    pad_ch = req_ch - ch
    pad_cw = req_cw - cw
    if pad_ch > 0 or pad_cw > 0:
        cand_padded = cv2.copyMakeBorder(candidate_init, 0, pad_ch, 0, pad_cw, cv2.BORDER_REPLICATE)
    else:
        cand_padded = candidate_init.copy()
        
    candidate = cand_padded
    source_opt = src_padded
    
    step_size = initial_step
    hr_sigma = base_sigma * scale_factor * config.get('psf_multiplier', 1.0)
    reg_weight = config.get('reg_weight', 0.001)
    overshoot_weight = config.get('overshoot_weight', 10.0)
    
    current_obj, fid, reg, over = compute_objective(
        candidate, source_opt, scale_factor, base_sigma, config.get('psf_multiplier', 1.0), reg_weight, overshoot_weight
    )
    
    history = []
    
    for it in range(max_iters):
        simulated = forward_model(candidate, source_opt.shape, scale_factor, base_sigma, config.get('psf_multiplier', 1.0))
        residual = source_opt - simulated
        bp_res = backproject_residual(residual, candidate.shape, hr_sigma)
        
        hr_luma = candidate[:, :, 0] if candidate.ndim == 3 and candidate.shape[2] == 1 else np.dot(candidate, [0.2126, 0.7152, 0.0722]).astype(np.float32)
        if bp_res.ndim == 2: bp_res = bp_res[:, :, np.newaxis]
        damped_res = edge_aware_damping(hr_luma, bp_res)
        
        proposed = candidate + step_size * damped_res
        proposed = apply_regularization(proposed, reg_weight, step_size)
        
        prop_obj, p_fid, p_reg, p_over = compute_objective(
            proposed, source_opt, scale_factor, base_sigma, config.get('psf_multiplier', 1.0), reg_weight, overshoot_weight
        )
        
        if prop_obj < current_obj:
            candidate = proposed
            history.append({'iteration': it, 'accepted': True, 'objective': float(prop_obj), 'fidelity': float(p_fid), 'regularization': float(p_reg), 'overshoot': float(p_over), 'step_size': float(step_size)})
            if (current_obj - prop_obj) / current_obj < 1e-4: break
            current_obj = prop_obj
        else:
            history.append({'iteration': it, 'accepted': False, 'objective': float(prop_obj), 'fidelity': float(p_fid), 'regularization': float(p_reg), 'overshoot': float(p_over), 'step_size': float(step_size)})
            step_size *= 0.5
            if step_size < 1e-3: break
                
    candidate = np.clip(candidate, 0.0, 1.0)
    return candidate[:ch, :cw], history
