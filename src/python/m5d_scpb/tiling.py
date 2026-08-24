import numpy as np
import cv2
from .backprojection import run_scpb_optimization

def compute_halo_size(scale_factor, base_sigma, config, max_iters):
    # Calculate exact dependency radius for PDE mathematically
    hr_sigma = base_sigma * scale_factor * config.get('psf_multiplier', 1.0)
    blur_ksize = int(np.ceil(3 * hr_sigma)) * 2 + 1
    blur_radius = blur_ksize // 2
    
    # Forward pass: blur (blur_radius) + downsample (scale_factor)
    # Backward pass: upsample (4 * scale_factor) + blur (blur_radius)
    # Regularization: finite difference (1)
    radius_per_iter = blur_radius * 2 + 5 * scale_factor + 1
    
    halo = max_iters * radius_per_iter + 4 * scale_factor + 4
    return int(halo)

def reconstruct_tiled(source_full, scale_factor, base_sigma, config, max_iters, initial_step, target_shape=None, tile_size=64):
    """
    Run SCPB using overlapping tiles mathematically equivalent to full-image processing.
    """
    if target_shape is not None:
        th, tw = target_shape
    else:
        sh, sw = source_full.shape[:2]
        th, tw = sh * scale_factor, sw * scale_factor
        
    halo = compute_halo_size(scale_factor, base_sigma, config, max_iters)
    
    candidate_full = cv2.resize(source_full, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    
    req_ch = max(th, source_full.shape[0] * scale_factor)
    req_cw = max(tw, source_full.shape[1] * scale_factor)
    
    if req_ch % scale_factor != 0: req_ch += scale_factor - (req_ch % scale_factor)
    if req_cw % scale_factor != 0: req_cw += scale_factor - (req_cw % scale_factor)
    
    pad_ch = req_ch - th
    pad_cw = req_cw - tw
    candidate_padded = cv2.copyMakeBorder(candidate_full, 0, pad_ch, 0, pad_cw, cv2.BORDER_REPLICATE)
    
    req_sh = req_ch // scale_factor
    req_sw = req_cw // scale_factor
    
    pad_sh = req_sh - source_full.shape[0]
    pad_sw = req_sw - source_full.shape[1]
    source_padded = cv2.copyMakeBorder(source_full, 0, pad_sh, 0, pad_sw, cv2.BORDER_REPLICATE)
    
    out_padded = np.zeros_like(candidate_padded)
    
    tile_count = 0
    peak_memory = 0
    
    for y in range(0, req_ch, tile_size):
        for x in range(0, req_cw, tile_size):
            # Compute padded tile coordinates in HR space
            y0 = max(0, y - halo)
            y1 = min(req_ch, y + tile_size + halo)
            x0 = max(0, x - halo)
            x1 = min(req_cw, x + tile_size + halo)
            
            sy0, sy1 = y0 // scale_factor, y1 // scale_factor
            sx0, sx1 = x0 // scale_factor, x1 // scale_factor
            
            # Ensure integer alignment
            y0, y1 = sy0 * scale_factor, sy1 * scale_factor
            x0, x1 = sx0 * scale_factor, sx1 * scale_factor
            
            src_tile = source_padded[sy0:sy1, sx0:sx1]
            cand_tile = candidate_padded[y0:y1, x0:x1].copy()
            
            out_tile, _ = run_scpb_optimization(
                src_tile, cand_tile, scale_factor, base_sigma, config, max_iters, initial_step
            )
            
            tile_mem = out_tile.nbytes * 5  # rough approximation
            if tile_mem > peak_memory: peak_memory = tile_mem
            
            valid_y0 = y - y0
            valid_y1 = valid_y0 + min(tile_size, req_ch - y)
            valid_x0 = x - x0
            valid_x1 = valid_x0 + min(tile_size, req_cw - x)
            
            wy0 = y
            wy1 = y + min(tile_size, req_ch - y)
            wx0 = x
            wx1 = x + min(tile_size, req_cw - x)
            
            out_padded[wy0:wy1, wx0:wx1] = out_tile[valid_y0:valid_y1, valid_x0:valid_x1]
            tile_count += 1
            
    return out_padded[:th, :tw], tile_count, halo, peak_memory
