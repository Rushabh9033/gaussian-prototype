import numpy as np
import cv2
from .backprojection import run_scpb_optimization

def process_tile(source_tile, candidate_tile, scale_factor, base_sigma, config, max_iters, initial_step, global_offset_x, global_offset_y):
    """
    Process a single tile. SCPB operates entirely locally within the padded tile, but we need to ensure the grid aligns if there were sub-pixel shifts. 
    However, SCPB uses standard convolutions and pixel-wise additions which are translation-invariant.
    As long as the forward model and backprojection use same padding and don't introduce boundary artifacts inside the valid region, 
    we just need sufficient tile padding.
    """
    # For SCPB, the operations are INTER_AREA, Gaussian Blur, INTER_NEAREST.
    # Blur has a kernel size, INTER_AREA has a window size.
    # They are completely translation invariant as long as padding is sufficient.
    
    out, history = run_scpb_optimization(source_tile, candidate_tile, scale_factor, base_sigma, config, max_iters, initial_step)
    
    return out

def reconstruct_tiled(source_full, scale_factor, base_sigma, config, max_iters, initial_step, target_shape=None, tile_size=64, padding=512):
    """
    Run SCPB using overlapping tiles.
    """
    sh, sw = source_full.shape[:2]
    if target_shape is not None:
        th, tw = target_shape[:2]
    else:
        th, tw = sh * scale_factor, sw * scale_factor
        
    candidate_full = cv2.resize(source_full, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    
    req_ch = max(th, sh * scale_factor)
    if req_ch % scale_factor != 0:
        req_ch += scale_factor - (req_ch % scale_factor)
    req_cw = max(tw, sw * scale_factor)
    if req_cw % scale_factor != 0:
        req_cw += scale_factor - (req_cw % scale_factor)
        
    req_sh = req_ch // scale_factor
    req_sw = req_cw // scale_factor
    
    pad_sh = req_sh - sh
    pad_sw = req_sw - sw
    if pad_sh > 0 or pad_sw > 0:
        source_padded = cv2.copyMakeBorder(source_full, 0, pad_sh, 0, pad_sw, cv2.BORDER_REPLICATE)
    else:
        source_padded = source_full
        
    pad_ch = req_ch - th
    pad_cw = req_cw - tw
    if pad_ch > 0 or pad_cw > 0:
        candidate_padded = cv2.copyMakeBorder(candidate_full, 0, pad_ch, 0, pad_cw, cv2.BORDER_REPLICATE)
    else:
        candidate_padded = candidate_full
        
    out_padded = np.zeros_like(candidate_padded)
    
    for y in range(0, req_ch, tile_size):
        for x in range(0, req_cw, tile_size):
            y0 = max(0, y - padding)
            y1 = min(req_ch, y + tile_size + padding)
            x0 = max(0, x - padding)
            x1 = min(req_cw, x + tile_size + padding)
            
            sy0 = y0 // scale_factor
            sy1 = y1 // scale_factor
            sx0 = x0 // scale_factor
            sx1 = x1 // scale_factor
            
            ty0 = sy0 * scale_factor
            ty1 = sy1 * scale_factor
            tx0 = sx0 * scale_factor
            tx1 = sx1 * scale_factor
            
            src_tile = source_padded[sy0:sy1, sx0:sx1].copy()
            cand_tile = candidate_padded[ty0:ty1, tx0:tx1].copy()
            
            out_tile = process_tile(src_tile, cand_tile, scale_factor, base_sigma, config, max_iters, initial_step, tx0, ty0)
            
            valid_y0 = y - ty0
            valid_y1 = min(y + tile_size, req_ch) - ty0
            valid_x0 = x - tx0
            valid_x1 = min(x + tile_size, req_cw) - tx0
            
            wy0 = ty0 + valid_y0
            wy1 = ty0 + valid_y1
            wx0 = tx0 + valid_x0
            wx1 = tx0 + valid_x1
            
            out_padded[wy0:wy1, wx0:wx1] = out_tile[valid_y0:valid_y1, valid_x0:valid_x1]
            
    return out_padded[:th, :tw]
