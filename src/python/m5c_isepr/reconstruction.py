import numpy as np
import cv2
import hashlib

from .patch_dictionary import extract_patches

def _create_hann_window(size):
    """Creates a 2D Hann window for overlap-add."""
    w1d = np.hanning(size + 2)[1:-1] # Avoid true zero at edges if possible, or just standard Hann
    w2d = np.outer(w1d, w1d)
    return w2d.astype(np.float32)

def calculate_confidence(distances, target_vars, k, config):
    """
    Confidence gating based on distance, ratio, and variance.
    Returns: weight array (N,)
    """
    # Strict thresholds from config
    max_dist = config.get("max_distance", 5.0)
    ratio_thresh = config.get("ratio_threshold", 0.9)
    
    dist_best = distances[:, 0]
    
    # Absolute distance gate
    conf_dist = np.maximum(0.0, 1.0 - (dist_best / max_dist))
    
    # Distance ratio gate (best vs second best) if k > 1
    if k > 1:
        dist_second = np.maximum(distances[:, 1], 1e-6)
        ratio = dist_best / dist_second
        conf_ratio = np.where(ratio < ratio_thresh, 1.0, 0.0)
    else:
        conf_ratio = 1.0
        
    # Combine
    confidence = conf_dist * conf_ratio
    return confidence

def process_tile(target_tile_rgb, dictionary, config, tile_x, tile_y, scale_factor, core_tx, core_ty, core_tsize):
    """
    Process a single output tile.
    target_tile_rgb is the baseline (Lanczos) linear light tile.
    """
    h, w = target_tile_rgb.shape[:2]
    patch_size = dictionary.patch_size
    stride = dictionary.stride
    k = config.get("top_k", 3)
    gain = config.get("residual_gain", 1.0)
    
    # Target luma
    target_luma = cv2.cvtColor(target_tile_rgb, cv2.COLOR_RGB2GRAY) if target_tile_rgb.shape[-1] == 3 else target_tile_rgb
    
    # Extract overlapping patches from target
    target_patches, coords = extract_patches(target_tile_rgb, patch_size, stride=1, global_offset_x=tile_x, global_offset_y=tile_y)
    target_patches_luma, _ = extract_patches(target_luma, patch_size, stride=1, global_offset_x=tile_x, global_offset_y=tile_y)
    
    # Filter patches to ONLY those whose top-left coordinate is within the core tile.
    # This prevents duplicate processing of overlap regions across tiles.
    global_x = coords[:, 0] + tile_x
    global_y = coords[:, 1] + tile_y
    core_mask = (global_x >= core_tx) & (global_x < core_tx + core_tsize) & (global_y >= core_ty) & (global_y < core_ty + core_tsize)
    
    target_patches_luma = target_patches_luma[core_mask]
    coords = coords[core_mask]
    
    # Avoid processing completely flat regions
    target_vars = np.var(target_patches_luma, axis=(1,2,3))
    active_mask = target_vars > 1e-5
    
    output_accum = np.zeros_like(target_tile_rgb)
    weight_accum = np.zeros((h, w, 1), dtype=np.float32)
    window = _create_hann_window(patch_size)[..., np.newaxis]
    
    confidence_map = np.zeros((h, w), dtype=np.float32)
    accepted_count = 0
    total_active = np.sum(active_mask)
    
    prov_records = []
    
    if total_active > 0:
        active_luma = target_patches_luma[active_mask]
        distances, indices = dictionary.search(active_luma, k=k)
        
        conf = calculate_confidence(distances, target_vars[active_mask], k, config)
        
        # We also need to enforce residual agreement
        res_candidates = dictionary.residuals[indices] # (N, K, P, P, 3)
        mean_res = np.mean(res_candidates, axis=1) # (N, P, P, 3)
        
        # Simple agreement: variance across K candidates. High variance = disagreement
        if k > 1:
            var_res = np.var(res_candidates, axis=1)
            var_mean = np.mean(var_res, axis=(1,2,3))
            conf *= np.maximum(0.0, 1.0 - (var_mean / config.get("max_residual_variance", 0.01)))
        
        # Apply gain and confidence
        final_res = mean_res * conf[:, np.newaxis, np.newaxis, np.newaxis] * gain
        
        # Limit residual energy relative to local target variance
        res_energy = np.var(final_res, axis=(1,2,3))
        max_energy = target_vars[active_mask] * config.get("energy_multiplier", 1.0)
        over_limit = res_energy > max_energy
        if np.any(over_limit):
            scale_down = np.sqrt(max_energy[over_limit] / (res_energy[over_limit] + 1e-6))
            final_res[over_limit] *= scale_down[:, np.newaxis, np.newaxis, np.newaxis]
        
        accepted_idx = np.where(conf > 0.0)[0]
        accepted_count = len(accepted_idx)
        
        act_coords = coords[active_mask]
        
        for i in range(len(conf)):
            c = conf[i]
            if c > 0:
                cx, cy = act_coords[i]
                output_accum[cy:cy+patch_size, cx:cx+patch_size] += final_res[i] * window
                
            # Accumulate weight for ALL patches, even if residual is zero, 
            # so we can normalize correctly (overlap-add baseline)
            cx, cy = act_coords[i]
            weight_accum[cy:cy+patch_size, cx:cx+patch_size] += window
            
            # Record provenance for a deterministic subset (e.g., every 100th active patch)
            if c > 0 and len(prov_records) < 100:
                # Get hash of residual
                res_bytes = final_res[i].tobytes()
                hsh = hashlib.sha256(res_bytes).hexdigest()[:8]
                prov_records.append({
                    "target_x": int(tile_x + cx),
                    "target_y": int(tile_y + cy),
                    "target_scale": scale_factor,
                    "source_scale": float(dictionary.provenance[indices[i,0], 0]),
                    "source_x": float(dictionary.provenance[indices[i,0], 1]),
                    "source_y": float(dictionary.provenance[indices[i,0], 2]),
                    "match_distance": float(distances[i,0]),
                    "confidence": float(c),
                    "residual_energy": float(res_energy[i]),
                    "residual_hash": hsh
                })

            if c > 0:
                confidence_map[cy:cy+patch_size, cx:cx+patch_size] = np.maximum(
                    confidence_map[cy:cy+patch_size, cx:cx+patch_size], c
                )
    
    # Fill in inactive areas with weight so division doesn't fail
    # Inactive areas had 0 target_vars, meaning they are flat. The baseline is perfect.
    inactive_mask = ~active_mask
    if np.any(inactive_mask):
        inact_coords = coords[inactive_mask]
        for i in range(len(inact_coords)):
            cx, cy = inact_coords[i]
            weight_accum[cy:cy+patch_size, cx:cx+patch_size] += window
            
    return output_accum, weight_accum, confidence_map, total_active, accepted_count, prov_records

def reconstruct_overlap_add(target_baseline, dictionary, config, scale_factor, tile_size=256, tile_order="normal"):
    """
    Tiled overlap-add reconstruction.
    target_baseline is the direct Lanczos scaled linear image.
    """
    h, w = target_baseline.shape[:2]
    
    accum_res = np.zeros_like(target_baseline)
    accum_weight = np.zeros((h, w, 1), dtype=np.float32)
    conf_map = np.zeros((h, w), dtype=np.float32)
    
    pad = dictionary.patch_size
    
    all_prov = []
    total_act = 0
    total_acc = 0
    
    tiles = []
    for ty in range(0, h, tile_size):
        for tx in range(0, w, tile_size):
            tiles.append((tx, ty))
            
    if tile_order == "reverse":
        tiles.reverse()
    
    # Process in tiles, with padding to handle overlap
    for tx, ty in tiles:
        y1 = max(0, ty - pad)
        y2 = min(h, ty + tile_size + pad)
        x1 = max(0, tx - pad)
        x2 = min(w, tx + tile_size + pad)
        
        tile = target_baseline[y1:y2, x1:x2].copy()
        t_res, t_w, t_conf, t_act, t_acc, t_prov = process_tile(tile, dictionary, config, x1, y1, scale_factor, tx, ty, tile_size)
        
        # Add to global accumulators
        accum_res[y1:y2, x1:x2] += t_res
        accum_weight[y1:y2, x1:x2] += t_w
        conf_map[y1:y2, x1:x2] = np.maximum(conf_map[y1:y2, x1:x2], t_conf)
        
        total_act += t_act
        total_acc += t_acc
        all_prov.extend(t_prov)
            
    # Normalize
    safe_weight = np.maximum(accum_weight, 1e-6)
    final_res = accum_res / safe_weight
    
    # Add to baseline
    final_output = target_baseline + final_res
    final_output = np.clip(final_output, 0.0, 1.0)
    
    stats = {
        "active_patches": total_act,
        "accepted_patches": total_acc,
        "provenance": all_prov[:1000] # Limit provenance array size
    }
    
    return final_output, final_res, conf_map, stats
