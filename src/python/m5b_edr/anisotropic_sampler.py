import cv2
import numpy as np

def sample_anisotropic(source_rgb: np.ndarray, x_in: np.ndarray, y_in: np.ndarray,
                       theta_map: np.ndarray, coherence_map: np.ndarray, uncertainty_map: np.ndarray,
                       config: dict, k_radius: int = 2):
    """
    Vectorized anisotropic elliptical sampler.
    For each sub-pixel mapped coordinate in x_in, y_in, gathers an area of (2*k+1)^2 from the source,
    computes weights based on edge orientation, and returns the blended result.
    """
    src_h, src_w, channels = source_rgb.shape
    out_shape = x_in.shape
    
    # Extract tensor properties at mapped coordinates using bilinear interpolation
    theta_sub = cv2.remap(theta_map, x_in, y_in, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    coh_sub = cv2.remap(coherence_map, x_in, y_in, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    unc_sub = cv2.remap(uncertainty_map, x_in, y_in, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Base center color (used for photometric penalty)
    center_color = cv2.remap(source_rgb, x_in, y_in, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    cos_t = np.cos(theta_sub)
    sin_t = np.sin(theta_sub)
    
    # Parameters
    t_scale = config["tangent_scale"]
    n_scale = config["normal_scale"]
    coh_thresh = config["coherence_thresh"]
    phot_sigma = config["photometric_sigma"]
    
    # Dynamic sigmas based on coherence and uncertainty
    # High coherence -> highly anisotropic. Low coherence or high uncertainty -> isotropic fallback.
    is_edge = np.clip((coh_sub - coh_thresh) / (1.0 - coh_thresh + 1e-5), 0, 1)
    is_edge = is_edge * (1.0 - unc_sub)
    
    sigma_t = 1.0 * (1.0 - is_edge) + t_scale * is_edge
    sigma_n = 1.0 * (1.0 - is_edge) + n_scale * is_edge
    
    inv_var_t = 1.0 / (2.0 * sigma_t**2 + 1e-5)
    inv_var_n = 1.0 / (2.0 * sigma_n**2 + 1e-5)
    inv_var_p = 1.0 / (2.0 * phot_sigma**2 + 1e-5)
    
    accum_color = np.zeros((*out_shape, channels), dtype=np.float32)
    accum_weight = np.zeros(out_shape, dtype=np.float32)
    
    x_in_floor = np.floor(x_in).astype(np.int32)
    y_in_floor = np.floor(y_in).astype(np.int32)
    
    # Iterate over the KxK neighborhood
    for dy in range(-k_radius, k_radius + 1):
        for dx in range(-k_radius, k_radius + 1):
            xs = x_in_floor + dx
            ys = y_in_floor + dy
            
            # Map sample coordinates safely with border reflection
            xs_safe = np.abs(xs)
            xs_safe = np.minimum(xs_safe, 2 * src_w - 2 - xs_safe)
            ys_safe = np.abs(ys)
            ys_safe = np.minimum(ys_safe, 2 * src_h - 2 - ys_safe)
            
            # Gather neighbor colors
            # (using advanced indexing, but remap with nearest is faster/cleaner for grid)
            xs_safe_f = xs_safe.astype(np.float32)
            ys_safe_f = ys_safe.astype(np.float32)
            c_s = cv2.remap(source_rgb, xs_safe_f, ys_safe_f, interpolation=cv2.INTER_NEAREST)
            
            # Spatial distance
            dist_x = xs - x_in
            dist_y = ys - y_in
            
            d_t = dist_x * cos_t + dist_y * sin_t
            d_n = -dist_x * sin_t + dist_y * cos_t
            
            w_spatial = np.exp(-(d_t**2 * inv_var_t + d_n**2 * inv_var_n))
            
            # Photometric distance (in RGB space)
            color_diff2 = np.sum((c_s - center_color)**2, axis=-1)
            w_photo = np.exp(-color_diff2 * inv_var_p)
            
            w_total = w_spatial * w_photo
            
            accum_color += c_s * w_total[..., np.newaxis]
            accum_weight += w_total
            
    # Normalize safely
    accum_weight = np.maximum(accum_weight, 1e-9)
    final_color = accum_color / accum_weight[..., np.newaxis]
    
    return final_color
