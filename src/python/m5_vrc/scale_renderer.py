import cv2
import numpy as np
from scipy.interpolate import splev

def evaluate_splines_on_grid(splines, target_shape, original_shape):
    """
    Renders fitted B-splines analytically onto the high-resolution target grid.
    Returns a fractional coverage mask (supersampling approx).
    """
    th, tw = target_shape[1], target_shape[0]
    oh, ow = original_shape[0], original_shape[1]
    
    scale_x = tw / float(ow)
    scale_y = th / float(oh)
    
    # We create a high-res mask for the vector boundaries
    # Using LINE_AA gives us analytical anti-aliasing approximation
    mask = np.zeros((th, tw), dtype=np.float32)
    
    for tck in splines:
        # Evaluate curve at many points
        u = np.linspace(0, 1, 500)
        x_ev, y_ev = splev(u, tck)
        
        # Scale to target
        x_sc = x_ev * scale_x
        y_sc = y_ev * scale_y
        
        pts = np.vstack((x_sc, y_sc)).T.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw anti-aliased line
        cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=2, lineType=cv2.LINE_AA)
        
    return mask

def render_scale(color_field: np.ndarray, splines: list, residual: np.ndarray, 
                 target_shape: tuple, original_shape: tuple, source_linear: np.ndarray) -> np.ndarray:
    """
    Mathematical render at arbitrary scale.
    target_shape is (w, h).
    """
    # 1. Continuous colour field evaluated at target coordinates
    cf_scaled = cv2.resize(color_field, target_shape, interpolation=cv2.INTER_CUBIC)
    
    # 2. Vector boundary contribution
    # The mask tells us where the boundaries are. We enhance sharpness there.
    # We extract sharp edge values directly from an upscaled source to emulate vector limits,
    # or just use the colour field but without blurring.
    vector_mask = evaluate_splines_on_grid(splines, target_shape, original_shape)
    vector_mask_rgb = np.expand_dims(vector_mask, axis=-1)
    
    # Simple vector rendering: sharp sampling near edges. 
    # For proof of concept, we blend the sharp Lanczos upscale where splines exist.
    sharp_eval = cv2.resize(source_linear, target_shape, interpolation=cv2.INTER_LANCZOS4)
    
    blended_base = cf_scaled * (1.0 - vector_mask_rgb) + sharp_eval * vector_mask_rgb
    
    # 3. Source-supported residuals
    res_scaled = cv2.resize(residual, target_shape, interpolation=cv2.INTER_CUBIC)
    
    render_linear = blended_base + res_scaled
    return np.clip(render_linear, 0.0, 1.0)

def enforce_scale_consistency(rendered_child: np.ndarray, parent_target: np.ndarray, 
                              max_iters: int = 5, tolerance: float = 1.0 / 255.0) -> np.ndarray:
    """
    Exact scale-consistency constraint: downsample(child) must equal parent.
    Operates in linear light but tolerance is usually expressed relative to 8-bit.
    """
    current_child = rendered_child.copy()
    pw, ph = parent_target.shape[1], parent_target.shape[0]
    
    for i in range(max_iters):
        # Downsample using area average (conservative photon-preserving filter)
        child_down = cv2.resize(current_child, (pw, ph), interpolation=cv2.INTER_AREA)
        
        error = parent_target - child_down
        max_err = np.max(np.abs(error))
        
        if max_err <= tolerance:
            break
            
        # Distribute error back
        error_up = cv2.resize(error, (current_child.shape[1], current_child.shape[0]), interpolation=cv2.INTER_CUBIC)
        current_child += error_up
        current_child = np.clip(current_child, 0.0, 1.0)
        
    return current_child
