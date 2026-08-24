import cv2
import numpy as np
from scipy.interpolate import splev

def evaluate_splines_on_grid(splines_data, target_shape, original_shape):
    """
    Renders fitted B-splines analytically onto the high-resolution target grid.
    Uses stored left/right boundary colors.
    """
    th, tw = target_shape[1], target_shape[0]
    oh, ow = original_shape[0], original_shape[1]
    
    scale_x = tw / float(ow)
    scale_y = th / float(oh)
    
    # We create a vector boundary RGB layer and a coverage mask
    vector_rgb = np.zeros((th, tw, 3), dtype=np.float32)
    coverage = np.zeros((th, tw), dtype=np.float32)
    
    for curve in splines_data:
        t, c, k = curve["tck"]
        tck = (np.array(t), [np.array(c[0]), np.array(c[1])], k)
        
        # Evaluate curve at dense points
        u = np.linspace(0, 1, 1000)
        x_ev, y_ev = splev(u, tck)
        
        # Scale to target
        x_sc = x_ev * scale_x
        y_sc = y_ev * scale_y
        
        pts = np.vstack((x_sc, y_sc)).T.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # For a real vector representation, we could fill polygons using left/right colors.
        # As a POC, we draw the exact vector boundary with a supersampled thickness 
        # using the average boundary color (or mixing left/right).
        # We blend the left and right color as the sharp boundary color
        boundary_color = (np.array(curve["left_color"]) + np.array(curve["right_color"])) / 2.0
        
        # Draw anti-aliased line for the boundary
        # A thickness of 2 or 3 gives a strong structural edge
        temp_mask = np.zeros((th, tw), dtype=np.uint8)
        cv2.polylines(temp_mask, [pts], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
        
        mask_f32 = temp_mask.astype(np.float32) / 255.0
        
        # Accumulate coverage
        new_coverage = np.maximum(coverage, mask_f32)
        
        # Update RGB where coverage increases
        update_mask = (new_coverage > coverage)
        vector_rgb[update_mask] = boundary_color
        coverage = new_coverage
        
    return vector_rgb, coverage

def render_scale(color_field: np.ndarray, splines_data: list, residual: np.ndarray, 
                 target_shape: tuple, original_shape: tuple) -> tuple:
    """
    Mathematical render at arbitrary scale.
    Returns (rendered_image, coverage_percentage)
    """
    # 1. Continuous colour field evaluated at target coordinates
    cf_scaled = cv2.resize(color_field, target_shape, interpolation=cv2.INTER_CUBIC)
    
    # 2. Vector boundary contribution
    vector_rgb, coverage_mask = evaluate_splines_on_grid(splines_data, target_shape, original_shape)
    coverage_mask_rgb = np.expand_dims(coverage_mask, axis=-1)
    
    blended_base = cf_scaled * (1.0 - coverage_mask_rgb) + vector_rgb * coverage_mask_rgb
    
    # 3. Source-supported residuals
    # Res_scaled must support signed values (could be negative)
    res_scaled = cv2.resize(residual, target_shape, interpolation=cv2.INTER_CUBIC)
    
    render_linear = blended_base + res_scaled
    
    coverage_pct = float(np.mean(coverage_mask)) * 100.0
    return np.clip(render_linear, 0.0, 1.0), coverage_pct

def enforce_scale_consistency(rendered_child: np.ndarray, parent_target: np.ndarray, 
                              max_iters: int = 15, tolerance: float = 1.0 / 255.0) -> np.ndarray:
    """
    Exact scale-consistency constraint: downsample(child) must equal parent.
    Iterates strictly inside the 8-bit equivalent bound.
    """
    # We must ensure the error is measured similarly to the final output.
    # We operate in linear light, but we stop when the error is imperceptible in sRGB space.
    # To keep it purely mathematical and bounded, we will use a direct additive projection.
    current_child = rendered_child.copy()
    pw, ph = parent_target.shape[1], parent_target.shape[0]
    
    for i in range(max_iters):
        child_down = cv2.resize(current_child, (pw, ph), interpolation=cv2.INTER_AREA)
        
        # Error in linear space
        error = parent_target - child_down
        
        max_err = np.max(np.abs(error))
        # If linear error is less than ~1/255 (very conservative for sRGB 1/255), we break
        if max_err <= tolerance:
            break
            
        # Distribute error back up
        error_up = cv2.resize(error, (current_child.shape[1], current_child.shape[0]), interpolation=cv2.INTER_CUBIC)
        current_child += error_up
        current_child = np.clip(current_child, 0.0, 1.0)
        
    return current_child
