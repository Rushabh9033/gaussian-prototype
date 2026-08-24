import cv2
import numpy as np
from scipy.interpolate import splprep

def compute_structure_tensor(luma: np.ndarray, sigma: float = 1.0):
    dx = cv2.Scharr(luma, cv2.CV_64F, 1, 0)
    dy = cv2.Scharr(luma, cv2.CV_64F, 0, 1)
    ixx = dx * dx
    iyy = dy * dy
    ixy = dx * dy
    if sigma > 0:
        ksize = int(2 * round(3 * sigma) + 1)
        ixx = cv2.GaussianBlur(ixx, (ksize, ksize), sigma)
        iyy = cv2.GaussianBlur(iyy, (ksize, ksize), sigma)
        ixy = cv2.GaussianBlur(ixy, (ksize, ksize), sigma)
    return ixx, iyy, ixy

def nms(magnitude, direction):
    # Quantize direction to 4 main angles
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180
    
    nms_out = np.zeros_like(magnitude)
    h, w = magnitude.shape
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            ang = angle[i, j]
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif (22.5 <= ang < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif (67.5 <= ang < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif (112.5 <= ang < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms_out[i, j] = magnitude[i, j]
            else:
                nms_out[i, j] = 0
    return nms_out

def extract_vector_edges(luma: np.ndarray, source_linear: np.ndarray, coherence_thresh: float = 0.5, strength_thresh: float = 0.1):
    ixx, iyy, ixy = compute_structure_tensor(luma)
    
    trace2 = (ixx + iyy) / 2.0
    det_term = np.sqrt(((ixx - iyy) / 2.0)**2 + ixy**2)
    l1 = trace2 + det_term
    l2 = trace2 - det_term
    
    denom = l1 + l2 + 1e-9
    coherence = ((l1 - l2) / denom)**2
    strength = l1 / (l1.max() + 1e-9)
    
    mask = (coherence > coherence_thresh) & (strength > strength_thresh)
    
    # Gradient direction for sampling
    dx = cv2.Sobel(luma, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(luma, cv2.CV_64F, 0, 1, ksize=3)
    direction = np.arctan2(dy, dx)
    mag = np.sqrt(dx**2 + dy**2)
    
    nms_mag = nms(mag, direction)
    final_mask = (mask & (nms_mag > 0)).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    accepted_curves = []
    rejected_count = 0
    
    h, w = luma.shape
    
    for cnt in contours:
        pts = cnt[:, 0, :]
        if len(pts) < 6:
            rejected_count += 1
            continue
            
        # Remove consecutive duplicates
        diffs = np.sum(np.abs(np.diff(pts, axis=0)), axis=1)
        valid = np.concatenate(([True], diffs > 0))
        pts = pts[valid]
        
        if len(pts) < 6:
            rejected_count += 1
            continue
            
        # Sub-pixel refinement and values
        sub_pts = []
        curve_strength = 0.0
        
        for p in pts:
            x, y = p
            # basic sub-pixel interpolation on mag
            if 0 < x < w-1 and 0 < y < h-1:
                # along x
                if direction[y, x] < np.pi/4 or direction[y, x] > 3*np.pi/4:
                    val_m1, val_0, val_p1 = mag[y, x-1], mag[y, x], mag[y, x+1]
                    denom_sp = 2 * (val_m1 - 2*val_0 + val_p1)
                    if denom_sp != 0:
                        x = x + (val_m1 - val_p1) / denom_sp
                # along y
                else:
                    val_m1, val_0, val_p1 = mag[y-1, x], mag[y, x], mag[y+1, x]
                    denom_sp = 2 * (val_m1 - 2*val_0 + val_p1)
                    if denom_sp != 0:
                        y = y + (val_m1 - val_p1) / denom_sp
                        
            sub_pts.append([x, y])
            curve_strength += strength[p[1], p[0]]
            
        curve_strength /= len(pts)
        sub_pts = np.array(sub_pts)
        
        try:
            tck, u = splprep([sub_pts[:, 0], sub_pts[:, 1]], s=1.0, k=3)
            # Evaluate fit error
            from scipy.interpolate import splev
            eval_x, eval_y = splev(u, tck)
            err = np.mean(np.sqrt((eval_x - sub_pts[:,0])**2 + (eval_y - sub_pts[:,1])**2))
            
            if err > 1.5:  # reject bad fits
                rejected_count += 1
                continue
                
            # Sample left/right colors using normals
            # Derivate of spline gives tangent
            dx_u, dy_u = splev(u, tck, der=1)
            norm_mag = np.sqrt(dx_u**2 + dy_u**2) + 1e-9
            nx = -dy_u / norm_mag
            ny = dx_u / norm_mag
            
            offset = 2.0
            lx = np.clip(sub_pts[:, 0] + nx * offset, 0, w-1).astype(np.int32)
            ly = np.clip(sub_pts[:, 1] + ny * offset, 0, h-1).astype(np.int32)
            rx = np.clip(sub_pts[:, 0] - nx * offset, 0, w-1).astype(np.int32)
            ry = np.clip(sub_pts[:, 1] - ny * offset, 0, h-1).astype(np.int32)
            
            left_color = np.mean(source_linear[ly, lx], axis=0).tolist()
            right_color = np.mean(source_linear[ry, rx], axis=0).tolist()
            
            avg_dir = np.mean(direction[pts[:,1], pts[:,0]])
            
            # Serialize knots and coeffs for JSON
            tck_serializable = (tck[0].tolist(), [c.tolist() for c in tck[1]], int(tck[2]))
            
            accepted_curves.append({
                "tck": tck_serializable,
                "confidence": float(curve_strength),
                "fitting_error": float(err),
                "edge_strength": float(curve_strength),
                "edge_direction": float(avg_dir),
                "left_color": left_color,
                "right_color": right_color
            })
            
        except Exception:
            rejected_count += 1
            
    return accepted_curves, final_mask, rejected_count
