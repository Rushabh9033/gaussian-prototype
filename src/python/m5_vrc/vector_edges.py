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

def extract_vector_edges(luma: np.ndarray, coherence_thresh: float = 0.5, strength_thresh: float = 0.1):
    """
    Extracts vector boundaries from luminance using the structure tensor.
    Returns:
        splines: List of B-spline parameterizations (tck).
        mask: The deterministic thresholded edge mask.
    """
    ixx, iyy, ixy = compute_structure_tensor(luma)
    
    trace2 = (ixx + iyy) / 2.0
    det_term = np.sqrt(((ixx - iyy) / 2.0)**2 + ixy**2)
    l1 = trace2 + det_term
    l2 = trace2 - det_term
    
    denom = l1 + l2 + 1e-9
    coherence = ((l1 - l2) / denom)**2
    strength = l1 / (l1.max() + 1e-9)
    
    mask = (coherence > coherence_thresh) & (strength > strength_thresh)
    mask_u8 = (mask * 255).astype(np.uint8)
    
    # Thin edges
    kernel = np.ones((3,3), np.uint8)
    skeleton = cv2.ximgproc.thinning(mask_u8) if hasattr(cv2, 'ximgproc') else mask_u8
    
    # In case ximgproc is not available, we use a basic erosion/skeleton or just the mask
    # Actually cv2.findContours on a thin mask is better
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    splines = []
    for cnt in contours:
        if len(cnt) > 5:
            pts = cnt[:, 0, :]
            # Remove consecutive duplicates
            diffs = np.sum(np.abs(np.diff(pts, axis=0)), axis=1)
            valid = np.concatenate(([True], diffs > 0))
            pts = pts[valid]
            if len(pts) > 5:
                x = pts[:, 0]
                y = pts[:, 1]
                try:
                    tck, _ = splprep([x, y], s=1.0, k=3)
                    splines.append(tck)
                except Exception:
                    pass
                    
    return splines, skeleton
