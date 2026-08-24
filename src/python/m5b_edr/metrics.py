import numpy as np
import cv2

def color_histogram_distance(img1, img2, bins=32):
    """
    Computes Bhattacharyya distance between color histograms of two images.
    """
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
        
    hist1 = cv2.calcHist([img1], [0,1,2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0,1,2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    
    cv2.normalize(hist1, hist1, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(hist2, hist2, alpha=1, norm_type=cv2.NORM_L1)
    
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))

def compute_edge_orientation_error(theta_gt, theta_pred, mask_gt):
    """
    Mean angular error (in radians) only where mask_gt > 0.
    Angle difference wraps at pi.
    """
    valid = mask_gt > 0
    if not np.any(valid):
        return 0.0
        
    diff = np.abs(theta_gt[valid] - theta_pred[valid])
    diff = np.minimum(diff, np.pi - diff)
    return float(np.mean(diff))

def compute_tile_seam_error(img, tile_size=512):
    """
    Measures unnatural jumps across tile boundaries.
    """
    h, w = img.shape[:2]
    errors = []
    
    # Horizontal seams (y = k * tile_size)
    for y in range(tile_size, h, tile_size):
        diff = np.abs(img[y, :] - img[y-1, :])
        errors.append(diff)
        
    # Vertical seams (x = k * tile_size)
    for x in range(tile_size, w, tile_size):
        diff = np.abs(img[:, x] - img[:, x-1])
        errors.append(diff)
        
    if not errors:
        return 0.0, 0.0
        
    all_err = np.concatenate([e.flatten() for e in errors])
    return float(np.max(all_err)), float(np.mean(all_err))
