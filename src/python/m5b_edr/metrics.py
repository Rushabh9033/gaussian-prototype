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

def compute_edge_orientation_error(theta_gt, theta_pred, strength_gt, threshold=0.1):
    """
    Mean angular error (in radians and degrees) only where ground-truth edge strength exceeds a documented threshold.
    Angle difference wraps at pi because edge orientation is periodic over pi (not 2pi).
    """
    valid = strength_gt > threshold
    if not np.any(valid):
        return 0.0, 0.0
        
    diff = np.abs(theta_gt[valid] - theta_pred[valid])
    diff = np.minimum(diff, np.pi - diff)
    mean_rad = float(np.mean(diff))
    mean_deg = float(np.degrees(mean_rad))
    return mean_rad, mean_deg

def compute_tiling_equivalence_error(untiled_img, tiled_img):
    """
    Genuine tiling-equivalence metric.
    Converts to float32 before subtraction to avoid uint8 wrap-around.
    Returns max absolute difference and mean absolute difference.
    """
    # Prevent uint8 wrap (e.g. 255 - 0 = 255 instead of 1)
    diff = np.abs(untiled_img.astype(np.float32) - tiled_img.astype(np.float32))
    return float(np.max(diff)), float(np.mean(diff))
