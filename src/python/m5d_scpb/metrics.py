import numpy as np
import cv2

def color_histogram_distance(img1, img2, bins=32):
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
    hist1 = cv2.calcHist([img1], [0,1,2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0,1,2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1, hist1, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(hist2, hist2, alpha=1, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))

def compute_tiling_equivalence_error(untiled_img, tiled_img):
    diff = np.abs(untiled_img.astype(np.float32) - tiled_img.astype(np.float32))
    return float(np.max(diff)), float(np.mean(diff))

def get_edge_map(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    edges = cv2.Canny(gray, 100, 200)
    return edges > 0

def edge_f1_score(true_img, pred_img, tolerance=1):
    import scipy.ndimage as ndimage
    true_edges = get_edge_map(true_img)
    pred_edges = get_edge_map(pred_img)
    
    struct = ndimage.generate_binary_structure(2, 2)
    true_dilated = ndimage.binary_dilation(true_edges, structure=struct, iterations=tolerance)
    pred_dilated = ndimage.binary_dilation(pred_edges, structure=struct, iterations=tolerance)
    
    matched_pred = np.logical_and(pred_edges, true_dilated).sum()
    matched_true = np.logical_and(true_edges, pred_dilated).sum()
    
    pred_total = pred_edges.sum()
    true_total = true_edges.sum()
    
    precision = matched_pred / pred_total if pred_total > 0 else 0.0
    recall = matched_true / true_total if true_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def calculate_ringing_percentage(true_img, pred_img):
    if true_img.dtype != np.float32:
        true_img = true_img.astype(np.float32) / 255.0
    if pred_img.dtype != np.float32:
        pred_img = pred_img.astype(np.float32) / 255.0
        
    if true_img.ndim == 3:
        true_img = np.mean(true_img, axis=-1)
    if pred_img.ndim == 3:
        pred_img = np.mean(pred_img, axis=-1)
        
    gx_t = cv2.Sobel(true_img, cv2.CV_32F, 1, 0, ksize=3)
    gy_t = cv2.Sobel(true_img, cv2.CV_32F, 0, 1, ksize=3)
    mag_t = np.sqrt(gx_t**2 + gy_t**2)
    
    gx_p = cv2.Sobel(pred_img, cv2.CV_32F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(pred_img, cv2.CV_32F, 0, 1, ksize=3)
    mag_p = np.sqrt(gx_p**2 + gy_p**2)
    
    flat_mask = mag_t < 0.05
    if not np.any(flat_mask):
        return 0.0
        
    ringing_mask = (mag_p > 0.1) & flat_mask
    return float(np.sum(ringing_mask) / np.sum(flat_mask) * 100.0)

def compute_laplacian_variance(img):
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def compute_gradient_energy(img):
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        img = np.mean(img, axis=-1)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sum(gx**2 + gy**2))
