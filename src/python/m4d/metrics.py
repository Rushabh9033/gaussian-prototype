import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import scipy.ndimage as ndimage

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(psnr_metric(img1, img2, data_range=255))

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    score = ssim_metric(img1, img2, data_range=255, channel_axis=-1)
    return float(score)

def get_edge_map(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges > 0

def edge_f1_score(true_img: np.ndarray, pred_img: np.ndarray, tolerance: int = 1) -> dict:
    true_edges = get_edge_map(true_img)
    pred_edges = get_edge_map(pred_img)
    
    # Documented small positional tolerance: dilate edges
    struct = ndimage.generate_binary_structure(2, 2)
    true_dilated = ndimage.binary_dilation(true_edges, structure=struct, iterations=tolerance)
    pred_dilated = ndimage.binary_dilation(pred_edges, structure=struct, iterations=tolerance)
    
    true_positives = np.logical_and(pred_edges, true_dilated).sum()
    false_positives = np.logical_and(pred_edges, ~true_dilated).sum()
    false_negatives = np.logical_and(true_edges, ~pred_dilated).sum()
    
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def compute_gradient_energy(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(sobelx**2 + sobely**2))

def compute_laplacian_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calculate_ringing_percentage(true_img: np.ndarray, pred_img: np.ndarray) -> float:
    """Calculate percentage of pixels in pred_img that overshoot/undershoot the true_img local neighborhood."""
    gray_true = cv2.cvtColor(true_img, cv2.COLOR_RGB2GRAY)
    gray_pred = cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local_min = cv2.erode(gray_true, kernel)
    local_max = cv2.dilate(gray_true, kernel)
    
    # Ringing/overshoot: pixels outside [local_min - threshold, local_max + threshold]
    threshold = 5
    overshoot = (gray_pred > local_max + threshold)
    undershoot = (gray_pred < local_min.astype(np.int16) - threshold)
    
    ringing_mask = np.logical_or(overshoot, undershoot)
    ringing_pct = ringing_mask.sum() / ringing_mask.size * 100.0
    return float(ringing_pct)
