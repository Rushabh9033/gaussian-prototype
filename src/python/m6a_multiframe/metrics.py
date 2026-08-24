import numpy as np
import cv2


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    if img1.ndim == 3:
        ssims = []
        for i in range(img1.shape[2]):
            mu1 = cv2.filter2D(img1[:, :, i], -1, window)[5:-5, 5:-5]
            mu2 = cv2.filter2D(img2[:, :, i], -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1[:, :, i] ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2[:, :, i] ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1[:, :, i] * img2[:, :, i], -1, window)[5:-5, 5:-5] - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            )
            ssims.append(ssim_map.mean())
        return float(np.mean(ssims))
    else:
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(ssim_map.mean())


def get_edge_map(img):
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 255)).astype(np.uint8)
    return cv2.Canny(gray, 100, 200) > 0


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
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}


def compute_gradient_energy(img):
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        img = np.mean(img, axis=-1)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sum(gx ** 2 + gy ** 2))


def compute_laplacian_variance(img):
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255)).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def color_histogram_distance(img1, img2, bins=32):
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 255)).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (np.clip(img2, 0, 255)).astype(np.uint8)
    if img1.ndim == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    if img2.ndim == 2:
        img2 = np.stack([img2] * 3, axis=-1)
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [bins] * 3, [0, 256] * 3)
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [bins] * 3, [0, 256] * 3)
    cv2.normalize(hist1, hist1, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(hist2, hist2, alpha=1, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))


def calculate_ringing_percentage(true_img, pred_img):
    t = true_img.astype(np.float32)
    p = pred_img.astype(np.float32)
    if t.max() > 1.5:
        t /= 255.0
    if p.max() > 1.5:
        p /= 255.0
    if t.ndim == 3:
        t = np.mean(t, axis=-1)
    if p.ndim == 3:
        p = np.mean(p, axis=-1)
    gx_t = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
    gy_t = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
    mag_t = np.sqrt(gx_t ** 2 + gy_t ** 2)
    gx_p = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    mag_p = np.sqrt(gx_p ** 2 + gy_p ** 2)
    flat_mask = mag_t < 0.05
    if not np.any(flat_mask):
        return 0.0
    ringing_mask = (mag_p > 0.1) & flat_mask
    return float(np.sum(ringing_mask) / np.sum(flat_mask) * 100.0)


def compute_clipping_percentage(img):
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return float(np.mean((img <= 0.0) | (img >= 1.0)) * 100.0)


def registration_rmse(est_translations, true_translations):
    """RMSE between estimated and true translations in LR pixels."""
    errors = []
    for (ex, ey), (tx, ty) in zip(est_translations, true_translations):
        errors.append((ex - tx) ** 2 + (ey - ty) ** 2)
    return float(np.sqrt(np.mean(errors)))


def compute_all_metrics(gt_img, pred_img, pred_lin=None):
    """Compute all required metrics. gt_img and pred_img are uint8 RGB."""
    m = {}
    m['psnr'] = compute_psnr(gt_img, pred_img)
    m['ssim'] = compute_ssim(gt_img, pred_img)
    ef = edge_f1_score(gt_img, pred_img)
    m['edge_precision'] = ef['precision']
    m['edge_recall'] = ef['recall']
    m['edge_f1'] = ef['f1']
    m['gradient_energy'] = compute_gradient_energy(pred_img)
    m['laplacian_variance'] = compute_laplacian_variance(pred_img)
    m['color_histogram_distance'] = color_histogram_distance(gt_img, pred_img)
    m['ringing_percent'] = calculate_ringing_percentage(gt_img, pred_img)
    m['clipping_percent'] = compute_clipping_percentage(pred_img)
    return m
