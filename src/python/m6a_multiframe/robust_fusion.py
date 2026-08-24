import numpy as np
import cv2


def shift_and_add(frames, translations, scale_factor):
    """
    Simple shift-and-add fusion baseline.
    Places each LR frame onto a HR grid according to its estimated translation,
    then averages overlapping contributions.
    
    translations: list of (dx, dy) in LR space
    """
    ref = frames[0]
    lr_h, lr_w = ref.shape[:2]
    hr_h, hr_w = lr_h * scale_factor, lr_w * scale_factor
    channels = ref.shape[2] if ref.ndim == 3 else 1
    
    accum = np.zeros((hr_h, hr_w, channels), dtype=np.float64)
    count = np.zeros((hr_h, hr_w, 1), dtype=np.float64)
    
    for frame, (dx, dy) in zip(frames, translations):
        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]
        # dx, dy are in LR space; convert to HR space
        hr_dx = dx * scale_factor
        hr_dy = dy * scale_factor
        
        # Upsample the frame to HR grid using nearest
        upsampled = cv2.resize(frame, (hr_w, hr_h), interpolation=cv2.INTER_NEAREST)
        if upsampled.ndim == 2:
            upsampled = upsampled[:, :, np.newaxis]
        
        # Warp to compensate for the translation
        M = np.float32([[1, 0, -hr_dx], [0, 1, -hr_dy]])
        for c in range(channels):
            accum[:, :, c] += cv2.warpAffine(
                upsampled[:, :, c], M, (hr_w, hr_h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        mask = cv2.warpAffine(
            np.ones((hr_h, hr_w), dtype=np.float32), M, (hr_w, hr_h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        count[:, :, 0] += mask
    
    count = np.maximum(count, 1e-8)
    result = (accum / count).astype(np.float32)
    if channels == 1:
        result = result[:, :, 0]
    return result


def robust_fusion(frames, translations, scale_factor, num_iters=3):
    """
    Robust median-based fusion onto HR grid.
    Places all frames, then takes weighted median at each pixel.
    Falls back to mean when insufficient samples.
    """
    ref = frames[0]
    lr_h, lr_w = ref.shape[:2]
    hr_h, hr_w = lr_h * scale_factor, lr_w * scale_factor
    channels = ref.shape[2] if ref.ndim == 3 else 1
    
    # Collect all warped contributions
    warped_stack = []
    mask_stack = []
    
    for frame, (dx, dy) in zip(frames, translations):
        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]
        hr_dx = dx * scale_factor
        hr_dy = dy * scale_factor
        
        upsampled = cv2.resize(frame, (hr_w, hr_h), interpolation=cv2.INTER_LANCZOS4)
        if upsampled.ndim == 2:
            upsampled = upsampled[:, :, np.newaxis]
        
        M = np.float32([[1, 0, -hr_dx], [0, 1, -hr_dy]])
        warped = np.zeros_like(upsampled)
        for c in range(channels):
            warped[:, :, c] = cv2.warpAffine(
                upsampled[:, :, c], M, (hr_w, hr_h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        mask = cv2.warpAffine(
            np.ones((hr_h, hr_w), dtype=np.float32), M, (hr_w, hr_h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        warped_stack.append(warped)
        mask_stack.append(mask)
    
    # Stack and compute weighted median per-pixel
    n = len(warped_stack)
    stack = np.stack(warped_stack, axis=0)  # (N, H, W, C)
    masks = np.stack(mask_stack, axis=0)    # (N, H, W)
    
    # Use Huber-weighted mean: iteratively reweight
    result = np.mean(stack, axis=0)  # initial estimate
    
    for _ in range(num_iters):
        residuals = stack - result[np.newaxis, :, :, :]  # (N, H, W, C)
        abs_res = np.abs(residuals)
        
        # Huber threshold
        delta = 0.05
        weights = np.where(abs_res < delta, 1.0, delta / (abs_res + 1e-8))
        # Zero out contributions where mask is low
        mask_weights = masks[:, :, :, np.newaxis]  # (N, H, W, 1)
        weights = weights * mask_weights
        
        w_sum = np.sum(weights, axis=0)
        w_sum = np.maximum(w_sum, 1e-8)
        result = np.sum(weights * stack, axis=0) / w_sum
    
    result = np.clip(result, 0, 1).astype(np.float32)
    if channels == 1:
        result = result[:, :, 0]
        
    # Coverage map: fraction of frames contributing to each pixel
    coverage = np.mean(masks > 0.5, axis=0).astype(np.float32)
    
    # Confidence map: inverse of variance at each pixel
    var = np.var(stack, axis=0).mean(axis=-1) if channels > 1 else np.var(stack[:, :, :, 0], axis=0)
    confidence = 1.0 / (var + 1e-6)
    confidence = confidence / (confidence.max() + 1e-8)
    
    return result, coverage, confidence
