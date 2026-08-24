import numpy as np
import cv2

def apply_blur(img: np.ndarray, blur_ksize: int, sigma: float) -> np.ndarray:
    if blur_ksize <= 1:
        return img
    # Apply gaussian blur
    return cv2.GaussianBlur(img, (blur_ksize, blur_ksize), sigma)

def downsample(img: np.ndarray, scale_factor: int) -> np.ndarray:
    if scale_factor <= 1:
        return img
    h, w = img.shape[:2]
    # Area interpolation for exact box filter downsampling
    return cv2.resize(img, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_AREA)

def forward_model(hr_img: np.ndarray, scale_factor: int, sigma: float, translation=(0.0, 0.0)) -> np.ndarray:
    """
    Applies translation, blur, and downsampling.
    translation is (dx, dy) in HR coordinates.
    """
    img = hr_img
    
    # 1. Warp if needed
    dx, dy = translation
    if dx != 0.0 or dy != 0.0:
        h, w = img.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        
    # 2. Blur
    if sigma > 0:
        blur_ksize = int(np.ceil(3 * sigma)) * 2 + 1
        img = apply_blur(img, blur_ksize, sigma)
        
    # 3. Downsample
    img = downsample(img, scale_factor)
    return img

def srgb_to_lin(img):
    img = img.astype(np.float32) / 255.0
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def lin_to_srgb(img):
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * (img ** (1 / 2.4)) - 0.055)
