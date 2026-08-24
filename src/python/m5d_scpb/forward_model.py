import numpy as np
import cv2

def forward_model(high_res, source_shape, scale_factor, base_sigma, config_multiplier=1.0):
    """
    Applies the mathematical camera forward model:
    1. Blur high_res using scale-adjusted PSF
    2. Area-downsample to original source dimensions
    
    source_shape: (h, w) of the original source
    scale_factor: enlargement factor (e.g. 2, 4, 8)
    base_sigma: PSF sigma estimated from the source image
    """
    # 1. Scale-adjusted PSF
    hr_sigma = base_sigma * scale_factor * config_multiplier
    
    # 2. Blur high_res
    k_size = int(np.ceil(hr_sigma * 3) * 2 + 1)
    if k_size < 3: k_size = 3
    if k_size % 2 == 0: k_size += 1
    
    kernel = cv2.getGaussianKernel(k_size, hr_sigma)
    kernel_2d = np.outer(kernel, kernel)
    
    # Apply blur with border replication
    blurred = cv2.filter2D(high_res, -1, kernel_2d, borderType=cv2.BORDER_REPLICATE)
    
    # 3. Area-downsample
    # To maintain exact coordinates and phase, we use INTER_AREA which corresponds to integrating over pixels
    sh, sw = source_shape[:2]
    downsampled = cv2.resize(blurred, (sw, sh), interpolation=cv2.INTER_AREA)
    
    return downsampled
