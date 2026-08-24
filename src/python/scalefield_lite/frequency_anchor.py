import cv2
import numpy as np
from PIL import Image

def apply_frequency_anchor(parent_img: Image.Image, raw_child_img: Image.Image, zoom_ratio: float) -> Image.Image:
    """
    Cross-scale frequency anchoring:
    - Take the matching central region from the previous accepted scale.
    - Downsample the new raw child result to the comparison resolution.
    - Measure the low-frequency difference between it and the parent region.
    - Upsample that difference as a smooth correction field.
    - Apply only the low-frequency/color correction to the child.
    - Preserve the newly generated high-frequency detail.
    - Clamp and convert the result safely to sRGB.
    """
    if zoom_ratio <= 1.0:
        return raw_child_img.copy()

    # Convert to float numpy arrays
    parent_arr = np.array(parent_img).astype(np.float32) / 255.0
    child_arr = np.array(raw_child_img).astype(np.float32) / 255.0
    
    h, w, c = parent_arr.shape
    
    # Calculate central region of parent that corresponds to the child
    crop_w = int(w / zoom_ratio)
    crop_h = int(h / zoom_ratio)
    
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    
    parent_center = parent_arr[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Downsample child to match this region
    child_downsampled = cv2.resize(child_arr, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
    
    # Calculate difference
    diff = parent_center - child_downsampled
    
    # Upsample difference as a smooth correction field
    diff_upsampled = cv2.resize(diff, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Apply strong Gaussian blur to ensure it only contains low frequencies (smooth correction)
    # Kernel size should be substantial relative to the image size
    kernel_size = 31
    diff_smooth = cv2.GaussianBlur(diff_upsampled, (kernel_size, kernel_size), 0)
    
    # Apply correction to the child
    anchored_child = child_arr + diff_smooth
    
    # Clamp safely
    anchored_child = np.clip(anchored_child, 0.0, 1.0)
    
    return Image.fromarray((anchored_child * 255.0).astype(np.uint8))
