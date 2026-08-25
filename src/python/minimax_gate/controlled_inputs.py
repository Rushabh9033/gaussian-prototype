import cv2
import numpy as np
import os

def generate_controlled_inputs(gt_path, out_dir):
    """
    Deterministically downsample 360x220 ground truth photograph 
    using AREA interpolation/antialiasing to create controlled inputs.
    """
    gt = cv2.imread(gt_path)
    if gt is None:
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
        
    os.makedirs(out_dir, exist_ok=True)
    
    # 2x input: 180x110
    in_2x = cv2.resize(gt, (180, 110), interpolation=cv2.INTER_AREA)
    # 4x input: 90x55
    in_4x = cv2.resize(gt, (90, 55), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(os.path.join(out_dir, "input_2x.png"), in_2x)
    cv2.imwrite(os.path.join(out_dir, "input_4x.png"), in_4x)
    
    return {
        '2x': os.path.join(out_dir, "input_2x.png"),
        '4x': os.path.join(out_dir, "input_4x.png")
    }
