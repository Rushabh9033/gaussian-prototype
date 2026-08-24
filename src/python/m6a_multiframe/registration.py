import numpy as np
import cv2

def estimate_translation(ref_img: np.ndarray, tgt_img: np.ndarray):
    """
    Estimates the (dx, dy) translation of tgt_img relative to ref_img using ECC.
    Both images are in LR space.
    Returns:
        dx, dy: Translation in LR pixels (such that warping ref by (dx, dy) gives tgt roughly)
        cc: Correlation coefficient (confidence)
    """
    if ref_img.ndim == 3:
        ref_gray = cv2.cvtColor(ref_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        tgt_gray = cv2.cvtColor(tgt_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    else:
        ref_gray = ref_img.astype(np.float32)
        tgt_gray = tgt_img.astype(np.float32)
        
    # Phase correlation for initial guess
    shift, _ = cv2.phaseCorrelate(tgt_gray, ref_gray)
    init_dx, init_dy = shift
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = init_dx
    warp_matrix[1, 2] = init_dy
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
    
    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_gray, tgt_gray, warp_matrix, 
            cv2.MOTION_TRANSLATION, criteria, None, 1
        )
        dx = warp_matrix[0, 2]
        dy = warp_matrix[1, 2]
        # Return dx, dy that aligns ref_gray TO tgt_gray
        return float(dx), float(dy), float(cc)
    except cv2.error:
        return float(init_dx), float(init_dy), 0.0

def register_frames(frames: list, ref_index: int = 0, cc_threshold: float = 0.9):
    """
    Registers all frames against the reference frame.
    Returns:
        estimated_translations: list of (dx, dy) in LR space.
        accepted_frames: list of boolean indicating if frame is accepted.
    """
    ref_img = frames[ref_index]
    estimated_translations = []
    accepted = []
    
    for i, frame in enumerate(frames):
        if i == ref_index:
            estimated_translations.append((0.0, 0.0))
            accepted.append(True)
            continue
            
        dx, dy, cc = estimate_translation(ref_img, frame)
        estimated_translations.append((dx, dy))
        accepted.append(cc >= cc_threshold)
        
    return estimated_translations, accepted
