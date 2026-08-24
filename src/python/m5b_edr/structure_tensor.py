import cv2
import numpy as np

def compute_structure_tensor(luma: np.ndarray, sigma: float = 1.0):
    """
    Computes the smoothed structure tensor of a luminance image.
    Uses Scharr derivatives for better rotational invariance than Sobel.
    """
    dx = cv2.Scharr(luma, cv2.CV_64F, 1, 0)
    dy = cv2.Scharr(luma, cv2.CV_64F, 0, 1)
    
    ixx = dx * dx
    iyy = dy * dy
    ixy = dx * dy
    
    if sigma > 0:
        ksize = int(2 * round(3 * sigma) + 1)
        ixx = cv2.GaussianBlur(ixx, (ksize, ksize), sigma)
        iyy = cv2.GaussianBlur(iyy, (ksize, ksize), sigma)
        ixy = cv2.GaussianBlur(ixy, (ksize, ksize), sigma)
        
    return ixx, iyy, ixy

def extract_edge_orientation(luma: np.ndarray, sigma: float = 1.0):
    """
    Extracts edge direction, coherence, and uncertainty.
    """
    ixx, iyy, ixy = compute_structure_tensor(luma, sigma)
    
    trace = ixx + iyy
    det = ixx * iyy - ixy * ixy
    
    # Eigenvalues
    # l1 = (trace + sqrt(trace^2 - 4*det))/2
    # l2 = (trace - sqrt(trace^2 - 4*det))/2
    diff = np.sqrt(np.maximum((ixx - iyy)**2 + 4 * ixy**2, 0))
    l1 = (trace + diff) / 2.0
    l2 = (trace - diff) / 2.0
    
    # Coherence
    denom = l1 + l2 + 1e-9
    coherence = ((l1 - l2) / denom)**2
    
    # Uncertainty (corner/junction measure based on l2)
    # High l2 means gradient energy in multiple directions
    uncertainty = l2 / (l1.max() + 1e-9)
    uncertainty = np.clip(uncertainty * 5.0, 0, 1) # Scale to [0,1]
    
    # Dominant orientation of the gradient (normal to edge)
    # Eigenvector for l1: (l1 - iyy, ixy)
    vec_x = l1 - iyy
    vec_y = ixy
    
    # Tangent to edge
    tangent_x = -vec_y
    tangent_y = vec_x
    
    mag = np.sqrt(tangent_x**2 + tangent_y**2) + 1e-9
    tangent_x /= mag
    tangent_y /= mag
    
    theta = np.arctan2(tangent_y, tangent_x)
    
    strength = l1 / (l1.max() + 1e-9)
    
    return {
        "theta": theta,
        "coherence": coherence,
        "uncertainty": uncertainty,
        "strength": strength
    }
