import numpy as np
import cv2

def compute_regularization_energy(img, epsilon=0.01):
    """
    Computes Huber-like edge-aware TV energy.
    """
    gx = img[:, 1:] - img[:, :-1]
    gy = img[1:, :] - img[:-1, :]
    
    # Huber penalty: sqrt(x^2 + e^2) - e
    energy_x = np.mean(np.sqrt(gx**2 + epsilon**2) - epsilon)
    energy_y = np.mean(np.sqrt(gy**2 + epsilon**2) - epsilon)
    
    return energy_x + energy_y

def apply_regularization(img, reg_weight, step_size, epsilon=0.01, chroma_factor=0.2):
    """
    Applies a gradient descent step for the regularization term.
    """
    # Compute gradients
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    
    # Diffusivity (Huber weighting: 1 / sqrt(grad^2 + eps^2))
    mag_x = np.sqrt(gx**2 + epsilon**2)
    mag_y = np.sqrt(gy**2 + epsilon**2)
    
    flux_x = gx / mag_x
    flux_y = gy / mag_y
    
    # Divergence
    div = np.zeros_like(img)
    div[:, 1:] += flux_x[:, :-1]
    div[:, :-1] -= flux_x[:, :-1]
    div[1:, :] += flux_y[:-1, :]
    div[:-1, :] -= flux_y[:-1, :]
    
    # Chroma factor
    if img.ndim == 3 and img.shape[2] == 3:
        # Luma
        luma = np.dot(img, [0.2126, 0.7152, 0.0722])[..., np.newaxis]
        chroma = img - luma
        
        # Apply divergence
        img_new = img + step_size * reg_weight * div
        
        # Luma new
        luma_new = np.dot(img_new, [0.2126, 0.7152, 0.0722])[..., np.newaxis]
        chroma_new = img_new - luma_new
        
        # Weaken chroma correction
        final_chroma = chroma + chroma_factor * (chroma_new - chroma)
        final_img = luma_new + final_chroma
    else:
        final_img = img + step_size * reg_weight * div
        
    return final_img
