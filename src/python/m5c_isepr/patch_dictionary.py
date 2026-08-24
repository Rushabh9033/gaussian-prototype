import numpy as np
import cv2
from scipy.spatial import cKDTree

def extract_patches(img, patch_size=5, stride=1, global_offset_x=0, global_offset_y=0):
    """
    Extracts patches from an image. If the image is a tile from a larger image, 
    global offsets ensure the local extraction aligns with the global stride grid.
    """
    h, w = img.shape[:2]
    
    start_x = (stride - (global_offset_x % stride)) % stride
    start_y = (stride - (global_offset_y % stride)) % stride
    
    num_y = max(0, (h - start_y - patch_size) // stride + 1)
    num_x = max(0, (w - start_x - patch_size) // stride + 1)
    
    if num_y <= 0 or num_x <= 0:
        channels = img.shape[2] if img.ndim == 3 else 1
        return np.zeros((0, patch_size, patch_size, channels), dtype=img.dtype), np.zeros((0, 2), dtype=int)
        
    patches = np.zeros((num_y * num_x, patch_size, patch_size, img.shape[-1] if img.ndim == 3 else 1), dtype=img.dtype)
    coords = np.zeros((num_y * num_x, 2), dtype=int)
    
    idx = 0
    for y in range(start_y, h - patch_size + 1, stride):
        for x in range(start_x, w - patch_size + 1, stride):
            p = img[y:y+patch_size, x:x+patch_size]
            if p.ndim == 2:
                p = p[..., np.newaxis]
            patches[idx] = p
            coords[idx] = [x, y]
            idx += 1
            
    return patches, coords

def compute_descriptor(patch):
    """
    Deterministic non-learned descriptor.
    patch: (N, P, P, 1) or (N, P, P)
    Features: normalized luminance, mean, std, gradient metrics.
    """
    if patch.ndim == 4:
        patch = patch[..., 0] # Use only single channel (luminance) for descriptor
        
    N, P, P2 = patch.shape
    flat = patch.reshape(N, P * P2)
    
    mean = np.mean(flat, axis=1, keepdims=True)
    std = np.std(flat, axis=1, keepdims=True) + 1e-6
    
    # Normalized pixels
    norm_pixels = (flat - mean) / std
    
    # Simple gradients over the patch
    dx = patch[:, :, 1:] - patch[:, :, :-1]
    dy = patch[:, 1:, :] - patch[:, :-1, :]
    
    # Mean absolute gradient
    grad_x_mean = np.mean(np.abs(dx), axis=(1,2)).reshape(N, 1)
    grad_y_mean = np.mean(np.abs(dy), axis=(1,2)).reshape(N, 1)
    
    # Combine (weighting can be documented here: pixels=1.0, mean=2.0, std=2.0, grad=1.0)
    # We want mean and std to match to prevent copying textures across unrelated brightness levels
    descriptor = np.concatenate([
        norm_pixels, 
        mean * 2.0, 
        std * 2.0, 
        grad_x_mean, 
        grad_y_mean
    ], axis=1)
    
    return descriptor

class PatchDictionary:
    def __init__(self, patch_size=5, stride=2):
        self.patch_size = patch_size
        self.stride = stride
        self.descriptors = []
        self.residuals = []
        self.provenance = []
        self.tree = None
        
    def add_scale_level(self, scale: float, high_img: np.ndarray, low_img: np.ndarray):
        high_luma = cv2.cvtColor(high_img, cv2.COLOR_RGB2GRAY) if high_img.shape[-1] == 3 else high_img
        low_luma = cv2.cvtColor(low_img, cv2.COLOR_RGB2GRAY) if low_img.shape[-1] == 3 else low_img
        
        # High and Low must be same shape
        patches_high, _ = extract_patches(high_img, self.patch_size, self.stride)
        patches_low, coords = extract_patches(low_img, self.patch_size, self.stride)
        patches_low_luma, _ = extract_patches(low_luma, self.patch_size, self.stride)
        
        desc = compute_descriptor(patches_low_luma)
        residuals = patches_high - patches_low
        
        # Filter flat patches to save space/time and avoid noise amplification
        var = np.var(patches_high, axis=(1,2,3))
        valid = var > 1e-5
        
        if np.any(valid):
            self.descriptors.append(desc[valid])
            self.residuals.append(residuals[valid])
            # Provenance: scale, x, y
            prov = np.zeros((np.sum(valid), 3), dtype=np.float32)
            prov[:, 0] = scale
            prov[:, 1:3] = coords[valid]
            self.provenance.append(prov)
            
    def build_tree(self):
        if not self.descriptors:
            return
        self.descriptors = np.vstack(self.descriptors)
        self.residuals = np.vstack(self.residuals)
        self.provenance = np.vstack(self.provenance)
        self.tree = cKDTree(self.descriptors)
        
    def search(self, target_luma_patches, k=5):
        if self.tree is None:
            N = len(target_luma_patches)
            return np.zeros((N, k)), np.zeros((N, k), dtype=int)
            
        desc = compute_descriptor(target_luma_patches)
        distances, indices = self.tree.query(desc, k=k, workers=-1)
        
        # Ensure outputs are 2D even if k=1
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
            
        return distances, indices
