import cv2
import numpy as np

def compute_perceptual_hash(img):
    # aHash implementation
    resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (8, 8))
    mean = np.mean(resized)
    return (resized > mean).flatten()

def hamming_distance(h1, h2):
    return np.count_nonzero(h1 != h2)

def compute_orb_geometric_inlier_ratio(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 4:
        return 0.0, len(matches)
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if mask is None:
        return 0.0, len(matches)
        
    inliers = np.sum(mask)
    return float(inliers) / len(matches), len(matches)
