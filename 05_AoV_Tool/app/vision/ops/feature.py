
import cv2
import numpy as np
from typing import Dict

def ensure_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def op_harris_corner(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    gray = ensure_gray(img)
    gray = np.float32(gray)
    block_size = int(params.get('blockSize', {}).get('default', 2))
    ksize = int(params.get('ksize', {}).get('default', 3))
    k = float(params.get('k', {}).get('default', 0.04))
    
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    
    output = img.copy()
    dst = cv2.dilate(dst, None)
    output[dst > 0.01 * dst.max()] = [0, 0, 255]
    return output

def op_fast_detector(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    gray = ensure_gray(img)
    threshold = int(params.get('threshold', {}).get('default', 10))
    nonmax = bool(params.get('nonmaxSuppression', {}).get('default', True))
    
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax)
    keypoints = fast.detect(gray, None)
    
    return cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))
