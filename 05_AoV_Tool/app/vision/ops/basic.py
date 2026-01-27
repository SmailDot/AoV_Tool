
import cv2
import numpy as np
from typing import Dict, Any

def op_gaussian_blur(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    ksize = tuple(params.get('ksize', {}).get('default', [5, 5]))
    sigmaX = params.get('sigmaX', {}).get('default', 0)
    sigmaY = params.get('sigmaY', {}).get('default', 0)
    return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)

def op_median_blur(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    ksize = int(params.get('ksize', {}).get('default', 5))
    if ksize % 2 == 0: ksize += 1
    return cv2.medianBlur(img, ksize)

def op_resize(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    w = int(params.get('width', {}).get('default', 640))
    h = int(params.get('height', {}).get('default', 480))
    
    # [Smart Resize]
    # If one dimension is 0 or -1, maintain aspect ratio
    current_h, current_w = img.shape[:2]
    
    # [Fix] Auto-Correction for Aspect Ratio Distortion
    # If user sets both w and h, but they cause significant distortion (>20%), 
    # and it looks like a "default" param set (e.g. 640x480 on a square image),
    # we force maintain aspect ratio based on width.
    if w > 0 and h > 0:
        target_ar = w / (h + 1e-6)
        src_ar = current_w / (current_h + 1e-6)
        distortion = abs(target_ar - src_ar) / src_ar
        
        # If distortion is high (>20%), assume user wants width priority and auto height
        # This is a heuristic to prevent "squashed" images from default params
        if distortion > 0.2:
            # print(f"[Resize] Detected distortion ({distortion:.2f}). Forcing auto-height.")
            h = 0 # Force auto calculation below

    # 如果使用者沒有指定（或指定為0/-1），則預設維持原圖尺寸，不進行任何縮放
    # 這是為了防止在沒有特別理由的情況下改變影像幾何結構
    if w <= 0 and h <= 0:
        return img 
        
    if w <= 0:
        ratio = h / current_h
        w = int(current_w * ratio)
    elif h <= 0:
        ratio = w / current_w
        h = int(current_h * ratio)
        
    return cv2.resize(img, (w, h))

def op_bgr2gray(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def op_bgr2hsv(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
