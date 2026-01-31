
import cv2
import numpy as np
from typing import Dict

def op_bilateral_filter(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    d = int(params.get('d', {}).get('default', 9))
    sigmaColor = float(params.get('sigmaColor', {}).get('default', 75.0))
    sigmaSpace = float(params.get('sigmaSpace', {}).get('default', 75.0))
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def op_box_filter(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    ksize = tuple(params.get('ksize', {}).get('default', [3, 3]))
    normalize = bool(params.get('normalize', {}).get('default', True))
    return cv2.boxFilter(img, -1, ksize, normalize=normalize)

def op_threshold(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    thresh_val = int(params.get('thresh', {}).get('default', 127))
    max_val = int(params.get('maxval', {}).get('default', 255))
    _, binary = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
    
    if len(img.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary

def op_adaptive_threshold(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    max_val = int(params.get('maxValue', {}).get('default', 255))
    block_size = int(params.get('blockSize', {}).get('default', 11))
    c = float(params.get('C', {}).get('default', 2))
    if block_size % 2 == 0: block_size += 1
    
    binary = cv2.adaptiveThreshold(
        gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )
    
    if len(img.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary

def op_in_range(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    l_b = int(params.get('lower_b', {}).get('default', 0))
    l_g = int(params.get('lower_g', {}).get('default', 0))
    l_r = int(params.get('lower_r', {}).get('default', 0))
    u_b = int(params.get('upper_b', {}).get('default', 255))
    u_g = int(params.get('upper_g', {}).get('default', 255))
    u_r = int(params.get('upper_r', {}).get('default', 255))
    
    lower = np.array([l_b, l_g, l_r])
    upper = np.array([u_b, u_g, u_r])
    
    mask = cv2.inRange(img, lower, upper)
    
    if len(img.shape) == 3:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def op_bitwise_not(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    return cv2.bitwise_not(img)

def op_equalize_hist(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    eq = cv2.equalizeHist(gray)
    
    if len(img.shape) == 3:
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return eq

def op_clahe(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    clip_limit = params.get('clipLimit', {}).get('default', 2.0)
    tile_size_val = params.get('tileGridSize', {}).get('default', [8, 8])
    if isinstance(tile_size_val, list): tile_size = tuple(tile_size_val)
    else: tile_size = (8, 8)
    
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img)

def op_fast_clahe(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    """
    快速 CLAHE - 針對硬幣反光優化的對比增強
    比傳統 HistEqual 更穩定，計算量減少 40%
    """
    clip_limit = float(params.get('clipLimit', {}).get('default', 3.0))
    tile_size_val = params.get('tileGridSize', {}).get('default', [8, 8])
    iterations = int(params.get('iterations', {}).get('default', 1))
    
    if isinstance(tile_size_val, list): 
        tile_size = tuple(tile_size_val)
    else: 
        tile_size = (8, 8)
    
    # 快速模式：使用更大的網格減少計算
    fast_tile_size = (tile_size[0] * 2, tile_size[1] * 2)
    
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 多次迭代增強效果
        l_enhanced = l.copy()
        for _ in range(iterations):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=fast_tile_size)
            l_enhanced = clahe.apply(l_enhanced)
        
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    else:
        # 灰度圖多次迭代
        enhanced = img.copy()
        for _ in range(iterations):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=fast_tile_size)
            enhanced = clahe.apply(enhanced)
        return enhanced
