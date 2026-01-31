
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

def op_hu_moments(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    """
    Hu 矩計算 - 判斷形狀是否為圓形的低功耗指標
    """
    gray = ensure_gray(img)
    
    # 獲取參數
    threshold_roundness = float(params.get('threshold_roundness', {}).get('default', 0.9))
    visualize = bool(params.get('visualize', {}).get('default', True))
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 找輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 準備輸出
    if len(img.shape) == 2:
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        output = img.copy()
    
    for i, cnt in enumerate(contours):
        if len(cnt) < 5:  # 至少需要 5 個點才能計算矩
            continue
            
        # 計算輪廓矩
        moments = cv2.moments(cnt)
        
        if moments['m00'] == 0:
            continue
        
        # 計算中心
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # 計算 Hu 矩
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Hu[0] 接近 0.16 表示圓形
        roundness = 1.0 / (1.0 + abs(hu_moments[0] - 0.16) * 100)
        is_circle = roundness > threshold_roundness
        
        if visualize:
            # 繪製輪廓
            color = (0, 255, 0) if is_circle else (0, 0, 255)
            cv2.drawContours(output, [cnt], -1, color, 2)
            
            # 繪製中心點
            cv2.circle(output, (cx, cy), 5, color, -1)
            
            # 顯示 Hu[0] 值
            label = f"Hu0: {hu_moments[0]:.4f}"
            cv2.putText(output, label, (cx - 40, cy - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 添加統計資訊
    total = len(contours)
    circles = sum(1 for i, cnt in enumerate(contours) if len(cnt) >= 5)
    info = f"Contours: {total}, Analyzed: {circles}"
    cv2.putText(output, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output
