
import cv2
import numpy as np
from typing import Dict

def op_watershed(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    """
    分水嶺分割 - 解決硬幣重疊、邊界模糊的終極手段
    """
    # 獲取參數
    use_markers = bool(params.get('use_markers', {}).get('default', True))
    marker_threshold = int(params.get('marker_threshold', {}).get('default', 127))
    min_marker_size = int(params.get('min_marker_size', {}).get('default', 10))
    
    # 1. 預處理 - 轉為灰度並模糊
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 2. 二值化（使用 OTSU）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. 形態學操作去除雜訊
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 4. 確定背景區域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 5. 距離變換找出前景中心
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # 6. 閾值化找出確定前景
    _, sure_fg = cv2.threshold(dist_transform, marker_threshold/255.0 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 7. 找出未知區域
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 8. 標記連通區域
    num_markers, markers = cv2.connectedComponents(sure_fg)
    
    # 9. 為所有標記加 1，背景不能是 0
    markers = markers + 1
    
    # 10. 未知區域標記為 0
    markers[unknown == 255] = 0
    
    # 11. 執行 Watershed
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()
    
    markers = cv2.watershed(img_color, markers)
    
    # 12. 視覺化結果
    output = img_color.copy()
    output[markers == -1] = [0, 0, 255]  # 邊界標記為紅色
    
    return output
