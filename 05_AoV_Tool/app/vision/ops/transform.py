
import cv2
import numpy as np
from typing import Dict

def op_flip(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    flip_code = int(params.get('flipCode', {}).get('default', 1))
    return cv2.flip(img, flip_code)

def op_rotate(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    code = int(params.get('rotateCode', {}).get('default', 0))
    cv2_codes = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    if 0 <= code < 3:
        return cv2.rotate(img, cv2_codes[code])
    return img

def op_perspective_transform(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    """
    透視變換 - 修正斜拍造成的變形
    """
    # 獲取原始四邊形頂點
    src_points = params.get('src_points', {}).get('default', [[100, 100], [400, 100], [400, 400], [100, 400]])
    dst_points = params.get('dst_points', {}).get('default', [[0, 0], [300, 0], [300, 300], [0, 300]])
    dsize = params.get('dsize', {}).get('default', [300, 300])
    
    # 轉換為 numpy array
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    
    # 計算變換矩陣
    M = cv2.getPerspectiveTransform(src, dst)
    
    # 執行透視變換
    return cv2.warpPerspective(img, M, tuple(dsize))
