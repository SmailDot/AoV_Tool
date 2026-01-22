
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
