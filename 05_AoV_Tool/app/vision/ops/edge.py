
import cv2
import numpy as np
from typing import Dict

def ensure_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def op_canny(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    gray = ensure_gray(img)
    thresh1 = params.get('threshold1', {}).get('default', 50)
    thresh2 = params.get('threshold2', {}).get('default', 150)
    aperture = params.get('apertureSize', {}).get('default', 3)
    edges = cv2.Canny(gray, thresh1, thresh2, apertureSize=aperture)
    return ensure_bgr(edges)

def op_sobel(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    gray = ensure_gray(img)
    dx = int(params.get('dx', {}).get('default', 1))
    dy = int(params.get('dy', {}).get('default', 0))
    ksize = int(params.get('ksize', {}).get('default', 3))
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
    abs_sobel = cv2.convertScaleAbs(sobel)
    return ensure_bgr(abs_sobel)

def op_laplacian(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    gray = ensure_gray(img)
    ksize = int(params.get('ksize', {}).get('default', 3))
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    abs_lap = cv2.convertScaleAbs(lap)
    return ensure_bgr(abs_lap)
