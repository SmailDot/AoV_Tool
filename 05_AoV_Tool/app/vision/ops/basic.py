
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
    return cv2.resize(img, (w, h))

def op_bgr2gray(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def op_bgr2hsv(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
