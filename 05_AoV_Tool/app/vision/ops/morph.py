
import cv2
import numpy as np
from typing import Dict

def op_dilate(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    iterations = int(params.get('iterations', {}).get('default', 1))
    k_val = params.get('kernel', {}).get('default', [3, 3])
    if isinstance(k_val, list): k_val = tuple(k_val)
    else: k_val = (3, 3)
    kernel = np.ones(k_val, np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)

def op_erode(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    iterations = int(params.get('iterations', {}).get('default', 1))
    k_size = int(params.get('kernel_size', {}).get('default', 3))
    kernel = np.ones((k_size, k_size), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)

def op_morph_open(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    k_size = int(params.get('kernel_size', {}).get('default', 5))
    iterations = int(params.get('iterations', {}).get('default', 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

def op_morph_close(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    k_size = int(params.get('kernel_size', {}).get('default', 5))
    iterations = int(params.get('iterations', {}).get('default', 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def op_morph_gradient(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    k_size = int(params.get('kernel_size', {}).get('default', 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
