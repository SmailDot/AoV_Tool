
import cv2
import numpy as np
import os
from typing import Dict

def ensure_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Any

# --- Helper Classes for Coin Logic (Ported from catch_coins) ---

@dataclass
class Circle:
    x: int
    y: int
    radius: int

    def distance_to(self, other: 'Circle') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def radius_difference(self, other: 'Circle') -> int:
        return abs(self.radius - other.radius)

    def is_similar_to(self, other: 'Circle', max_center_distance: int = 30, max_radius_diff: int = 15) -> bool:
        return (self.distance_to(other) < max_center_distance and
                self.radius_difference(other) < max_radius_diff)

    @staticmethod
    def average(circles: List['Circle']) -> 'Circle':
        if not circles: return Circle(0,0,0)
        avg_x = int(np.mean([c.x for c in circles]))
        avg_y = int(np.mean([c.y for c in circles]))
        avg_r = int(np.mean([c.radius for c in circles]))
        return Circle(avg_x, avg_y, avg_r)

class StabilityFilter:
    def __init__(self, buffer_size: int = 10, min_occurrences: int = 6):
        self._buffer = deque(maxlen=buffer_size)
        self._min_occurrences = min_occurrences

    def add_detection(self, circles: List[Circle]):
        self._buffer.append(circles)

    def get_stable_circles(self) -> List[Circle]:
        if not self._buffer: return []
        
        # Flatten with frame tags
        tagged = []
        for frame_idx, circles in enumerate(self._buffer):
            for c in circles:
                tagged.append((c, frame_idx))
        
        if not tagged: return []

        # Grouping
        groups = []
        used = set()
        for i, (c_i, tag_i) in enumerate(tagged):
            if i in used: continue
            group = [(c_i, tag_i)]
            used.add(i)
            for j, (c_j, tag_j) in enumerate(tagged):
                if j in used: continue
                if any(c_j.is_similar_to(gc) for gc, _ in group):
                    group.append((c_j, tag_j))
                    used.add(j)
            groups.append(group)

        # Filtering
        stable = []
        for group in groups:
            unique_frames = len(set(t for _, t in group))
            if unique_frames >= self._min_occurrences:
                circles = [c for c, _ in group]
                stable.append(Circle.average(circles))
        
        return stable

def op_advanced_coin_logic(img: np.ndarray, params: Dict, debug: bool, context: Dict = None) -> np.ndarray:
    """ 
    高階硬幣偵測邏輯 (完整移植版，支援 State Context)
    """
    # 1. 狀態管理 (State Persistence)
    # context 是由 Processor 傳入的字典，生命週期跨越多次 execute_pipeline
    if context is not None:
        state_key = "coin_stability_filter"
        if state_key not in context:
            # 初始化 Filter
            # 判斷是否為單張圖片模式 (Heuristic: 如果沒有 frame_id 或 frame_count < 2)
            # 在目前的 Tool 中，我們假設沒有 Video Stream 支援，所以預設為 Single Image Mode
            # 但為了未來擴充，我們可以看 context['is_video']
            
            is_video = context.get('is_video', False)
            buf_size = 10 if is_video else 1
            min_occ = 6 if is_video else 1
            
            context[state_key] = StabilityFilter(buffer_size=buf_size, min_occurrences=min_occ)
            if debug: print(f"[Coin] Init Filter (Video={is_video}, Buf={buf_size})")
            
        stability_filter = context[state_key]
    else:
        # Fallback: 無狀態模式 (每次都新建一個 size=1 的 filter)
        stability_filter = StabilityFilter(buffer_size=1, min_occurrences=1)

    # 2. 預處理 (Resize)
    max_size = int(params.get('max_image_size', {}).get('default', 800))
    h, w = img.shape[:2]
    longest_side = max(h, w)
    
    scale_factor = 1.0
    if longest_side > max_size:
        scale_factor = max_size / longest_side
        new_size = (int(w * scale_factor), int(h * scale_factor))
        img_processed = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    else:
        img_processed = img.copy()

    # 3. 偵測 (Hough)
    gray = ensure_gray(img_processed)
    blur_ksize = int(params.get('blur_ksize', {}).get('default', 13))
    if blur_ksize % 2 == 0: blur_ksize += 1
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    detected = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=float(params.get('dp', {}).get('default', 1.2)),
        minDist=int(params.get('min_distance', {}).get('default', 40)),
        param1=int(params.get('canny_threshold', {}).get('default', 50)),
        param2=int(params.get('accumulator_threshold', {}).get('default', 30)),
        minRadius=int(params.get('min_radius', {}).get('default', 20)),
        maxRadius=int(params.get('max_radius', {}).get('default', 80))
    )

    # Convert to Circle objects
    raw_circles = []
    if detected is not None:
        raw_circles = [Circle(int(x), int(y), int(r)) for x, y, r in np.round(detected[0, :])]

    # 4. 穩定性過濾 (Update State)
    stability_filter.add_detection(raw_circles)
    final_circles = stability_filter.get_stable_circles()

    # 5. 繪製結果
    output = img_processed.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    for c in final_circles:
        cv2.circle(output, (c.x, c.y), c.radius, (0, 255, 0), 2)
        cv2.circle(output, (c.x, c.y), 2, (0, 0, 255), 3)
        
    label = f"Stable: {len(final_circles)}"
    cv2.putText(output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return output

def op_hough_circles(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    gray = ensure_gray(img)
    dp = float(params.get('dp', {}).get('default', 1.2))
    minDist = int(params.get('minDist', {}).get('default', 30))
    param1 = int(params.get('param1', {}).get('default', 50))
    param2 = int(params.get('param2', {}).get('default', 30))
    minRadius = int(params.get('minRadius', {}).get('default', 0))
    maxRadius = int(params.get('maxRadius', {}).get('default', 0))
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp, minDist,
        param1=param1, param2=param2,
        minRadius=minRadius, maxRadius=maxRadius
    )
    
    output = img.copy()
    if circles is not None:
        circles_rounded = np.around(circles).astype(np.uint16)
        for i in circles_rounded[0, :]:
            cv2.circle(output, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
            cv2.circle(output, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)
    return output

def op_find_contours(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    gray = ensure_gray(img)
    mode_val = int(params.get('mode', {}).get('default', 1)) 
    method_val = int(params.get('method', {}).get('default', 2))
    min_area = int(params.get('min_area', {}).get('default', 100))
    
    contours, _ = cv2.findContours(gray, mode_val, method_val)
    
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    return output

def op_cascade_classifier(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    gray = ensure_gray(img)
    model_name = str(params.get('model_type', {}).get('default', 'haarcascade_frontalface_default.xml'))
    scale = float(params.get('scaleFactor', {}).get('default', 1.1))
    min_neighbors = int(params.get('minNeighbors', {}).get('default', 3))
    
    cv2_data_path = cv2.data.haarcascades
    xml_path = os.path.join(cv2_data_path, model_name)
    
    if not os.path.exists(xml_path):
         if os.path.exists(model_name):
             xml_path = model_name
         else:
             output = ensure_bgr(gray)
             cv2.putText(output, "Model Missing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             return output

    clf = cv2.CascadeClassifier(xml_path)
    rects = clf.detectMultiScale(gray, scaleFactor=scale, minNeighbors=min_neighbors)
    
    output = ensure_bgr(gray)
    for (x, y, w, h) in rects:
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return output

def op_hog_descriptor(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    ws = params.get('winStride', {}).get('default', [8, 8])
    if isinstance(ws, list): ws = tuple(ws)
    
    pad = params.get('padding', {}).get('default', [8, 8])
    if isinstance(pad, list): pad = tuple(pad)
    
    scale = float(params.get('scale', {}).get('default', 1.05))
    
    regions, _ = hog.detectMultiScale(img, winStride=ws, padding=pad, scale=scale)
    
    output = img.copy()
    for (x, y, w, h) in regions:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 255), 2)
    return output

def op_distance_transform(img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
    """
    距離變換 - 找出物體中心，輔助 Watershed 分離目標
    """
    gray = ensure_gray(img)
    
    # 二值化（確保輸入是二值圖）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 獲取參數
    distance_type_str = params.get('distanceType', {}).get('default', 'DIST_L2')
    mask_size = int(params.get('maskSize', {}).get('default', 5))
    normalize = bool(params.get('normalize', {}).get('default', True))
    
    # 映射距離類型字串到 OpenCV 常數
    distance_types = {
        'DIST_L1': cv2.DIST_L1,
        'DIST_L2': cv2.DIST_L2,
        'DIST_C': cv2.DIST_C
    }
    distance_type = distance_types.get(distance_type_str, cv2.DIST_L2)
    
    # 執行距離變換
    dist_transform = cv2.distanceTransform(binary, distance_type, mask_size)
    
    # 正規化到 [0, 255] 以便顯示
    if normalize:
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
    
    # 轉換為 8-bit 影像
    dist_transform = np.uint8(dist_transform)
    
    return ensure_bgr(dist_transform)
