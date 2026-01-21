"""
Processor Module for NKUST AoV Tool
負責人：Preview_Artist (Visual_Feedback_Engineer)

職責：
1. 執行 Pipeline（將 JSON 轉換為實際的 OpenCV 操作）
2. 提供即時預覽功能
3. 錯誤處理與異常保護
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import traceback

class ImageProcessor:
    """
    影像處理引擎 - 負責執行 Pipeline 並產生預覽
    
    使用 Dispatcher Pattern 將字串操作名稱映射到實際函數
    """
    
    def __init__(self):
        """
        初始化處理器
        """
        # Dispatcher 字典：將操作名稱映射到處理函數
        self.operation_map: Dict[str, Callable] = {
            # --- Official (Old) ---
            "GaussianBlur": self._op_gaussian_blur,
            "gaussian_blur": self._op_gaussian_blur,
            
            "Canny": self._op_canny,
            "canny_edge": self._op_canny,
            
            "Dilate": self._op_dilate,
            "dilate": self._op_dilate,
            
            "Erode": self._op_erode,
            "erode": self._op_erode,
            
            "HoughCircles": self._op_hough_circles,
            "hough_circles": self._op_hough_circles,
            
            "Threshold": self._op_threshold,
            "threshold": self._op_threshold,
            
            "MorphologicalOpen": self._op_morph_open,
            "morphological_open": self._op_morph_open,
            
            "CLAHE": self._op_clahe,
            "clahe_enhance": self._op_clahe,

            "background_subtractor": self._op_mog2,
            "optical_flow": self._op_optical_flow,

            # --- New Standard Library (2025 Expansion) ---
            # Filtering
            "median_blur": self._op_median_blur,
            "bilateral_filter": self._op_bilateral_filter,
            "box_filter": self._op_box_filter,

            # Morphology
            "morph_open": self._op_morph_open, # Re-use
            "morph_close": self._op_morph_close,
            "morph_gradient": self._op_morph_gradient,

            # Edge
            "sobel": self._op_sobel,
            "laplacian": self._op_laplacian,

            # Geometric
            "resize": self._op_resize,
            "flip": self._op_flip,
            "rotate": self._op_rotate,

            # Thresholding
            "threshold_binary": self._op_threshold_binary,
            "adaptive_threshold": self._op_adaptive_threshold,

            # Color
            "bgr2gray": self._op_bgr2gray,
            "bgr2hsv": self._op_bgr2hsv,
            "in_range": self._op_in_range,

            # Feature
            "harris_corner": self._op_harris_corner,
            "fast_detector": self._op_fast_detector,

            # Detection
            "find_contours": self._op_find_contours,
            "cascade_classifier": self._op_cascade_classifier,
            "hog_descriptor": self._op_hog_descriptor,

            # Arithmetic / Enhancement
            "bitwise_not": self._op_bitwise_not,
            "equalize_hist": self._op_equalize_hist,

            # --- Advanced Logic Nodes ---
            "advanced_coin_detection": self._op_advanced_coin_logic,
        }
    
    def _op_advanced_coin_logic(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 
        高階硬幣辨識邏輯 (Advanced Coin Detector)
        移植自: catch_coins/main.py 
        包含: Resize -> Gray -> Blur -> Hough -> Stability Heuristic (Single Frame)
        """
        # 1. 預處理 (Resize)
        max_size = int(params.get('max_image_size', {}).get('default', 800))
        h, w = img.shape[:2]
        longest_side = max(h, w)
        if longest_side > max_size:
            scale = max_size / longest_side
            new_size = (int(w * scale), int(h * scale))
            img_processed = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            img_processed = img.copy()

        # 2. 轉灰階 & 模糊 (Preprocessing)
        gray = self._ensure_gray(img_processed)
        blur_ksize = int(params.get('blur_ksize', {}).get('default', 13))
        if blur_ksize % 2 == 0: blur_ksize += 1
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # 3. 霍夫圓偵測 (Hough)
        dp = float(params.get('dp', {}).get('default', 1.2))
        min_dist = int(params.get('min_distance', {}).get('default', 40))
        canny_thresh = int(params.get('canny_threshold', {}).get('default', 50))
        acc_thresh = int(params.get('accumulator_threshold', {}).get('default', 30))
        min_radius = int(params.get('min_radius', {}).get('default', 20))
        max_radius = int(params.get('max_radius', {}).get('default', 80))

        detected = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=canny_thresh,
            param2=acc_thresh,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        output = img_processed.copy()
        
        # 4. 繪製結果 & 模擬穩定性過濾 (Logic)
        # 由於這是單張影像，我們無法做 Temporal Stability，
        # 但我們可以加上 "Radius Consistency" 或 "Overlap Removal" (簡單模擬)
        
        count = 0
        if detected is not None:
            # 轉換為整數
            circles_rounded = np.round(detected[0, :]).astype(int)
            
            # 簡單的過濾邏輯：移除過於重疊的圓 (Optional, Hough minDist handles mostly)
            # 這裡我們直接繪製，模擬 main.py 的 Visualizer
            
            for (x, y, r) in circles_rounded:
                # 畫圓 (綠色)
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                # 畫圓心 (紅色)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
                count += 1
            
            # 加上標籤文字
            label = f"Stable Coins: {count}" # 為了符合 main.py 風格，我們保留這個標籤名
            cv2.putText(output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if debug:
            print(f"    [AdvancedCoin] Found {count} coins")

        return output

        """
        [AI-Friendly] 回傳所有支援的操作與其參數定義 (Schema)
        """
        schema = {}
        for op_name, func in self.operation_map.items():
            doc = func.__doc__.strip() if func.__doc__ else "No description"
            schema[op_name] = {
                "description": doc,
                "type": "computer_vision_operation"
            }
        return schema

    def execute_pipeline(
        self, 
        image_bgr: np.ndarray, 
        pipeline_json: List[Dict[str, Any]],
        debug_mode: bool = False
    ) -> np.ndarray:
        """
        執行完整的 Pipeline
        """
        if image_bgr is None or image_bgr.size == 0:
            print("[Error] Invalid input image")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        current_image = image_bgr.copy()
        
        print(f"\n{'=' * 60}")
        print(f"Executing Pipeline ({len(pipeline_json)} nodes)")
        print(f"{'=' * 60}")
        
        for idx, node in enumerate(pipeline_json):
            try:
                node_id = node.get('id', f'node_{idx}')
                func_name = node.get('function', node.get('name', 'Unknown'))
                params = node.get('parameters', {})
                
                print(f"\n[{node_id}] {func_name}")
                
                if func_name in self.operation_map:
                    operation_func = self.operation_map[func_name]
                    current_image = operation_func(current_image, params, debug_mode)
                    
                    # Ensure image is valid after op
                    if current_image is None:
                        print(f"  [ERROR] Operation returned None! Resetting to black image.")
                        current_image = np.zeros((100, 100, 3), dtype=np.uint8)
                    
                    print(f"  [OK] Executed successfully. Output shape: {current_image.shape}")
                else:
                    print(f"  [WARN] Unknown operation '{func_name}'. Skipping...")
                
            except Exception as e:
                print(f"  [ERROR] Error executing node {node.get('id', idx)}: {e}")
                if debug_mode:
                    traceback.print_exc()
        
        print(f"\n{'=' * 60}")
        print(f"Pipeline execution complete.")
        print(f"{'=' * 60}\n")
        
        return current_image
    
    # ==================== Helper ====================
    def _ensure_gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # ==================== Implementation ====================
    
    def _op_gaussian_blur(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 高斯模糊 """
        ksize = tuple(params.get('ksize', {}).get('default', [5, 5]))
        sigmaX = params.get('sigmaX', {}).get('default', 0)
        sigmaY = params.get('sigmaY', {}).get('default', 0)
        return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)

    def _op_median_blur(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 中值濾波 """
        ksize = int(params.get('ksize', {}).get('default', 5))
        if ksize % 2 == 0: ksize += 1
        return cv2.medianBlur(img, ksize)

    def _op_bilateral_filter(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 雙邊濾波 """
        d = int(params.get('d', {}).get('default', 9))
        sigmaColor = float(params.get('sigmaColor', {}).get('default', 75.0))
        sigmaSpace = float(params.get('sigmaSpace', {}).get('default', 75.0))
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    def _op_box_filter(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 方框濾波 """
        ksize = tuple(params.get('ksize', {}).get('default', [3, 3]))
        normalize = bool(params.get('normalize', {}).get('default', True))
        return cv2.boxFilter(img, -1, ksize, normalize=normalize)

    def _op_canny(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ Canny 邊緣檢測 """
        gray = self._ensure_gray(img)
        thresh1 = params.get('threshold1', {}).get('default', 50)
        thresh2 = params.get('threshold2', {}).get('default', 150)
        aperture = params.get('apertureSize', {}).get('default', 3)
        edges = cv2.Canny(gray, thresh1, thresh2, apertureSize=aperture)
        return self._ensure_bgr(edges)

    def _op_sobel(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ Sobel 邊緣檢測 """
        gray = self._ensure_gray(img)
        dx = int(params.get('dx', {}).get('default', 1))
        dy = int(params.get('dy', {}).get('default', 0))
        ksize = int(params.get('ksize', {}).get('default', 3))
        # Use CV_64F to capture negative edges, then convert back
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        abs_sobel = cv2.convertScaleAbs(sobel)
        return self._ensure_bgr(abs_sobel)

    def _op_laplacian(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ Laplacian 邊緣檢測 """
        gray = self._ensure_gray(img)
        ksize = int(params.get('ksize', {}).get('default', 3))
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        abs_lap = cv2.convertScaleAbs(lap)
        return self._ensure_bgr(abs_lap)

    def _op_dilate(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 膨脹 """
        iterations = int(params.get('iterations', {}).get('default', 1))
        k_size_val = params.get('kernel', {}).get('default', [3, 3])
        if isinstance(k_size_val, list): k_size_val = tuple(k_size_val)
        else: k_size_val = (3, 3)
        
        kernel = np.ones(k_size_val, np.uint8)
        return cv2.dilate(img, kernel, iterations=iterations)

    def _op_erode(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 侵蝕 """
        iterations = int(params.get('iterations', {}).get('default', 1))
        k_size = int(params.get('kernel_size', {}).get('default', 3))
        kernel = np.ones((k_size, k_size), np.uint8)
        return cv2.erode(img, kernel, iterations=iterations)

    def _op_morph_open(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 開運算 """
        k_size = int(params.get('kernel_size', {}).get('default', 5))
        iterations = int(params.get('iterations', {}).get('default', 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

    def _op_morph_close(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 閉運算 """
        k_size = int(params.get('kernel_size', {}).get('default', 5))
        iterations = int(params.get('iterations', {}).get('default', 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    def _op_morph_gradient(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 形態學梯度 """
        k_size = int(params.get('kernel_size', {}).get('default', 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    def _op_hough_circles(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 霍夫圓形檢測 """
        gray = self._ensure_gray(img)
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
            
            if debug: print(f"    Found {len(circles_rounded[0])} circles")
        
        return output

    def _op_threshold(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 舊版 Threshold """
        gray = self._ensure_gray(img)
        thresh_val = int(params.get('thresh', {}).get('default', 127))
        max_val = int(params.get('maxval', {}).get('default', 255))
        _, binary = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
        return self._ensure_bgr(binary)

    def _op_threshold_binary(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 新版 Binary Threshold """
        return self._op_threshold(img, params, debug)

    def _op_adaptive_threshold(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 自適應二值化 """
        gray = self._ensure_gray(img)
        max_val = int(params.get('maxValue', {}).get('default', 255))
        block_size = int(params.get('blockSize', {}).get('default', 11))
        c = float(params.get('C', {}).get('default', 2))
        if block_size % 2 == 0: block_size += 1
        
        binary = cv2.adaptiveThreshold(
            gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )
        return self._ensure_bgr(binary)

    def _op_resize(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 調整大小 """
        w = int(params.get('width', {}).get('default', 640))
        h = int(params.get('height', {}).get('default', 480))
        return cv2.resize(img, (w, h))

    def _op_flip(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 翻轉 """
        flip_code = int(params.get('flipCode', {}).get('default', 1))
        return cv2.flip(img, flip_code)

    def _op_rotate(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 旋轉 """
        code = int(params.get('rotateCode', {}).get('default', 0))
        cv2_codes = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        if 0 <= code < 3:
            return cv2.rotate(img, cv2_codes[code])
        return img

    def _op_bgr2gray(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ BGR 轉 Gray (Display as BGR) """
        gray = self._ensure_gray(img)
        return self._ensure_bgr(gray)

    def _op_bgr2hsv(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ BGR 轉 HSV """
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def _op_in_range(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 顏色範圍遮罩 """
        l_b = int(params.get('lower_b', {}).get('default', 0))
        l_g = int(params.get('lower_g', {}).get('default', 0))
        l_r = int(params.get('lower_r', {}).get('default', 0))
        u_b = int(params.get('upper_b', {}).get('default', 255))
        u_g = int(params.get('upper_g', {}).get('default', 255))
        u_r = int(params.get('upper_r', {}).get('default', 255))
        
        lower = np.array([l_b, l_g, l_r])
        upper = np.array([u_b, u_g, u_r])
        
        mask = cv2.inRange(img, lower, upper)
        return self._ensure_bgr(mask)

    def _op_harris_corner(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ Harris Corner """
        gray = self._ensure_gray(img)
        gray = np.float32(gray)
        block_size = int(params.get('blockSize', {}).get('default', 2))
        ksize = int(params.get('ksize', {}).get('default', 3))
        k = float(params.get('k', {}).get('default', 0.04))
        
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        
        # Visualize
        output = img.copy()
        # Dilate to mark corners
        dst = cv2.dilate(dst, None)
        # Red corners
        output[dst > 0.01 * dst.max()] = [0, 0, 255]
        return output

    def _op_fast_detector(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ FAST Feature Detector """
        gray = self._ensure_gray(img)
        threshold = int(params.get('threshold', {}).get('default', 10))
        nonmax = bool(params.get('nonmaxSuppression', {}).get('default', True))
        
        fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax)
        keypoints = fast.detect(gray, None)
        
        return cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

    def _op_bitwise_not(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ Bitwise NOT """
        return cv2.bitwise_not(img)

    def _op_equalize_hist(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 直方圖均衡化 """
        gray = self._ensure_gray(img)
        eq = cv2.equalizeHist(gray)
        return self._ensure_bgr(eq)

    def _op_clahe(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ CLAHE """
        clip_limit = params.get('clipLimit', {}).get('default', 2.0)
        tile_size_val = params.get('tileGridSize', {}).get('default', [8, 8])
        if isinstance(tile_size_val, list): tile_size = tuple(tile_size_val)
        else: tile_size = (8, 8)
        
        # If color, apply to Lightness channel in LAB
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

    def _op_mog2(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ Background Subtractor MOG2 (Mock implementation for single image) """
        # MOG2 requires a sequence. For a single image tool, this is tricky.
        # We will just return the image with a warning or mock it.
        # But user might feed a video frame sequence if extended.
        # Here we just implement the API call but it won't do much on static img without history.
        if not hasattr(self, '_mog2_subtractor'):
            history = int(params.get('history', {}).get('default', 500))
            varThreshold = float(params.get('varThreshold', {}).get('default', 16))
            detectShadows = bool(params.get('detectShadows', {}).get('default', True))
            self._mog2_subtractor = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
        
        mask = self._mog2_subtractor.apply(img)
        return self._ensure_bgr(mask)

    def _op_optical_flow(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ Optical Flow (Farneback) - Needs previous frame """
        # Static image tool limitation.
        # We store 'prev_gray' in self if we are in a session.
        gray = self._ensure_gray(img)
        
        if not hasattr(self, '_prev_gray_flow'):
            self._prev_gray_flow = gray
            return img # First frame, no flow
        
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray_flow, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self._prev_gray_flow = gray
        
        # Visualize flow (HSV)
        h, w = gray.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _op_find_contours(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ 找出輪廓並繪製邊界框 """
        # Need binary input
        gray = self._ensure_gray(img)
        # Assuming input is already binary or somewhat segmented, but findContours works on any grayscale
        # Better to ensure binary if not
        
        mode_val = int(params.get('mode', {}).get('default', 1)) # RETR_LIST
        method_val = int(params.get('method', {}).get('default', 2)) # CHAIN_APPROX_SIMPLE
        min_area = int(params.get('min_area', {}).get('default', 100))
        draw_original = bool(params.get('draw_on_original', {}).get('default', False))
        
        contours, _ = cv2.findContours(gray, mode_val, method_val)
        
        # If we want to draw on original, we can't because we lost it in the pipeline chain
        # (Current pipeline design passes processed image forward)
        # So we draw on a color conversion of the current input
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                count += 1
        
        if debug: print(f"    Found {count} contours > {min_area}")
        return output

    def _op_cascade_classifier(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ Haar Cascade Classifier """
        gray = self._ensure_gray(img)
        
        model_name = str(params.get('model_type', {}).get('default', 'haarcascade_frontalface_default.xml'))
        scale = float(params.get('scaleFactor', {}).get('default', 1.1))
        min_neighbors = int(params.get('minNeighbors', {}).get('default', 3))
        
        # In a real app, we need path management. 
        # For now, we assume standard opencv data or local files.
        # We try to load from cv2 data
        import os
        cv2_data_path = cv2.data.haarcascades
        xml_path = os.path.join(cv2_data_path, model_name)
        
        if not os.path.exists(xml_path):
             # Try local
             if os.path.exists(model_name):
                 xml_path = model_name
             else:
                 if debug: print(f"    [ERROR] Model file not found: {xml_path}")
                 # Draw text error
                 output = self._ensure_bgr(gray)
                 cv2.putText(output, "Model Missing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                 return output

        clf = cv2.CascadeClassifier(xml_path)
        rects = clf.detectMultiScale(gray, scaleFactor=scale, minNeighbors=min_neighbors)
        
        output = self._ensure_bgr(gray) # Better to draw on BGR version of input
        
        for (x, y, w, h) in rects:
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        if debug: print(f"    Detected {len(rects)} objects")
        return output

    def _op_hog_descriptor(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """ HOG Pedestrian Detector """
        # HOG usually expects color or gray, but resize might be needed
        # Standard People Detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # winStride
        ws = params.get('winStride', {}).get('default', [8, 8])
        if isinstance(ws, list): ws = tuple(ws)
        
        # padding
        pad = params.get('padding', {}).get('default', [8, 8])
        if isinstance(pad, list): pad = tuple(pad)
        
        scale = float(params.get('scale', {}).get('default', 1.05))
        
        # Resize if too large to speed up (optional)
        # img = cv2.resize(img, (640, 480))
        
        regions, _ = hog.detectMultiScale(img, winStride=ws, padding=pad, scale=scale)
        
        output = img.copy()
        for (x, y, w, h) in regions:
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
        if debug: print(f"    Detected {len(regions)} pedestrians")
        return output

    def create_thumbnail(self, image: np.ndarray, max_width: int = 640) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            return cv2.resize(image, (max_width, new_h))
        return image

if __name__ == "__main__":
    print("Processor loaded.")
