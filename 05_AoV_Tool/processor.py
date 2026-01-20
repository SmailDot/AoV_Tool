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
        }
    
    def execute_pipeline(
        self, 
        image_bgr: np.ndarray, 
        pipeline_json: List[Dict[str, Any]],
        debug_mode: bool = False
    ) -> np.ndarray:
        """
        執行完整的 Pipeline
        
        Args:
            image_bgr: 輸入影像（BGR 格式）
            pipeline_json: Pipeline 節點列表（來自 logic_engine）
            debug_mode: 是否顯示詳細除錯資訊
        
        Returns:
            np.ndarray: 處理後的影像
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
                
                # 查找對應的操作函數
                if func_name in self.operation_map:
                    operation_func = self.operation_map[func_name]
                    current_image = operation_func(current_image, params, debug_mode)
                    print(f"  [OK] Executed successfully. Output shape: {current_image.shape}")
                else:
                    print(f"  [WARN] Unknown operation '{func_name}'. Skipping...")
                    # 不中斷，繼續執行下一個節點
                
            except Exception as e:
                print(f"  [ERROR] Error executing node {node.get('id', idx)}: {e}")
                if debug_mode:
                    traceback.print_exc()
                # 繼續執行，不要因為單一節點錯誤而崩潰
        
        print(f"\n{'=' * 60}")
        print(f"Pipeline execution complete.")
        print(f"{'=' * 60}\n")
        
        return current_image
    
    # ==================== 操作函數 (Dispatcher Targets) ====================
    
    def _op_gaussian_blur(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """
        高斯模糊
        """
        ksize = params.get('ksize', {}).get('default', [5, 5])
        if isinstance(ksize, list):
            ksize = tuple(ksize)
        
        sigmaX = params.get('sigmaX', {}).get('default', 0)
        sigmaY = params.get('sigmaY', {}).get('default', 0)
        
        if debug:
            print(f"    ksize={ksize}, sigmaX={sigmaX}, sigmaY={sigmaY}")
        
        return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)
    
    def _op_canny(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """
        Canny 邊緣檢測
        """
        # Canny 需要灰階影像
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        thresh1 = params.get('threshold1', {}).get('default', 50)
        thresh2 = params.get('threshold2', {}).get('default', 150)
        aperture = params.get('apertureSize', {}).get('default', 3)
        
        if debug:
            print(f"    threshold1={thresh1}, threshold2={thresh2}, aperture={aperture}")
        
        edges = cv2.Canny(gray, thresh1, thresh2, apertureSize=aperture)
        
        # 將單通道結果轉回 BGR 以便後續處理
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def _op_dilate(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """
        形態學膨脹
        """
        iterations = params.get('iterations', {}).get('default', 1)
        kernel_size = params.get('kernel', {}).get('default', (3, 3))
        
        if isinstance(kernel_size, str):
            kernel_size = (3, 3)  # Fallback
        
        kernel = np.ones(kernel_size, np.uint8)
        
        if debug:
            print(f"    iterations={iterations}, kernel_size={kernel_size}")
        
        return cv2.dilate(img, kernel, iterations=iterations)
    
    def _op_erode(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """
        形態學侵蝕
        """
        iterations = params.get('iterations', {}).get('default', 1)
        kernel_size = (3, 3)
        kernel = np.ones(kernel_size, np.uint8)
        
        if debug:
            print(f"    iterations={iterations}")
        
        return cv2.erode(img, kernel, iterations=iterations)
    
    def _op_hough_circles(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """
        霍夫圓形檢測
        """
        # 需要灰階影像
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        dp = params.get('dp', {}).get('default', 1.2)
        minDist = params.get('minDist', {}).get('default', 30)
        param1 = params.get('param1', {}).get('default', 50)
        param2 = params.get('param2', {}).get('default', 30)
        minRadius = params.get('minRadius', {}).get('default', 0)
        maxRadius = params.get('maxRadius', {}).get('default', 0)
        
        if debug:
            print(f"    dp={dp}, minDist={minDist}, param1={param1}, param2={param2}")
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp, minDist,
            param1=param1, param2=param2,
            minRadius=minRadius, maxRadius=maxRadius
        )
        
        # 繪製結果（OUTLINE ONLY - NO FILL）
        output = img.copy() if len(img.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        coin_count = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            coin_count = len(circles[0])
            
            for i in circles[0, :]:
                # 畫圓周（輪廓，不填充）
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # 畫圓心
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
            
            if debug:
                print(f"    Found {coin_count} circles")
        else:
            if debug:
                print(f"    No circles found")
        
        # 添加計數標籤（左上角）
        label_text = f"Detected: {coin_count} coins"
        cv2.putText(output, label_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return output
    
    def _op_threshold(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """
        二值化
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        thresh_val = params.get('thresh', {}).get('default', 127)
        max_val = params.get('maxval', {}).get('default', 255)
        
        _, binary = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _op_morph_open(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """
        形態學開運算
        """
        kernel_size = (15, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        
        if debug:
            print(f"    kernel_size={kernel_size}")
        
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    def _op_clahe(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        """
        CLAHE 對比度增強
        """
        clip_limit = params.get('clipLimit', {}).get('default', 2.0)
        tile_size = params.get('tileGridSize', {}).get('default', [8, 8])
        
        if isinstance(tile_size, list):
            tile_size = tuple(tile_size)
        
        # 轉換到 LAB 色彩空間
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 對 L 通道應用 CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_clahe = clahe.apply(l)
        
        # 合併回 BGR
        lab_clahe = cv2.merge((l_clahe, a, b))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        if debug:
            print(f"    clipLimit={clip_limit}, tileGridSize={tile_size}")
        
        return result


# ==================== 輔助函數 ====================

def create_thumbnail(image: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    建立縮圖（用於 UI 顯示）
    
    Args:
        image: 輸入影像
        max_width: 最大寬度
    
    Returns:
        縮放後的影像
    """
    if image is None or image.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        return cv2.resize(image, (max_width, new_h))
    
    return image


# ==================== 測試 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Processor Test Suite")
    print("=" * 60)
    
    # 建立測試影像
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 建立測試 Pipeline
    test_pipeline = [
        {
            "id": "node_0",
            "function": "GaussianBlur",
            "parameters": {
                "ksize": {"default": [5, 5]},
                "sigmaX": {"default": 0}
            }
        },
        {
            "id": "node_1",
            "function": "Canny",
            "parameters": {
                "threshold1": {"default": 50},
                "threshold2": {"default": 150}
            }
        }
    ]
    
    # 執行測試
    processor = ImageProcessor()
    result = processor.execute_pipeline(test_img, test_pipeline, debug_mode=True)
    
    print(f"\nResult shape: {result.shape}")
    print("=" * 60)
