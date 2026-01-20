# n8n AI Integration Guide

> **For n8n AI Agents & Workflow Builders**
> This document describes how to programmatically interact with the NKUST AoV Tool codebase.

## 1. Core Architecture for Automation
The system is designed to be modular. n8n workflows should interact with specific modules rather than the UI (`aov_app.py`).

| Module | Purpose | Key Method | Return Type |
|--------|---------|------------|-------------|
| `logic_engine.py` | **Planner**. Converts text to pipeline JSON. | `process_user_query(text)` | `List[Dict]` (Pipeline JSON) |
| `processor.py` | **Executor**. Runs OpenCV ops. | `execute_pipeline(img, json)` | `np.ndarray` (Image) |
| `processor.py` | **Introspection**. Lists capabilities. | `get_supported_operations()` | `Dict` (Schema) |
| `library_manager.py` | **Knowledge Base**. Manages algorithms. | `list_algorithms()` | `List[Dict]` |

## 2. Introspection (How to learn what I can do)
To avoid hallucinations, AI agents should query the `ImageProcessor` to understand available tools.

```python
from processor import ImageProcessor
proc = ImageProcessor()

# Returns a JSON schema of all supported operations and their descriptions
capabilities = proc.get_supported_operations()
print(json.dumps(capabilities, indent=2))
```

**Output Example:**
```json
{
  "GaussianBlur": {
    "description": "Apply Gaussian Blur to reduce noise.",
    "type": "computer_vision_operation"
  },
  "Canny": {
    "description": "Detect edges using Canny algorithm.",
    "type": "computer_vision_operation"
  }
}
```

## 3. Pipeline JSON Schema
The core data structure is the **Pipeline List**. AI agents should generate JSON matching this format:

```json
[
  {
    "id": "node_0",
    "function": "GaussianBlur",  // Must match a key in get_supported_operations()
    "parameters": {
      "ksize": {"default": [5, 5]},
      "sigmaX": {"default": 0}
    },
    "fpga_constraints": {
      "estimated_clk": 150,
      "resource_usage": "Medium"
    }
  }
]
```

## 4. Automation Workflow Example (Python Script)
An n8n `Execute Command` node can run a script like this to process images without the UI:

```python
import cv2
from processor import ImageProcessor
from logic_engine import LogicEngine

# 1. Initialize
engine = LogicEngine()
processor = ImageProcessor()

# 2. Plan (Text -> JSON)
pipeline = engine.process_user_query("Detect edges in this image")

# 3. Execute (Image + JSON -> Image)
img = cv2.imread("input.jpg")
result = processor.execute_pipeline(img, pipeline)

# 4. Save
cv2.imwrite("output.jpg", result)
```

## 5. File Structure for Context
- **`tech_lib.json`**: The Single Source of Truth for algorithm data. Read this to understand parameter constraints.
- **`processor.py`**: The execution logic. Read `operation_map` to see implementation details.

---

# tech_lib.json (Knowledge Base)

```json
{
    "schema_version": "1.0.0",
    "last_updated": "2026-01-13T14:23:00+08:00",
    "description": "NKUST Vision Lab - OpenCV Algorithm Library with FPGA Constraints",
    "libraries": {
        "official": {
            "gaussian_blur": {
                "id": "cv_gaussian_blur_v1",
                "name": "Gaussian Blur",
                "category": "preprocessing",
                "description": "平滑濾波器，用於降噪與模糊處理",
                "opencv_function": "cv2.GaussianBlur",
                "parameters": {
                    "ksize": {
                        "type": "tuple",
                        "default": [
                            9,
                            9
                        ],
                        "range": [
                            [
                                3,
                                3
                            ],
                            [
                                31,
                                31
                            ]
                        ],
                        "constraint": "必須為奇數",
                        "description": "Kernel 大小（更大的值可更強力去除紋理）"
                    },
                    "sigmaX": {
                        "type": "float",
                        "default": 2.0,
                        "range": [
                            0,
                            10
                        ],
                        "description": "X 方向標準差（更大的值模糊更強）"
                    },
                    "sigmaY": {
                        "type": "float",
                        "default": 0,
                        "range": [
                            0,
                            10
                        ],
                        "description": "Y 方向標準差（0 表示自動計算）"
                    }
                },
                "fpga_constraints": {
                    "estimated_clk": 150,
                    "resource_usage": "Medium",
                    "latency_type": "Pipeline",
                    "dsp_count": 4,
                    "lut_count": 2500,
                    "bram_kb": 18,
                    "notes": "Kernel 越大，資源消耗越高。建議 5x5 或 7x7。"
                },
                "author": "NKUST_Vision_Lab",
                "date_added": "2024-09-01",
                "name_zh": "高斯模糊"
            },
            "canny_edge": {
                "id": "cv_canny_v1",
                "name": "Canny Edge Detection",
                "category": "edge_detection",
                "description": "經典邊緣檢測演算法，包含梯度計算與非極大值抑制",
                "opencv_function": "cv2.Canny",
                "parameters": {
                    "threshold1": {
                        "type": "int",
                        "default": 50,
                        "range": [
                            0,
                            255
                        ],
                        "description": "第一閾值（低閾值）"
                    },
                    "threshold2": {
                        "type": "int",
                        "default": 150,
                        "range": [
                            0,
                            255
                        ],
                        "description": "第二閾值（高閾值）"
                    },
                    "apertureSize": {
                        "type": "int",
                        "default": 3,
                        "range": [
                            3,
                            7
                        ],
                        "constraint": "必須為奇數",
                        "description": "Sobel 算子的 Aperture 大小"
                    },
                    "L2gradient": {
                        "type": "bool",
                        "default": false,
                        "description": "是否使用 L2 範數計算梯度"
                    }
                },
                "fpga_constraints": {
                    "estimated_clk": 450,
                    "resource_usage": "High",
                    "latency_type": "Pipeline",
                    "dsp_count": 12,
                    "lut_count": 8500,
                    "bram_kb": 72,
                    "notes": "包含 Sobel, NMS, Hysteresis 三大模組。建議預留充足資源。"
                },
                "author": "NKUST_Vision_Lab",
                "date_added": "2024-09-01",
                "name_zh": "Canny邊緣偵測"
            },
            "dilate": {
                "id": "cv_dilate_v1",
                "name": "Morphological Dilation",
                "category": "morphology",
                "description": "形態學膨脹運算，用於填充小孔洞與連接斷裂區域",
                "opencv_function": "cv2.dilate",
                "parameters": {
                    "kernel": {
                        "type": "ndarray",
                        "default": "np.ones((3,3), np.uint8)",
                        "description": "結構元素（Kernel）"
                    },
                    "iterations": {
                        "type": "int",
                        "default": 1,
                        "range": [
                            1,
                            10
                        ],
                        "description": "迭代次數"
                    }
                },
                "fpga_constraints": {
                    "estimated_clk": 80,
                    "resource_usage": "Low",
                    "latency_type": "Pipeline",
                    "dsp_count": 0,
                    "lut_count": 1200,
                    "bram_kb": 9,
                    "notes": "單次 Dilate 非常輕量。但多次迭代會累積延遲。"
                },
                "author": "NKUST_Vision_Lab",
                "date_added": "2024-09-01",
                "name_zh": "膨脹運算"
            },
            "hough_circles": {
                "id": "cv_hough_circles_v1",
                "name": "Hough Circle Transform",
                "category": "shape_detection",
                "description": "霍夫圓形檢測，用於辨識圓形物體",
                "opencv_function": "cv2.HoughCircles",
                "parameters": {
                    "method": {
                        "type": "int",
                        "default": "cv2.HOUGH_GRADIENT",
                        "description": "檢測方法"
                    },
                    "dp": {
                        "type": "float",
                        "default": 1.2,
                        "range": [
                            1.0,
                            3.0
                        ],
                        "description": "累加器解析度與圖像解析度的反比"
                    },
                    "minDist": {
                        "type": "int",
                        "default": 50,
                        "range": [
                            10,
                            200
                        ],
                        "description": "檢測到的圓心之間的最小距離（至少應為影像高度的1/8）"
                    },
                    "param1": {
                        "type": "int",
                        "default": 50,
                        "range": [
                            10,
                            200
                        ],
                        "description": "Canny 邊緣檢測的高閾值"
                    },
                    "param2": {
                        "type": "int",
                        "default": 50,
                        "range": [
                            10,
                            100
                        ],
                        "description": "累加器閾值（更高的值 = 更少誤檢，建議 >= 50）"
                    },
                    "minRadius": {
                        "type": "int",
                        "default": 20,
                        "range": [
                            0,
                            500
                        ],
                        "description": "最小圓半徑（避免偵測到小點）"
                    },
                    "maxRadius": {
                        "type": "int",
                        "default": 100,
                        "range": [
                            0,
                            500
                        ],
                        "description": "最大圓半徑（避免偵測到大區域）"
                    }
                },
                "fpga_constraints": {
                    "estimated_clk": 8500,
                    "resource_usage": "Very High",
                    "latency_type": "Iterative",
                    "dsp_count": 24,
                    "lut_count": 45000,
                    "bram_kb": 512,
                    "notes": "極度耗時！建議在 FPGA 上使用硬體加速或考慮替代方案（如 FindContours）。"
                },
                "author": "NKUST_Vision_Lab",
                "date_added": "2024-09-01",
                "name_zh": "霍夫圓偵測"
            },
            "background_subtractor": {
                "id": "cv_mog2_v1",
                "name": "前後景分割 (MOG2)",
                "category": "segmentation",
                "description": "混合高斯模型背景分割，用於偵測移動物體",
                "opencv_function": "cv2.createBackgroundSubtractorMOG2",
                "parameters": {
                    "history": {
                        "type": "int",
                        "default": 500,
                        "range": [
                            1,
                            10000
                        ],
                        "description": "歷史幀數量"
                    },
                    "varThreshold": {
                        "type": "float",
                        "default": 16.0,
                        "range": [
                            0,
                            255
                        ],
                        "description": "閾值，決定前景/背景"
                    },
                    "detectShadows": {
                        "type": "bool",
                        "default": true,
                        "description": "是否偵測陰影"
                    }
                },
                "fpga_constraints": {
                    "estimated_clk": 3500,
                    "resource_usage": "High",
                    "latency_type": "Iterative",
                    "dsp_count": 8,
                    "lut_count": 5000,
                    "bram_kb": 64,
                    "notes": "需要Frame Buffer，適合視訊處理"
                },
                "author": "NKUST_Vision_Lab",
                "date_added": "2026-01-16",
                "name_zh": "前後景分割"
            },
            "optical_flow": {
                "id": "cv_farneback_v1",
                "name": "光流演算法 (Farneback)",
                "category": "motion_analysis",
                "description": "密集光流估計，用於追蹤像素運動",
                "opencv_function": "cv2.calcOpticalFlowFarneback",
                "parameters": {
                    "pyr_scale": {
                        "type": "float",
                        "default": 0.5,
                        "range": [
                            0.1,
                            1.0
                        ],
                        "description": "金字塔縮放比例"
                    },
                    "levels": {
                        "type": "int",
                        "default": 3,
                        "range": [
                            1,
                            8
                        ],
                        "description": "金字塔層數"
                    },
                    "winsize": {
                        "type": "int",
                        "default": 15,
                        "range": [
                            5,
                            50
                        ],
                        "description": "平均窗口大小"
                    },
                    "iterations": {
                        "type": "int",
                        "default": 3,
                        "range": [
                            1,
                            10
                        ],
                        "description": "迭代次數"
                    },
                    "poly_n": {
                        "type": "int",
                        "default": 5,
                        "range": [
                            5,
                            7
                        ],
                        "description": "像素鄰域大小"
                    },
                    "poly_sigma": {
                        "type": "float",
                        "default": 1.2,
                        "range": [
                            1.1,
                            2.0
                        ],
                        "description": "高斯標準差"
                    }
                },
                "fpga_constraints": {
                    "estimated_clk": 15000,
                    "resource_usage": "Very High",
                    "latency_type": "Iterative",
                    "dsp_count": 16,
                    "lut_count": 12000,
                    "bram_kb": 128,
                    "notes": "計算量極大，建議降低解析度或使用疏性光流"
                },
                "author": "NKUST_Vision_Lab",
                "date_added": "2026-01-16",
                "name_zh": "光流估計"
            }
        },
        "contributed": {
            "clahe_enhance": {
                "id": "student_clahe_v1",
                "name": "CLAHE (Contrast Limited Adaptive Histogram Equalization)",
                "category": "preprocessing",
                "description": "自適應直方圖均衡化，用於處理光線不均的影像（學長姐貢獻）",
                "opencv_function": "cv2.createCLAHE",
                "parameters": {
                    "clipLimit": {
                        "type": "float",
                        "default": 2.0,
                        "range": [
                            1.0,
                            10.0
                        ],
                        "description": "對比度限制閾值"
                    },
                    "tileGridSize": {
                        "type": "tuple",
                        "default": [
                            8,
                            8
                        ],
                        "range": [
                            [
                                4,
                                4
                            ],
                            [
                                16,
                                16
                            ]
                        ],
                        "description": "網格大小"
                    }
                },
                "fpga_constraints": {
                    "estimated_clk": 320,
                    "resource_usage": "Medium",
                    "latency_type": "Pipeline",
                    "dsp_count": 2,
                    "lut_count": 3500,
                    "bram_kb": 36,
                    "notes": "需要大量記憶體來存儲區域直方圖。"
                },
                "author": "Student_2025_CoinRecognition",
                "date_added": "2025-01-09",
                "contributor_note": "此演算法在硬幣辨識專案中解決反光問題時被發現非常有效。",
                "name_zh": "CLAHE增強"
            }
        }
    },
    "_metadata": {
        "total_algorithms": 7,
        "official_count": 6,
        "contributed_count": 1,
        "maintainers": [
            "NKUST_Vision_Lab"
        ],
        "license": "MIT"
    }
}
```

---

# processor.py (Executor Logic)

```python
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
    
    def get_supported_operations(self) -> Dict[str, Dict]:
        """
        [AI-Friendly] 回傳所有支援的操作與其參數定義 (Schema)
        
        這讓 AI Agent (如 n8n) 可以動態查詢此 Processor 支援哪些功能
        而不需硬編碼。
        """
        schema = {}
        for op_name, func in self.operation_map.items():
            # 簡單解析 docstring
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
            # 轉換為 uint16 之前確保類型安全，但 numpy 處理通常是動態的
            # Explicitly cast to prevent type confusion in some linters
            circles_uint16 = np.uint16(np.around(circles))
            coin_count = len(circles_uint16[0])
            
            for i in circles_uint16[0, :]:
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
```

---

# logic_engine.py (LLM Planner Logic)

```python
"""
Logic Engine for NKUST AoV Tool
負責人：Prompt_Master (LLM_Orchestrator) + Bridge_Builder (Integration_Specialist)

職責：
1. 接收使用者的自然語言需求
2. 呼叫 LLM 獲取演算法建議（骨架）
3. 從資料庫附加 FPGA 約束資訊（血肉）
4. 處理 Fallback 機制
"""

import json
import os
from typing import List, Dict, Optional, Any
from library_manager import LibraryManager

# ==================== Utility: Safe Print (Windows Encoding Fix) ====================

def safe_print(message: str):
    """
    安全的 print 函數，避免 Windows cp950 編碼問題
    
    Args:
        message: 要輸出的訊息
    """
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: 移除非 ASCII 字符
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message)

# ==================== Prompt_Master 區域 ====================

class PromptMaster:
    """
    LLM Orchestrator - 負責設計 Prompt 並呼叫 LLM
    """
    
    # Prompt 模板（嚴格控制 LLM 輸出格式）
    SYSTEM_PROMPT = """You are an expert Computer Vision algorithm advisor for FPGA implementation.

**CRITICAL RULES:**
1. You MUST ONLY return a JSON array of OpenCV function names.
2. DO NOT generate any code, explanations, or parameters.
3. DO NOT use markdown formatting. Return pure JSON only.
4. Use standard OpenCV function names (e.g., "GaussianBlur", "Canny", "HoughCircles").
5. Arrange functions in a logical pipeline order.

**Output Format (STRICT):**
["FunctionName1", "FunctionName2", "FunctionName3"]

**Example Input:** "Detect edges in a noisy image"
**Example Output:** ["GaussianBlur", "Canny"]

**Example Input:** "Detect coins on a keyboard"
**Example Output:** ["GaussianBlur", "Canny", "HoughCircles"]
"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", base_url: str = "https://api.openai.com/v1"):
        """
        初始化 Prompt Master
        
        Args:
            api_key: OpenAI API Key（若為 None，從環境變數讀取）
            model: 使用的模型（預設 gpt-4）
            base_url: API Base URL (預設 OpenAI 官方)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            print("[Warning] No OpenAI API Key found. LLM features will be disabled.")
            self.llm_available = False
        else:
            self.llm_available = True
    
    def get_llm_suggestion(self, user_query: str, use_mock: bool = False) -> List[str]:
        """
        獲取 LLM 建議的演算法列表
        
        Args:
            user_query: 使用者的自然語言需求（例如：「偵測硬幣」）
            use_mock: 若為 True，使用 Mock 資料（用於測試）
        
        Returns:
            List[str]: 演算法函數名稱列表（骨架）
        """
        if use_mock or not self.llm_available:
            return self._get_mock_suggestion(user_query)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.3,  # 降低隨機性，確保輸出穩定
                max_tokens=200
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # 嚴格解析 JSON
            try:
                skeleton = json.loads(raw_output)
                if not isinstance(skeleton, list):
                    raise ValueError("LLM output is not a list")
                
                print(f"[Prompt_Master] LLM returned: {skeleton}")
                return skeleton
                
            except json.JSONDecodeError as e:
                print(f"[Error] LLM returned invalid JSON: {raw_output}")
                print(f"  Parse error: {e}")
                return self._get_fallback_suggestion(user_query)
        
        except Exception as e:
            print(f"[Error] LLM API call failed: {e}")
            return self._get_fallback_suggestion(user_query)
    
    def _get_mock_suggestion(self, user_query: str) -> List[str]:
        """
        Mock LLM 建議（用於測試或無 API Key 情況）
        
        使用嚴格的保守參數，避免誤檢
        """
        query_lower = user_query.lower()
        
        # 關鍵字匹配規則（STRICT CONSERVATIVE DEFAULTS）
        if "coin" in query_lower or "硬幣" in query_lower or "circle" in query_lower:
            # 硬幣偵測：必須使用嚴格的前處理
            return ["GaussianBlur", "Canny", "HoughCircles"]
        elif "edge" in query_lower or "邊緣" in query_lower:
            return ["GaussianBlur", "Canny"]
        elif "denoise" in query_lower or "降噪" in query_lower or "noise" in query_lower:
            return ["GaussianBlur", "Morphological_Open"]
        elif "blur" in query_lower or "模糊" in query_lower:
            return ["GaussianBlur"]
        else:
            # 預設通用前處理流程
            return ["GaussianBlur", "Canny"]
    
    def _get_fallback_suggestion(self, user_query: str) -> List[str]:
        """
        當 LLM 失敗時的 Fallback 機制
        """
        print("[Warning] Using fallback suggestion mechanism.")
        return self._get_mock_suggestion(user_query)


# ==================== Bridge_Builder 區域 ====================

class BridgeBuilder:
    """
    Integration Specialist - 負責將骨架與血肉結合
    """
    
    def __init__(self, library_manager: LibraryManager):
        """
        初始化 Bridge Builder
        
        Args:
            library_manager: LibraryManager 實例
        """
        self.lib_manager = library_manager
        self.verilog_guru = VerilogGuru()  # Fallback 策略提供者
    
    def hydrate_pipeline(self, skeleton_list: List[str]) -> List[Dict[str, Any]]:
        """
        將 LLM 的骨架（函數列表）附加資料庫資訊（血肉）
        
        Args:
            skeleton_list: LLM 回傳的函數名稱列表
        
        Returns:
            List[Dict]: 完整的 Pipeline 節點列表，每個節點包含：
                - id: 節點唯一 ID
                - name: 函數名稱
                - category: 類別
                - fpga_constraints: FPGA 約束
                - parameters: 參數定義
                - source: "official" / "contributed" / "unknown"
        """
        hydrated_pipeline = []
        
        for idx, func_name in enumerate(skeleton_list):
            print(f"[Bridge_Builder] Hydrating '{func_name}'...")
            
            # Step 1: 嘗試在資料庫中查找（先查 official，再查 contributed）
            algo_data = self._lookup_algorithm(func_name)
            
            if algo_data:
                # 成功找到！附加完整資訊
                node = {
                    "id": f"node_{idx}",
                    "name": algo_data['name'],
                    "function": func_name,
                    "category": algo_data['category'],
                    "description": algo_data['description'],
                    "fpga_constraints": algo_data['fpga_constraints'],
                    "parameters": algo_data.get('parameters', {}),
                    "source": algo_data.get('_library_type', 'official'),
                    "next_node_id": f"node_{idx + 1}" if idx < len(skeleton_list) - 1 else None
                }
                print(f"  [OK] Found in database ({node['source']})")
                
            else:
                # 找不到！啟動 Fallback 機制（由 Verilog_Guru 提供）
                print(f"  [X] Not found in database. Using fallback...")
                node = self.verilog_guru.create_fallback_node(func_name, idx, len(skeleton_list))
            
            hydrated_pipeline.append(node)
        
        return hydrated_pipeline
    
    def _lookup_algorithm(self, func_name: str) -> Optional[Dict]:
        """
        在資料庫中查找演算法（強化模糊匹配）
        
        嘗試多種匹配策略：
        1. 完全匹配 algo_id (例如 "gaussian_blur")
        2. CamelCase 轉 snake_case (例如 "GaussianBlur" -> "gaussian_blur")
        3. 部分名稱匹配 (例如 "Gaussian" 匹配到 "Gaussian Blur")
        4. 移除空格和特殊字符後匹配
        """
        # 策略 1: 直接查找（轉小寫 + 底線）
        normalized_name = func_name.lower().replace(" ", "_")
        
        for lib_type in ['official', 'contributed']:
            algo = self.lib_manager.get_algorithm(normalized_name, lib_type)
            if algo:
                algo['_library_type'] = lib_type
                print(f"    [Match Strategy 1] Exact match: {normalized_name}")
                return algo
        
        # 策略 2: CamelCase 轉 snake_case
        # "GaussianBlur" -> "gaussian_blur"
        snake_case_name = self._camel_to_snake(func_name)
        
        for lib_type in ['official', 'contributed']:
            algo = self.lib_manager.get_algorithm(snake_case_name, lib_type)
            if algo:
                algo['_library_type'] = lib_type
                print(f"    [Match Strategy 2] CamelCase->snake_case: {snake_case_name}")
                return algo
        
        # 策略 3: 部分名稱匹配
        # "HoughCircles" -> search for "hough" in names
        func_lower = func_name.lower()
        
        for lib_type in ['official', 'contributed']:
            for algo_id, algo_data in self.lib_manager.data['libraries'][lib_type].items():
                algo_name_lower = algo_data.get('name', '').lower()
                
                # 檢查是否包含關鍵字
                if func_lower in algo_name_lower or algo_name_lower in func_lower:
                    algo_copy = self.lib_manager.get_algorithm(algo_id, lib_type)
                    if algo_copy:
                        algo_copy['_library_type'] = lib_type
                        print(f"    [Match Strategy 3] Partial match: {algo_id} (found '{func_name}' in '{algo_data.get('name')}')")
                        return algo_copy
        
        # 策略 4: 搜尋名稱（最後手段）
        results = self.lib_manager.search_by_name(func_name)
        if results:
            print(f"    [Match Strategy 4] Search by name: {results[0].get('name')}")
            return results[0]
        
        return None
    
    def _camel_to_snake(self, name: str) -> str:
        """
        將 CamelCase 轉換為 snake_case
        
        Examples:
            "GaussianBlur" -> "gaussian_blur"
            "HoughCircles" -> "hough_circles"
            "CLAHE" -> "clahe"
        """
        import re
        # 在大寫字母前插入底線，然後轉小寫
        snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return snake.lower()
    
    def export_to_json(self, pipeline: List[Dict], output_path: str = "pipeline_output.json"):
        """
        導出 Pipeline 為 JSON 檔案（給 FPGA 或其他工具使用）
        
        Args:
            pipeline: hydrate_pipeline() 的輸出
            output_path: 輸出檔案路徑
        """
        output_data = {
            "schema_version": "1.0.0",
            "pipeline_name": "Auto_Generated_Pipeline",
            "total_nodes": len(pipeline),
            "total_estimated_clk": sum(node['fpga_constraints']['estimated_clk'] for node in pipeline),
            "nodes": pipeline
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[Bridge_Builder] Pipeline exported to '{output_path}'")
        print(f"  - Total Nodes: {len(pipeline)}")
        print(f"  - Total CLK: {output_data['total_estimated_clk']}")


# ==================== Verilog_Guru 區域 ====================

class VerilogGuru:
    """
    FPGA Hardware Specialist - 提供 Fallback 策略與預設約束
    """
    
    @staticmethod
    def estimate_hardware_cost(op_name: str, img_width: int = 640, img_height: int = 480) -> Dict[str, Any]:
        """
        启發式 CLK 估算（當資料庫中無該演算法時）
        
        Args:
            op_name: 演算法名稱
            img_width: 圖像寬度
            img_height: 圖像高度
        
        Returns:
            Dict: FPGA 約束資訊
        """
        pixels = img_width * img_height
        op_lower = op_name.lower()
        
        # 分類估算
        if any(kw in op_lower for kw in ['threshold', 'color', 'convert', 'grayscale']):
            # Point Operations: 1 clk per pixel
            estimated_clk = pixels
            resource = "Low"
            latency = "Pipeline"
            notes = "Point operation: 1 clk/pixel heuristic"
        
        elif any(kw in op_lower for kw in ['blur', 'gaussian', 'median', 'dilate', 'erode', 'sobel', 'canny']):
            # Window Operations: kernel_size^2 * pixels (assume 5x5 kernel)
            kernel_size = 5
            estimated_clk = (kernel_size ** 2) * pixels
            resource = "Medium"
            latency = "Pipeline"
            notes = f"Window operation: {kernel_size}x{kernel_size} kernel heuristic"
        
        elif any(kw in op_lower for kw in ['hough', 'voting', 'transform']):
            # Complex Voting: Very expensive
            estimated_clk = pixels * 100  # Aggressive estimate
            resource = "Very High"
            latency = "Iterative"
            notes = "Complex voting operation: Conservative high estimate"
        
        elif any(kw in op_lower for kw in ['morpholog', 'morph', 'open', 'close']):
            # Morphological: Similar to blur but simpler
            estimated_clk = pixels * 4
            resource = "Low"
            latency = "Pipeline"
            notes = "Morphological operation: 4 clk/pixel heuristic"
        
        else:
            # Unknown: Conservative estimate
            estimated_clk = pixels * 10
            resource = "Unknown"
            latency = "Software_Only"
            notes = "Unknown operation: Conservative 10 clk/pixel estimate"
        
        return {
            "estimated_clk": estimated_clk,
            "resource_usage": resource,
            "latency_type": latency,
            "dsp_count": 0,
            "lut_count": 0,
            "bram_kb": 0,
            "notes": f"HEURISTIC: {notes} (Resolution: {img_width}x{img_height})"
        }
    
    def create_fallback_node(self, func_name: str, idx: int, total_nodes: int) -> Dict:
        """
        為未知函數建立 Fallback 節點（使用启發式估算）
        
        Args:
            func_name: 函數名稱
            idx: 節點索引
            total_nodes: 總節點數
        
        Returns:
            Dict: Fallback 節點資訊
        """
        # 使用启發式估算取代 999 預設值
        fpga_constraints = self.estimate_hardware_cost(func_name)
        
        return {
            "id": f"node_{idx}",
            "name": f"{func_name} (Heuristic)",
            "function": func_name,
            "category": "unknown",
            "description": f"LLM 建議的函數，使用启發式 CLK 估算。",
            "fpga_constraints": fpga_constraints,
            "parameters": {},
            "source": "llm_heuristic",
            "next_node_id": f"node_{idx + 1}" if idx < total_nodes - 1 else None,
            "_warning": "此節點使用启發式估算，建議人工驗證！"
        }
```
