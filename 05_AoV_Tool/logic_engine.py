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
    
    SYSTEM_PROMPT = """你是一位資深的電腦視覺演算法架構師 (Senior CV Architect)。
你的任務是根據使用者的自然語言需求，設計出邏輯嚴謹的 OpenCV Pipeline。

**核心變革 (Graph Logic):**
我們現在支援 **非線性流程 (DAG)**。你可以設計並行處理的路線，最後再合併。
- 預設情況下，節點會接收「上一個節點」的輸出。
- **但你可以明確指定 `inputs`** 來獲取更早之前的節點輸出，實現並行與合併。

**輸出格式 (STRICT JSON):**
1. "reasoning": (String) 繁體中文的設計思路。解釋並行處理的理由（例如：一路做邊緣偵測，一路做色彩過濾，最後 `bitwise_and` 合併）。
2. "pipeline": (List) 節點列表。
   - "id": (String) 節點唯一 ID (如 "source", "blur_branch", "color_branch", "merge")。
   - "function": (String) OpenCV 函數名。
   - "inputs": (List[String], Optional) 指定輸入來源的 ID。若省略，預設為上一個節點。**"source" 代表原始影像。**
   - "params": (Dict) 參數。

**合併操作 (Merge Functions):**
當你需要合併兩條路線時，請使用以下函數：
- `add`: 影像相加
- `addWeighted`: 影像權重相加 (Blending)
- `bitwise_and`: 取交集 (Masking)
- `bitwise_or`: 取聯集
- `absdiff`: 取差異

**範例輸入:** "幫我去除背景，保留紅色物體"
**範例輸出:**
{
  "reasoning": "為了精確保留紅色物體，我將採用雙流設計。第一路 (Branch A) 將影像轉為 HSV 並使用 inRange 抓取紅色遮罩。第二路 (Branch B) 保持原圖。最後使用 `bitwise_and` 將原圖與遮罩合併，達成去背效果。",
  "pipeline": [
    { "id": "to_hsv", "function": "cvtColor", "params": { "code": "COLOR_BGR2HSV" }, "inputs": ["source"] },
    { "id": "red_mask", "function": "inRange", "params": { "lower": [0, 100, 100], "upper": [10, 255, 255] }, "inputs": ["to_hsv"] },
    { "id": "result", "function": "bitwise_and", "params": {}, "inputs": ["source", "red_mask"] }
  ]
}

**可用函數庫:**
- Basic: GaussianBlur, MedianBlur, cvtColor, resize
- Edge: Canny, Sobel, Laplacian
- Morph: Morphological_Open, Morphological_Close, dilate, erode
- Merge: add, addWeighted, bitwise_and, bitwise_or, bitwise_xor, absdiff
"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            print("[Warning] No OpenAI API Key found. LLM features will be disabled.")
            self.llm_available = False
        else:
            self.llm_available = True
            
        # [NEW] Default fallback reasoning
        self.last_reasoning = ""
    
    def get_llm_suggestion(self, user_query: str, use_mock: bool = False) -> Dict[str, Any]:
        """
        獲取 LLM 建議
        """
        if use_mock:
            return self._get_mock_suggestion(user_query)
            
        if not self.llm_available:
            return {
                "pipeline": self._get_fallback_pipeline(user_query),
                "reasoning": "無法連線至 LLM (Missing API Key)，已切換至 Fallback 模式。",
                "error": "Missing API Key"
            }
        
        # [Strategy 1] Google Native SDK (Priority for Gemini models)
        if "google" in self.base_url or "generativelanguage" in self.base_url:
            return self._call_google_native(user_query)

        # [Strategy 2] OpenAI Compatible SDK (Default)
        return self._call_openai_compatible(user_query)

    def _call_google_native(self, user_query: str) -> Dict[str, Any]:
        """
        使用 Google Generative AI SDK (google-generativeai)
        """
        try:
            import google.generativeai as genai
            
            print(f"[LLM-Google] Using Native SDK for {self.model}")
            genai.configure(api_key=self.api_key)
            
            # Clean model name (remove 'google/' prefix if exists)
            model_name = self.model.replace("google/", "")
            
            # Create Model
            # Note: system_instruction is supported in newer SDKs
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=self.SYSTEM_PROMPT,
                generation_config={"response_mime_type": "application/json"}
            )
            
            response = model.generate_content(user_query)
            raw_output = response.text
            
            # Parse JSON
            data = json.loads(raw_output)
            
            if isinstance(data, dict) and "pipeline" in data:
                return {
                    "pipeline": data["pipeline"],
                    "reasoning": data.get("reasoning", "AI 未提供理由"),
                    "error": None
                }
            else:
                return {
                    "pipeline": self._get_fallback_pipeline(user_query),
                    "reasoning": f"Format Error: {raw_output[:100]}",
                    "error": "Invalid JSON format"
                }
                
        except Exception as e:
            return {
                "pipeline": [],
                "reasoning": f"Google SDK Error: {str(e)}",
                "error": str(e)
            }

    def _call_openai_compatible(self, user_query: str) -> Dict[str, Any]:
        """
        使用 OpenAI SDK (OpenAI, Groq, DeepSeek, Local)
        """
        try:
            from openai import OpenAI
            import httpx
            
            print(f"[LLM-OpenAI] Calling {self.model} at {self.base_url}...")
            
            try:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except TypeError as te:
                if "proxies" in str(te):
                    return {
                        "pipeline": [],
                        "reasoning": "版本衝突偵測: openai 與 httpx 套件版本不相容。",
                        "error": "Dependency Error: Please run `pip install -U openai httpx`"
                    }
                raise te

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks
            if "```" in raw_output:
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()

            data = json.loads(raw_output)
            
            if isinstance(data, dict) and "pipeline" in data:
                return {
                    "pipeline": data["pipeline"],
                    "reasoning": data.get("reasoning", "AI 未提供理由"),
                    "error": None
                }
            elif isinstance(data, list):
                return {
                    "pipeline": data,
                    "reasoning": "AI 回傳了舊版格式 (List Only)",
                    "error": None
                }
            else:
                raise ValueError(f"Unknown JSON format")
                
        except Exception as e:
            return {
                "pipeline": [], 
                "reasoning": f"API 連線失敗: {str(e)}",
                "error": str(e)
            }
    
    def _get_mock_suggestion(self, user_query: str) -> Dict[str, Any]:
        """
        Mock LLM 建議 - 提供詳細的 JSON 分析報告
        """
        query_lower = user_query.lower()
        pipeline = []
        reasoning = ""
        
        if "coin" in query_lower or "硬幣" in query_lower:
            pipeline = [
                {
                    "function": "resize", 
                    "params": {"width": 640},
                    "reason": "縮小圖片以提升處理速度與減少噪點影響"
                },
                {
                    "function": "gaussian_blur", 
                    "params": {"ksize": [9, 9], "sigmaX": 2},
                    "reason": "使用較大的核心 (9x9) 強力去除鍵盤紋理與背景雜訊"
                },
                {
                    "function": "hough_circles", 
                    "params": {"dp": 1.2, "minDist": 30, "param1": 50, "param2": 30, "minRadius": 20, "maxRadius": 80},
                    "reason": "使用霍夫圓變換偵測圓形物體，參數已針對一般硬幣大小優化"
                }
            ]
            reasoning = (
                "**[Mock 分析報告]**\n"
                "1. **需求理解**: 使用者希望偵測硬幣，通常背景可能會有雜訊 (如鍵盤、桌紋)。\n"
                "2. **預處理策略**: 為了避免誤判，我們首先將影像 Resize 至 640px 寬度，這能有效標準化硬幣的像素大小。接著使用強度較高的 GaussianBlur (9x9) 來抹除硬幣上的刻痕與背景紋理。\n"
                "3. **核心演算法**: 選擇 HoughCircles，這是經典且對遮擋具有一定魯棒性的圓形偵測演算法。\n"
                "4. **參數建議**: 設定 minRadius=20, maxRadius=80 以過濾掉過小(噪點)或過大(杯子)的圓形。"
            )
            
        elif "edge" in query_lower or "邊緣" in query_lower:
            pipeline = [
                {
                    "function": "gaussian_blur",
                    "params": {"ksize": [5, 5]},
                    "reason": "初步降噪，避免 Canny 偵測到過多假邊緣"
                },
                {
                    "function": "canny_edge",
                    "params": {"threshold1": 50, "threshold2": 150},
                    "reason": "使用 Canny 演算法，雙閾值設定 (50, 150) 以保留顯著邊緣"
                }
            ]
            reasoning = (
                "**[Mock 分析報告]**\n"
                "1. **需求理解**: 使用者需要提取影像中的邊緣特徵。\n"
                "2. **策略**: 邊緣偵測對噪點非常敏感，因此必須先進行平滑處理。\n"
                "3. **流程**: GaussianBlur (5x5) -> Canny Edge Detector。\n"
                "4. **參數**: Canny 的高低閾值比例設為 3:1 (150/50)，這是 OpenCV 官方推薦的經驗值。"
            )
            
        elif "denoise" in query_lower or "降噪" in query_lower:
            pipeline = [
                {
                    "function": "bilateral_filter",
                    "params": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75},
                    "reason": "使用雙邊濾波器，能在去除噪點的同時保留邊緣細節 (Edge-Preserving)"
                }
            ]
            reasoning = (
                "**[Mock 分析報告]**\n"
                "1. **需求理解**: 目標是去除噪點但不想讓畫面變模糊。\n"
                "2. **演算法選擇**: 放棄普通的高斯模糊，改用計算量較大但效果更好的 Bilateral Filter。\n"
                "3. **參數**: sigmaColor=75 允許較大的色彩差異混合，適合處理色彩噪聲。"
            )
            
        else:
            # Default Fallback
            pipeline = [
                {"function": "resize", "params": {"width": 480}},
                {"function": "gaussian_blur", "params": {"ksize": [3, 3]}},
                {"function": "canny_edge", "params": {"threshold1": 30, "threshold2": 100}}
            ]
            reasoning = (
                "**[Mock 分析報告]**\n"
                "1. **狀況**: 未偵測到特定關鍵字，系統提供一組通用的預處理與特徵提取流程。\n"
                "2. **建議**: 若您有特定目標 (如：人臉、車牌、顏色)，請在查詢中明確描述。"
            )

        return {
            "pipeline": pipeline,
            "reasoning": reasoning,
            "error": None
        }

    def _get_fallback_pipeline(self, user_query: str) -> List[Any]:
        return self._get_mock_suggestion(user_query)["pipeline"]

    def test_connection(self, api_key: str, base_url: str, model: str) -> tuple[bool, str]:
        """
        [NEW] 測試 API 連線有效性
        """
        if not api_key:
            return False, "API Key 為空"
            
        try:
            from openai import OpenAI
            # 使用傳入的參數建立臨時 Client，不影響全域設定
            # [Fix] Catch potential dependency errors here too
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
            except TypeError as te:
                if "proxies" in str(te):
                    return False, "版本衝突: 請執行 `pip install -U openai httpx`"
                raise te
            
            print(f"[Test] Pinging {base_url} with model {model}...")
            
            # 發送極簡請求 (max_tokens=1 節省成本)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            return True, "連線成功"
            
        except Exception as e:
            error_msg = str(e)
            # 簡化常見錯誤訊息
            if "401" in error_msg:
                return False, "認證失敗 (401): 請檢查 API Key"
            elif "404" in error_msg:
                return False, f"模型不存在 (404): 請確認 '{model}' 是否支援"
            elif "Connection" in error_msg:
                return False, "連線失敗: 請檢查 Base URL"
            else:
                return False, f"錯誤: {error_msg}"


# ==================== Bridge_Builder 區域 ====================
# ... (BridgeBuilder remains the same) ...


class BridgeBuilder:
    """
    Integration Specialist - 負責將骨架與血肉結合
    """
    
    def __init__(self, library_manager: LibraryManager):
        self.lib_manager = library_manager
        self.verilog_guru = VerilogGuru()
    
    def hydrate_pipeline(self, skeleton_list: List[Any]) -> List[Dict[str, Any]]:
        """
        將 LLM 的骨架（列表）附加資料庫資訊（血肉）
        支援輸入格式: List[str] 或 List[Dict] (新版含參數)
        """
        hydrated_pipeline = []
        
        for idx, item in enumerate(skeleton_list):
            # 正規化輸入
            if isinstance(item, str):
                func_name = item
                param_overrides = {}
                node_id = f"node_{idx}"
                inputs = []
            elif isinstance(item, dict):
                func_name = item.get("function", "Unknown")
                param_overrides = item.get("params", {})
                node_id = item.get("id", f"node_{idx}")
                inputs = item.get("inputs", []) # List of source IDs
            else:
                continue

            # Auto-link logic: If inputs is empty and not the first node, link to previous
            if not inputs and idx > 0:
                prev_id = hydrated_pipeline[-1]['id']
                inputs = [prev_id]
            elif not inputs and idx == 0:
                inputs = ["source"] # First node takes source image

            print(f"[Bridge] Hydrating '{func_name}' (ID: {node_id}, Inputs: {inputs})...")
            
            # Step 1: 查找演算法
            algo_data = self._lookup_algorithm(func_name)
            
            node_template = {
                "id": node_id,
                "inputs": inputs, # [NEW] Graph Support
                "name": func_name,
                "function": func_name,
                "category": "unknown",
                "description": "",
                "fpga_constraints": {},
                "parameters": {},
                "source": "unknown"
            }

            if algo_data:
                node_template.update({
                    "name": algo_data['name'],
                    "category": algo_data['category'],
                    "description": algo_data['description'],
                    "fpga_constraints": algo_data['fpga_constraints'],
                    "parameters": algo_data.get('parameters', {}).copy(),
                    "opencv_function": algo_data.get('opencv_function', None),
                    "source": algo_data.get('_library_type', 'official')
                })
                
                # Parameter Injection
                if param_overrides:
                    for p_key, p_val in param_overrides.items():
                        if p_key in node_template['parameters']:
                            node_template['parameters'][p_key]['default'] = p_val
                
            else:
                # Fallback for merge functions (add, bitwise_and) that might not be in library yet
                if func_name in ["add", "addWeighted", "bitwise_and", "bitwise_or", "bitwise_xor", "absdiff", "inRange"]:
                     node_template.update({
                        "category": "merge_logic",
                        "description": "Multi-input merge operation",
                        "fpga_constraints": {"estimated_clk": 100, "resource_usage": "Low", "latency_type": "Pipeline"},
                        "source": "builtin_merge"
                     })
                else:
                    fallback = self.verilog_guru.create_fallback_node(func_name, idx, len(skeleton_list))
                    node_template.update(fallback) # Merge fallback props
            
            hydrated_pipeline.append(node_template)
        
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
            # [Fix] Safe access to data structure
            libs = self.lib_manager.data.get('libraries', {})
            type_libs = libs.get(lib_type, {})
            
            for algo_id, algo_data in type_libs.items():
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

    def recalculate_pipeline_stats(self, pipeline: List[Dict], width: int, height: int):
        """
        [Dynamic FPGA Update] 根據影像解析度重新計算 Estimated CLK
        
        Args:
            pipeline: Pipeline 列表 (將被原地修改)
            width: 影像寬度
            height: 影像高度
        """
        print(f"[VerilogGuru] Recalculating CLK for resolution {width}x{height}")
        
        for node in pipeline:
            fpga = node.get('fpga_constraints', {})
            formula = fpga.get('clk_formula')
            
            if formula:
                try:
                    # 安全評估公式
                    # 允許的變數: width, height, pixels
                    allowed_locals = {
                        "width": width,
                        "height": height,
                        "pixels": width * height,
                        "math": __import__("math")
                    }
                    
                    # 簡單的安全檢查：不允許 __, import, exec 等
                    if any(bad in formula for bad in ["__", "exec", "eval", "import"]):
                        print(f"  [WARN] Unsafe formula detected in {node['name']}: {formula}")
                        continue
                        
                    new_clk = eval(formula, {"__builtins__": {}}, allowed_locals)
                    
                    # Update
                    old_clk = fpga.get('estimated_clk', 0)
                    fpga['estimated_clk'] = int(new_clk)
                    
                    print(f"  [{node['name']}] CLK updated: {old_clk} -> {int(new_clk)} (Formula: {formula})")
                    
                except Exception as e:
                    print(f"  [Error] Failed to eval formula for {node['name']}: {e}")
            else:
                # 若沒有 formula，且是 Unknown 節點，我們可能需要根據 VerilogGuru 的 heuristic 再次重算?
                # 目前先保留既有邏輯
                pass


# ==================== 整合介面 ====================

class LogicEngine:
    """
    整合 Prompt_Master + Bridge_Builder + Verilog_Guru
    """
    
    def __init__(self, lib_manager: LibraryManager = None):
        """
        初始化 Logic Engine
        
        Args:
            lib_manager: LibraryManager instance (if None, creates new one)
        """
        # 若未提供 LibraryManager，則建立新實例
        if lib_manager is None:
            self.lib_manager = LibraryManager()
        else:
            self.lib_manager = lib_manager
        
        # 建立子模組
        self.prompt_master = PromptMaster()
        self.bridge_builder = BridgeBuilder(self.lib_manager)
        self.verilog_guru = VerilogGuru()
    
    def process_user_query(self, user_query: str, use_mock_llm: bool = False) -> Dict[str, Any]:
        """
        完整處理使用者需求的主流程
        
        Returns:
            Dict: {
                "pipeline": List[Dict],
                "reasoning": str,
                "error": Optional[str]
            }
        """
        print(f"\n{'=' * 60}")
        print(f"User Query: '{user_query}'")
        print(f"{'=' * 60}")
        
        # Step 1: LLM 產生骨架 (Now returns dict with reasoning/error)
        llm_result = self.prompt_master.get_llm_suggestion(user_query, use_mock=use_mock_llm)
        
        skeleton = llm_result["pipeline"]
        reasoning = llm_result["reasoning"]
        error = llm_result["error"]
        
        print(f"\n[Step 1] Skeleton: {skeleton}")
        print(f"[Step 1] Reasoning: {reasoning}")
        
        # 如果有錯誤，直接回傳
        if error:
            return {
                "pipeline": [],
                "reasoning": reasoning,
                "error": error
            }

        # Step 2: 附加血肉
        pipeline = self.bridge_builder.hydrate_pipeline(skeleton)
        print(f"\n[Step 2] Hydrated {len(pipeline)} nodes")
        
        # Step 3: 顯示摘要
        self._print_pipeline_summary(pipeline)
        
        return {
            "pipeline": pipeline,
            "reasoning": reasoning,
            "error": None
        }
    
    def _print_pipeline_summary(self, pipeline: List[Dict]):
        """
        顯示 Pipeline 摘要
        """
        print(f"\n{'=' * 60}")
        print("Pipeline Summary:")
        print(f"{'=' * 60}")
        
        for node in pipeline:
            fpga = node['fpga_constraints']
            print(f"[{node['id']}] {node['name']}")
            print(f"  - CLK: {fpga['estimated_clk']}, Resource: {fpga['resource_usage']}, Source: {node['source']}")
            if '_warning' in node:
                print(f"  [WARN] {node['_warning']}")
        
        total_clk = sum(n['fpga_constraints']['estimated_clk'] for n in pipeline)
        print(f"\nTotal Estimated CLK: {total_clk}")
        print(f"{'=' * 60}\n")


# ==================== 測試與範例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Logic Engine Test Suite")
    print("=" * 60)
    
    # 初始化引擎（使用 Mock LLM）
    engine = LogicEngine()
    
    # 測試案例 1: 硬幣偵測
    print("\n[Test Case 1] Coin Detection on Keyboard")
    pipeline1 = engine.process_user_query(
        "Detect coins on a noisy keyboard background",
        use_mock_llm=True
    )
    
    # 導出 JSON
    if pipeline1.get("pipeline"):
        engine.bridge_builder.export_to_json(pipeline1["pipeline"], "test_pipeline_coins.json")
    else:
        print("[Test] Skipping export due to error in pipeline generation")
    
    # 測試案例 2: 邊緣偵測
    print("\n[Test Case 2] Edge Detection")
    pipeline2 = engine.process_user_query(
        "Detect edges in a blurry image",
        use_mock_llm=True
    )
    
    # 測試案例 3: 未知函數（測試 Fallback）
    print("\n[Test Case 3] Unknown Function (Fallback Test)")
    # 手動建立一個包含未知函數的 Skeleton
    unknown_skeleton = ["GaussianBlur", "UnknownDeepLearningModel", "Canny"]
    pipeline3 = engine.bridge_builder.hydrate_pipeline(unknown_skeleton)
    engine._print_pipeline_summary(pipeline3)
    
    print("\n" + "=" * 60)
    print("All tests completed! Check 'test_pipeline_coins.json'")
    print("=" * 60)
