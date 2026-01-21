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
    
    # Prompt 模板（升級為支援 Reasoning 與參數建議）
    SYSTEM_PROMPT = """You are an expert Computer Vision algorithm advisor for FPGA implementation.

**CRITICAL RULES:**
1. You MUST return a JSON object with two fields: "reasoning" and "pipeline".
2. "reasoning": Explain WHY you chose these algorithms and parameters. Be specific.
3. "pipeline": An array of objects, each with "function" (name) and "params" (optional overrides).
4. Use standard OpenCV function names or known aliases.
5. If the user asks for "Coin Detection", prioritize "advanced_coin_detection" node.

**Output Format (STRICT JSON):**
{
  "reasoning": "Since the image is noisy, I suggest a larger blur kernel...",
  "pipeline": [
    { "function": "GaussianBlur", "params": { "ksize": [7, 7] } },
    { "function": "Canny", "params": { "threshold1": 30, "threshold2": 100 } }
  ]
}

**Example Input:** "Detect coins on a keyboard"
**Example Output:**
{
  "reasoning": "For coins on complex backgrounds, we should use the specialized advanced detector which handles resize and hough transform robustly.",
  "pipeline": [
    { "function": "advanced_coin_detection", "params": { "min_radius": 25, "max_radius": 90 } }
  ]
}
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
    
    def get_llm_suggestion(self, user_query: str, use_mock: bool = False) -> List[Any]:
        """
        獲取 LLM 建議的演算法列表 (支援舊版 List[str] 與新版 List[Dict])
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
                temperature=0.3,
                max_tokens=500
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # 解析 JSON
            try:
                data = json.loads(raw_output)
                
                # Case 1: 新版 Rich Output (Object with reasoning)
                if isinstance(data, dict) and "pipeline" in data:
                    print(f"\n[AI Reasoning] {data.get('reasoning', 'No reasoning provided')}")
                    return data["pipeline"] # 回傳 pipeline list (包含 params)
                
                # Case 2: 舊版 Simple Output (List of strings)
                elif isinstance(data, list):
                    print("[Prompt_Master] LLM returned simple list format.")
                    return data
                
                else:
                    raise ValueError("Unknown JSON format")
                
            except json.JSONDecodeError as e:
                print(f"[Error] LLM returned invalid JSON: {raw_output}")
                return self._get_fallback_suggestion(user_query)
        
        except Exception as e:
            print(f"[Error] LLM API call failed: {e}")
            return self._get_fallback_suggestion(user_query)
    
    def _get_mock_suggestion(self, user_query: str) -> List[Any]:
        """
        Mock LLM 建議 (升級版，回傳 List[Dict] 以支援參數)
        """
        query_lower = user_query.lower()
        
        if "coin" in query_lower or "硬幣" in query_lower:
            print("[Mock] Detected 'coin' -> Using Advanced Coin Detector")
            return [{"function": "advanced_coin_detection"}]
            
        elif "edge" in query_lower or "邊緣" in query_lower:
            return ["GaussianBlur", "Canny"] # 舊版格式兼容測試
            
        elif "denoise" in query_lower or "降噪" in query_lower:
            return [{"function": "GaussianBlur"}, {"function": "Morphological_Open"}]
            
        else:
            return ["GaussianBlur", "Canny"]

    def _get_fallback_suggestion(self, user_query: str) -> List[Any]:
        print("[Warning] Using fallback suggestion mechanism.")
        return self._get_mock_suggestion(user_query)

# ==================== Bridge_Builder 區域 ====================

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
            # 正規化輸入：取得名稱與參數覆蓋
            if isinstance(item, str):
                func_name = item
                param_overrides = {}
            elif isinstance(item, dict):
                func_name = item.get("function", "Unknown")
                param_overrides = item.get("params", {})
            else:
                continue

            print(f"[Bridge_Builder] Hydrating '{func_name}'...")
            
            # Step 1: 查找演算法
            algo_data = self._lookup_algorithm(func_name)
            
            if algo_data:
                # 複製以避免汙染原始資料庫
                node = {
                    "id": f"node_{idx}",
                    "name": algo_data['name'],
                    "function": func_name,
                    "category": algo_data['category'],
                    "description": algo_data['description'],
                    "fpga_constraints": algo_data['fpga_constraints'],
                    "parameters": algo_data.get('parameters', {}).copy(), # Deep copy structure
                    "opencv_function": algo_data.get('opencv_function', None),
                    "source": algo_data.get('_library_type', 'official'),
                    "next_node_id": f"node_{idx + 1}" if idx < len(skeleton_list) - 1 else None
                }
                
                # Step 1.5: 套用 LLM 建議的參數 (Parameter Injection)
                if param_overrides:
                    print(f"  [AI] Applying suggested parameters: {param_overrides}")
                    for p_key, p_val in param_overrides.items():
                        if p_key in node['parameters']:
                            node['parameters'][p_key]['default'] = p_val
                
                print(f"  [OK] Found in database ({node['source']})")
                
            else:
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
    
    def process_user_query(self, user_query: str, use_mock_llm: bool = False) -> List[Dict]:
        """
        完整處理使用者需求的主流程
        
        Args:
            user_query: 使用者輸入的自然語言需求
            use_mock_llm: 是否使用 Mock LLM（測試用）
        
        Returns:
            List[Dict]: 完整的 Hydrated Pipeline
        """
        print(f"\n{'=' * 60}")
        print(f"User Query: '{user_query}'")
        print(f"{'=' * 60}")
        
        # Step 1: LLM 產生骨架
        skeleton = self.prompt_master.get_llm_suggestion(user_query, use_mock=use_mock_llm)
        print(f"\n[Step 1] Skeleton: {skeleton}")
        
        # Step 2: 附加血肉
        pipeline = self.bridge_builder.hydrate_pipeline(skeleton)
        print(f"\n[Step 2] Hydrated {len(pipeline)} nodes")
        
        # Step 3: 顯示摘要
        self._print_pipeline_summary(pipeline)
        
        return pipeline
    
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
    engine.bridge_builder.export_to_json(pipeline1, "test_pipeline_coins.json")
    
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
