"""
Library Manager for NKUST AoV Tool
負責人：Legacy_Keeper (Database Librarian)

職責：
1. 載入/儲存 tech_lib.json
2. 提供演算法查詢介面
3. 支援新增學生貢獻的演算法
4. 版本控制與相容性檢查
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import copy


class LibraryManager:
    """
    演算法庫管理器
    
    Single Source of Truth for all OpenCV algorithms with FPGA constraints.
    """
    
    def __init__(self, lib_path: str = "tech_lib.json"):
        """
        初始化庫管理器
        
        Args:
            lib_path: tech_lib.json 的路徑
        """
        self.lib_path = Path(lib_path)
        # 初始化預設結構，防止 load 失敗時後續崩潰
        self.data: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "description": "Fallback Library",
            "libraries": {
                "official": {},
                "contributed": {}
            },
            "_metadata": {
                "total_algorithms": 0,
                "official_count": 0,
                "contributed_count": 0,
                "maintainers": [],
                "license": "Unknown"
            }
        }
        self.load()
    
    def load(self) -> bool:
        """
        從磁碟載入演算法庫
        
        Returns:
            bool: 載入成功與否
        """
        try:
            if not self.lib_path.exists():
                print(f"[Warning] {self.lib_path} not found. Creating default library...")
                self._create_default_library()
                return False
            
            with open(self.lib_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            print(f"[LibraryManager] Loaded library v{self.data.get('schema_version', 'unknown')}")
            print(f"  - Official algorithms: {len(self.data['libraries']['official'])}")
            print(f"  - Contributed algorithms: {len(self.data['libraries']['contributed'])}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"[Error] JSON parsing failed: {e}")
            return False
        except Exception as e:
            print(f"[Error] Failed to load library: {e}")
            return False
    
    def save(self) -> bool:
        """
        將當前庫狀態儲存至磁碟
        
        Returns:
            bool: 儲存成功與否
        """
        try:
            # Update metadata
            self.data['last_updated'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")
            self.data['_metadata']['total_algorithms'] = (
                len(self.data['libraries']['official']) + 
                len(self.data['libraries']['contributed'])
            )
            self.data['_metadata']['official_count'] = len(self.data['libraries']['official'])
            self.data['_metadata']['contributed_count'] = len(self.data['libraries']['contributed'])
            
            with open(self.lib_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            
            print(f"[LibraryManager] Library saved successfully.")
            return True
            
        except Exception as e:
            print(f"[Error] Failed to save library: {e}")
            return False
    
    def get_algorithm(self, algo_id: str, library_type: str = "official") -> Optional[Dict]:
        """
        獲取特定演算法的完整資訊
        
        Args:
            algo_id: 演算法 ID（例如 'gaussian_blur'）
            library_type: 'official' 或 'contributed'
        
        Returns:
            Dict: 演算法資訊字典，若不存在則回傳 None
        """
        try:
            return copy.deepcopy(self.data['libraries'][library_type].get(algo_id))
        except KeyError:
            return None
    
    def list_algorithms(self, library_type: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        """
        列出演算法清單
        
        Args:
            library_type: 若指定，僅回傳該類型的演算法（'official' / 'contributed'）
            category: 若指定，僅回傳該類別的演算法（'preprocessing' / 'edge_detection' 等）
        
        Returns:
            List[Dict]: 演算法資訊列表
        """
        results = []
        
        libraries_to_search = ['official', 'contributed'] if library_type is None else [library_type]
        
        for lib_type in libraries_to_search:
            for algo_id, algo_data in self.data['libraries'][lib_type].items():
                if category is None or algo_data.get('category') == category:
                    algo_copy = copy.deepcopy(algo_data)
                    algo_copy['_library_type'] = lib_type
                    algo_copy['_algo_id'] = algo_id
                    results.append(algo_copy)
        
        return results
    
    def add_algorithm(self, algo_id: str, algo_data: Dict, library_type: str = "contributed") -> bool:
        """
        新增演算法至庫中（通常用於學生貢獻）
        
        Args:
            algo_id: 演算法唯一 ID
            algo_data: 演算法完整資訊
            library_type: 預設為 'contributed'
        
        Returns:
            bool: 新增成功與否
        """
        if algo_id in self.data['libraries'][library_type]:
            print(f"[Warning] Algorithm '{algo_id}' already exists in {library_type} library.")
            return False
        
        # 驗證必要欄位
        required_fields = ['name', 'category', 'opencv_function', 'fpga_constraints']
        for field in required_fields:
            if field not in algo_data:
                print(f"[Error] Missing required field: {field}")
                return False
        
        # 自動添加時間戳
        if 'date_added' not in algo_data:
            algo_data['date_added'] = datetime.now().strftime("%Y-%m-%d")
        
        self.data['libraries'][library_type][algo_id] = algo_data
        print(f"[LibraryManager] Added '{algo_id}' to {library_type} library.")
        
        return self.save()
    
    def get_fpga_constraints(self, algo_id: str, library_type: str = "official") -> Optional[Dict]:
        """
        快速獲取演算法的 FPGA 約束資訊
        
        Args:
            algo_id: 演算法 ID
            library_type: 'official' 或 'contributed'
        
        Returns:
            Dict: FPGA 約束字典
        """
        algo = self.get_algorithm(algo_id, library_type)
        return algo.get('fpga_constraints') if algo else None
    
    def search_by_name(self, keyword: str) -> List[Dict]:
        """
        根據名稱或描述關鍵字搜尋演算法
        
        Args:
            keyword: 搜尋關鍵字
        
        Returns:
            List[Dict]: 符合的演算法列表
        """
        results = []
        keyword_lower = keyword.lower()
        
        for lib_type in ['official', 'contributed']:
            for algo_id, algo_data in self.data['libraries'][lib_type].items():
                if (keyword_lower in algo_data.get('name', '').lower() or
                    keyword_lower in algo_data.get('description', '').lower()):
                    algo_copy = copy.deepcopy(algo_data)
                    algo_copy['_library_type'] = lib_type
                    algo_copy['_algo_id'] = algo_id
                    results.append(algo_copy)
        
        return results
    
    def _create_default_library(self):
        """
        建立預設的空庫結構
        """
        self.data = {
            "schema_version": "1.0.0",
            "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "description": "NKUST Vision Lab - OpenCV Algorithm Library",
            "libraries": {
                "official": {},
                "contributed": {}
            },
            "_metadata": {
                "total_algorithms": 0,
                "official_count": 0,
                "contributed_count": 0,
                "maintainers": ["NKUST_Vision_Lab"],
                "license": "MIT"
            }
        }
        self.save()
    
    def export_for_llm(self, library_type: Optional[str] = None) -> str:
        """
        導出適合給 LLM 閱讀的簡化版本
        
        專為 Prompt_Master (LLM_Orchestrator) 設計
        只保留演算法名稱、類別、描述，移除詳細參數
        
        Args:
            library_type: 若指定，僅導出該類型
        
        Returns:
            str: JSON 字串
        """
        simplified = []
        
        libraries_to_export = ['official', 'contributed'] if library_type is None else [library_type]
        
        for lib_type in libraries_to_export:
            for algo_id, algo_data in self.data['libraries'][lib_type].items():
                simplified.append({
                    "algo_id": algo_id,
                    "name": algo_data.get('name'),
                    "category": algo_data.get('category'),
                    "description": algo_data.get('description'),
                    "library": lib_type
                })
        
        return json.dumps(simplified, indent=2, ensure_ascii=False)


# ==================== 測試與範例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("LibraryManager Test Suite")
    print("=" * 60)
    
    # 1. 初始化
    manager = LibraryManager("tech_lib.json")
    
    # 2. 列出所有 preprocessing 類別的演算法
    print("\n[Test 1] List all preprocessing algorithms:")
    preprocessing_algos = manager.list_algorithms(category="preprocessing")
    for algo in preprocessing_algos:
        print(f"  - {algo['name']} ({algo['_algo_id']}) from {algo['_library_type']}")
    
    # 3. 獲取 Canny 的 FPGA 約束
    print("\n[Test 2] Get FPGA constraints for Canny:")
    canny_fpga = manager.get_fpga_constraints("canny_edge")
    if canny_fpga:
        print(f"  - Estimated CLK: {canny_fpga['estimated_clk']}")
        print(f"  - Resource Usage: {canny_fpga['resource_usage']}")
        print(f"  - Latency Type: {canny_fpga['latency_type']}")
    
    # 4. 搜尋關鍵字
    print("\n[Test 3] Search for 'blur':")
    results = manager.search_by_name("blur")
    for r in results:
        print(f"  - {r['name']}")
    
    # 5. 新增一個學生貢獻的演算法（測試）
    print("\n[Test 4] Add a new contributed algorithm:")
    new_algo = {
        "name": "Bilateral Filter",
        "category": "preprocessing",
        "description": "保邊濾波器（學生測試）",
        "opencv_function": "cv2.bilateralFilter",
        "fpga_constraints": {
            "estimated_clk": 280,
            "resource_usage": "Medium",
            "latency_type": "Pipeline"
        },
        "author": "Test_Student"
    }
    manager.add_algorithm("bilateral_test", new_algo, "contributed")
    
    # 6. 導出給 LLM 的簡化版本
    print("\n[Test 5] Export simplified version for LLM:")
    llm_export = manager.export_for_llm()
    print(llm_export[:500] + "...")  # 只顯示前 500 字元
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
