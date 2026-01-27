"""
Project Manager for NKUST AoV Tool
負責人：Legacy_Keeper (Database Librarian) + Project_Manager (Captain_Vision)

職責：
1. 專案匯出至 JSON（供學弟妹傳承）
2. 專案從 JSON 匯入（重現前人成果）
3. 版本控制與相容性檢查
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class ProjectManager:
    """
    專案管理器 - 處理 AoV 專案的存取
    
    確保學弟妹能夠：
    1. 保存自己的實驗成果
    2. 載入學長姐的專案學習
    3. 追溯專案的歷史脈絡
    """
    
    # 專案檔案格式版本
    PROJECT_SCHEMA_VERSION = "1.0"
    
    @staticmethod
    def export_project_to_json(
        pipeline_data: List[Dict[str, Any]],
        author: str = "Unknown",
        notes: str = "",
        project_name: str = "Untitled_Project"
    ) -> str:
        """
        將專案匯出為 JSON 字串
        
        Args:
            pipeline_data: Pipeline 節點列表（來自 st.session_state['pipeline']）
            author: 作者姓名（學生/研究生）
            notes: 實驗備註
            project_name: 專案名稱
        
        Returns:
            str: 完整的 JSON 字串（可直接下載）
        """
        # 計算硬體摘要
        total_clk = sum(
            node.get('fpga_constraints', {}).get('estimated_clk', 0) 
            for node in pipeline_data
        )
        total_nodes = len(pipeline_data)
        
        # 統計資源消耗
        resource_summary = {
            'Low': 0,
            'Medium': 0,
            'High': 0,
            'Very High': 0,
            'Unknown': 0
        }
        
        for node in pipeline_data:
            resource = node.get('fpga_constraints', {}).get('resource_usage', 'Unknown')
            if resource in resource_summary:
                resource_summary[resource] += 1
        
        # 建立完整專案結構
        project_data = {
            "meta": {
                "project_name": project_name,
                "author": author,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notes": notes,
                "version": ProjectManager.PROJECT_SCHEMA_VERSION,
                "schema_type": "NKUST_AoV_Project",
                "lab": "NKUST Vision Lab"
            },
            
            "pipeline": pipeline_data,
            
            "hardware_summary": {
                "total_clk": total_clk,
                "total_nodes": total_nodes,
                "resource_distribution": resource_summary,
                "estimated_fps_1080p": ProjectManager._estimate_fps(total_clk),
                "complexity_level": ProjectManager._assess_complexity(total_clk, total_nodes)
            },
            
            "_readme": {
                "purpose": "此檔案為 NKUST AoV Tool 專案檔，包含完整的演算法流程與 FPGA 約束資訊。",
                "how_to_use": "請在 AoV Tool 中使用「匯入專案」功能載入此檔案。",
                "maintainer": "NKUST Vision Lab"
            }
        }
        
        # 轉換為 JSON（繁體中文友善）
        return json.dumps(project_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_project_from_json(json_content: str) -> Dict[str, Any]:
        """
        從 JSON 字串載入專案
        
        Args:
            json_content: JSON 字串內容
        
        Returns:
            Dict: 包含以下欄位的字典
                - success: bool（是否成功）
                - pipeline: List[Dict]（Pipeline 資料）
                - meta: Dict（元資料）
                - error: str（錯誤訊息，若有）
        
        Raises:
            不拋出例外，所有錯誤透過 return dict 回報
        """
        try:
            # 解析 JSON
            project_data = json.loads(json_content)
            
            # 驗證必要欄位
            if "meta" not in project_data:
                return {
                    "success": False,
                    "error": "缺少 'meta' 欄位。此檔案可能不是有效的 AoV 專案檔。"
                }
            
            if "pipeline" not in project_data:
                return {
                    "success": False,
                    "error": "缺少 'pipeline' 欄位。此檔案可能不是有效的 AoV 專案檔。"
                }
            
            # 版本相容性檢查
            file_version = project_data["meta"].get("version", "unknown")
            if file_version != ProjectManager.PROJECT_SCHEMA_VERSION:
                # 警告但不阻止（向後相容）
                print(f"[Warning] Project version mismatch: {file_version} vs {ProjectManager.PROJECT_SCHEMA_VERSION}")
            
            # 驗證 Pipeline 結構
            pipeline = project_data["pipeline"]
            if not isinstance(pipeline, list):
                return {
                    "success": False,
                    "error": "Pipeline 格式錯誤：應為陣列。"
                }
            
            if len(pipeline) == 0:
                return {
                    "success": False,
                    "error": "Pipeline 為空，無法載入。"
                }
            
            # 成功！
            return {
                "success": True,
                "pipeline": pipeline,
                "meta": project_data["meta"],
                "hardware_summary": project_data.get("hardware_summary", {}),
                "error": None
            }
        
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON 解析失敗：{str(e)}"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"載入失敗：{str(e)}"
            }
    
    @staticmethod
    def load_project_from_file(file_path: str) -> Dict[str, Any]:
        """
        從檔案路徑載入專案
        
        Args:
            file_path: JSON 檔案路徑
        
        Returns:
            Dict: 同 load_project_from_json
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            return ProjectManager.load_project_from_json(json_content)
        
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"檔案不存在：{file_path}"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"讀取檔案失敗：{str(e)}"
            }
    
    @staticmethod
    def save_project_to_file(
        pipeline_data: List[Dict[str, Any]],
        output_path: str,
        author: str = "Unknown",
        notes: str = "",
        project_name: str = "Untitled_Project"
    ) -> bool:
        """
        直接儲存專案至檔案
        
        Args:
            pipeline_data: Pipeline 資料
            output_path: 輸出檔案路徑
            author: 作者
            notes: 備註
            project_name: 專案名稱
        
        Returns:
            bool: 是否成功
        """
        try:
            json_str = ProjectManager.export_project_to_json(
                pipeline_data, author, notes, project_name
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            print(f"[ProjectManager] Project saved to: {output_path}")
            return True
        
        except Exception as e:
            print(f"[Error] Failed to save project: {e}")
            return False
    
    # ==================== 輔助函數 ====================
    
    @staticmethod
    def _estimate_fps(total_clk: int, clock_freq_mhz: int = 100) -> float:
        """
        估算 FPS（假設 1080p 影像）
        
        Args:
            total_clk: 總時脈數
            clock_freq_mhz: FPGA 時脈頻率（MHz）
        
        Returns:
            float: 預估 FPS
        """
        if total_clk == 0:
            return 0.0
        
        # 1080p = 1920x1080 = 2,073,600 pixels
        pixels_per_frame = 1920 * 1080
        
        # Throughput = clock_freq / (total_clk * pixels_per_frame)
        # 簡化計算（粗估）
        cycles_per_frame = total_clk * pixels_per_frame / 1e6  # 轉為 M cycles
        fps = (clock_freq_mhz * 1e6) / (total_clk * pixels_per_frame)
        
        return round(fps, 2)
    
    @staticmethod
    def _assess_complexity(total_clk: int, total_nodes: int) -> str:
        """
        評估 Pipeline 複雜度
        
        Returns:
            str: "Simple" / "Moderate" / "Complex" / "Very Complex"
        """
        if total_clk < 500:
            return "Simple"
        elif total_clk < 2000:
            return "Moderate"
        elif total_clk < 10000:
            return "Complex"
        else:
            return "Very Complex"
    
    @staticmethod
    def generate_project_filename(author: str = "Unknown") -> str:
        """
        自動生成專案檔名
        
        格式: nkust_aov_{author}_{timestamp}.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 移除特殊字元
        safe_author = "".join(c if c.isalnum() else "_" for c in author)
        return f"nkust_aov_{safe_author}_{timestamp}.json"


# ==================== 測試 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ProjectManager Test Suite")
    print("=" * 60)
    
    # 建立測試 Pipeline
    test_pipeline = [
        {
            "id": "node_0",
            "name": "Gaussian Blur",
            "function": "GaussianBlur",
            "category": "preprocessing",
            "fpga_constraints": {
                "estimated_clk": 150,
                "resource_usage": "Medium",
                "latency_type": "Pipeline"
            },
            "parameters": {"ksize": {"default": [5, 5]}}
        },
        {
            "id": "node_1",
            "name": "Canny Edge Detection",
            "function": "Canny",
            "category": "edge_detection",
            "fpga_constraints": {
                "estimated_clk": 450,
                "resource_usage": "High",
                "latency_type": "Pipeline"
            },
            "parameters": {
                "threshold1": {"default": 50},
                "threshold2": {"default": 150}
            }
        }
    ]
    
    # Test 1: 匯出專案
    print("\n[Test 1] Export Project")
    json_str = ProjectManager.export_project_to_json(
        test_pipeline,
        author="Test_Student",
        notes="這是一個測試專案，用於驗證邊緣檢測功能。",
        project_name="Edge_Detection_Test"
    )
    
    print(json_str[:500] + "...")
    
    # Test 2: 儲存至檔案
    print("\n[Test 2] Save to File")
    success = ProjectManager.save_project_to_file(
        test_pipeline,
        "test_project.json",
        author="Test_Student",
        notes="測試專案"
    )
    print(f"Save success: {success}")
    
    # Test 3: 從檔案載入
    print("\n[Test 3] Load from File")
    result = ProjectManager.load_project_from_file("test_project.json")
    
    if result["success"]:
        print(f"✓ Loaded successfully!")
        print(f"  Author: {result['meta']['author']}")
        print(f"  Nodes: {len(result['pipeline'])}")
        print(f"  Total CLK: {result['hardware_summary']['total_clk']}")
    else:
        print(f"✗ Load failed: {result['error']}")
    
    # Test 4: 錯誤處理（無效 JSON）
    print("\n[Test 4] Error Handling (Invalid JSON)")
    result_bad = ProjectManager.load_project_from_json("{invalid json")
    print(f"Expected error: {result_bad['error']}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
