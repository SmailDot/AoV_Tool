"""
Multi-Format Import Parser for NKUST AoV Tool
負責人：Data_Migration_Specialist

職責：
1. 支援多種格式匯入 (JSON, DOT, TXT)
2. 自動偵測檔案格式
3. 統一的Pipeline資料結構輸出
"""

import json
import re
from typing import List, Dict, Any, Optional
from library_manager import LibraryManager


class ImportParser:
    """
    多格式匯入解析器
    支援：JSON (完整專案)、DOT (Graphviz)、TXT (文字工單)
    """
    
    def __init__(self, lib_manager: Optional[LibraryManager] = None):
        """
        初始化解析器
        
        Args:
            lib_manager: LibraryManager實例，用於補全節點資訊
        """
        self.lib_manager = lib_manager or LibraryManager()
    
    @staticmethod
    def detect_format(content: str) -> str:
        """
        自動偵測檔案格式
        
        Args:
            content: 檔案內容
        
        Returns:
            str: 'json', 'dot', 或 'txt'
        """
        content_stripped = content.strip()
        
        # 檢查JSON
        if content_stripped.startswith('{') and content_stripped.endswith('}'):
            try:
                json.loads(content)
                return 'json'
            except:
                pass
        
        # 檢查DOT (Graphviz)
        if 'digraph' in content.lower() or 'graph' in content.lower():
            if '->' in content or '--' in content:
                return 'dot'
        
        # 預設為文字工單格式
        return 'txt'
    
    def parse(self, content: str, format_type: Optional[str] = None) -> Dict[str, Any]:
        """
        統一解析入口
        
        Args:
            content: 檔案內容
            format_type: 指定格式 (若為None則自動偵測)
        
        Returns:
            Dict: 包含 'success', 'pipeline', 'format', 'error' 等欄位
        """
        if format_type is None:
            format_type = self.detect_format(content)
        
        try:
            if format_type == 'json':
                pipeline = self.parse_json(content)
            elif format_type == 'dot':
                pipeline = self.parse_dot(content)
            elif format_type == 'txt':
                pipeline = self.parse_text_workflow(content)
            else:
                return {
                    'success': False,
                    'error': f"不支援的格式: {format_type}"
                }
            
            return {
                'success': True,
                'pipeline': pipeline,
                'format': format_type,
                'node_count': len(pipeline)
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'format': format_type
            }
    
    def parse_json(self, content: str) -> List[Dict]:
        """
        解析JSON格式（現有project_manager格式）
        
        Args:
            content: JSON字串
        
        Returns:
            List[Dict]: Pipeline節點列表
        """
        data = json.loads(content)
        
        # 支援完整專案格式
        if 'pipeline' in data:
            return data['pipeline']
        
        # 支援純Pipeline陣列
        elif isinstance(data, list):
            return data
        
        else:
            raise ValueError("JSON格式不正確：缺少'pipeline'欄位")
    
    def parse_dot(self, content: str) -> List[Dict]:
        """
        解析Graphviz DOT格式
        
        格式範例:
        digraph pipeline {
            node_0 [label="GaussianBlur\\nksize=9", clk=150, resource="Medium"]
            node_1 [label="Canny\\nT1=50,T2=150", clk=450]
            node_0 -> node_1
        }
        
        Args:
            content: DOT字串
        
        Returns:
            List[Dict]: Pipeline節點列表
        """
        nodes = {}
        edges = []
        
        # 解析節點定義
        # 格式: node_0 [label="Name\nparam=value", clk=123, ...]
        node_pattern = r'(\w+)\s*\[(.*?)\]'
        
        for match in re.finditer(node_pattern, content, re.DOTALL):
            node_id = match.group(1)
            attributes_str = match.group(2)
            
            # 解析屬性
            node_data = self._parse_dot_attributes(node_id, attributes_str)
            nodes[node_id] = node_data
        
        # 解析邊
        # 格式: node_0 -> node_1
        edge_pattern = r'(\w+)\s*->\s*(\w+)'
        
        for match in re.finditer(edge_pattern, content):
            from_node = match.group(1)
            to_node = match.group(2)
            edges.append((from_node, to_node))
        
        # 建立Pipeline（按邊的順序）
        if not edges:
            # 無邊：按節點ID排序
            sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])
            return [node_data for _, node_data in sorted_nodes]
        
        # 有邊：建立有序列表
        pipeline = self._build_pipeline_from_edges(nodes, edges)
        
        return pipeline
    
    def _parse_dot_attributes(self, node_id: str, attributes_str: str) -> Dict:
        """
        解析DOT節點屬性
        
        Args:
            node_id: 節點ID
            attributes_str: 屬性字串 (例如: label="Name\nkey=val", clk=150)
        
        Returns:
            Dict: 節點資料
        """
        node_data = {
            'id': node_id,
            'name': node_id,
            'function': node_id.replace('node_', ''),
            'category': 'unknown',
            'parameters': {},
            'fpga_constraints': {
                'estimated_clk': 999,
                'resource_usage': 'Unknown',
                'latency_type': 'Unknown'
            },
            'source': 'dot_import'
        }
        
        # 解析label屬性
        label_match = re.search(r'label\s*=\s*"([^"]+)"', attributes_str)
        if label_match:
            label_content = label_match.group(1)
            lines = label_content.split('\\n')
            
            # 第一行是函數名稱
            if lines:
                func_name = lines[0].strip()
                node_data['name'] = func_name
                node_data['function'] = func_name
            
            # 後續行是參數
            for line in lines[1:]:
                if '=' in line or ':' in line:
                    param_match = re.match(r'(\w+)\s*[:=]\s*(.+)', line.strip())
                    if param_match:
                        param_name = param_match.group(1)
                        param_value = param_match.group(2).strip()
                        
                        # 嘗試轉換型別
                        try:
                            param_value = json.loads(param_value)
                        except:
                            pass
                        
                        node_data['parameters'][param_name] = {'default': param_value}
        
        # 解析CLK
        clk_match = re.search(r'clk\s*=\s*(\d+)', attributes_str)
        if clk_match:
            node_data['fpga_constraints']['estimated_clk'] = int(clk_match.group(1))
        
        # 解析resource
        resource_match = re.search(r'resource\s*=\s*"([^"]+)"', attributes_str)
        if resource_match:
            node_data['fpga_constraints']['resource_usage'] = resource_match.group(1)
        
        # 從資料庫補全資訊
        self._enrich_node_from_library(node_data)
        
        return node_data
    
    def _build_pipeline_from_edges(self, nodes: Dict, edges: List[tuple]) -> List[Dict]:
        """
        根據邊建立有序Pipeline
        
        Args:
            nodes: 節點字典
            edges: 邊列表 [(from, to), ...]
        
        Returns:
            List[Dict]: 有序節點列表
        """
        # 建立鄰接表
        graph = {node_id: [] for node_id in nodes}
        for from_node, to_node in edges:
            graph[from_node].append(to_node)
        
        # 尋找起始節點（入度為0）
        in_degree = {node_id: 0 for node_id in nodes}
        for from_node, to_node in edges:
            in_degree[to_node] += 1
        
        start_nodes = [node_id for node_id, degree in in_degree.items() if degree == 0]
        
        # 拓撲排序
        pipeline = []
        visited = set()
        
        def dfs(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            pipeline.append(nodes[node_id])
            
            for next_node in graph.get(node_id, []):
                dfs(next_node)
        
        for start in start_nodes:
            dfs(start)
        
        # 重新編號
        for i, node in enumerate(pipeline):
            node['id'] = f'node_{i}'
        
        return pipeline
    
    def parse_text_workflow(self, content: str) -> List[Dict]:
        """
        解析文字工單格式
        
        格式範例:
        # Pipeline Title
        
        1. GaussianBlur
           - ksize: 9
           - sigma: 2.0
           - CLK: 150
        
        2. Canny
           - threshold1: 50
           - threshold2: 150
           - CLK: 450
        
        Args:
            content: 文字工單內容
        
        Returns:
            List[Dict]: Pipeline節點列表
        """
        lines = content.split('\n')
        pipeline = []
        current_node = None
        
        for line in lines:
            # 跳過註解和空行
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            # 檢查是否為節點定義 (1., 2., ...)
            node_match = re.match(r'^(\d+)\.\s+(\w+)', line.strip())
            if node_match:
                # 儲存前一個節點
                if current_node:
                    pipeline.append(current_node)
                
                # 建立新節點
                idx = int(node_match.group(1)) - 1
                func_name = node_match.group(2)
                
                current_node = {
                    'id': f'node_{idx}',
                    'name': func_name,
                    'function': func_name,
                    'category': 'unknown',
                    'parameters': {},
                    'fpga_constraints': {
                        'estimated_clk': 999,
                        'resource_usage': 'Unknown',
                        'latency_type': 'Unknown'
                    },
                    'source': 'txt_import'
                }
                
                continue
            
            # 檢查是否為參數定義 (- key: value)
            param_match = re.match(r'^\s*-\s+(\w+):\s*(.+)', line.strip())
            if param_match and current_node:
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                
                # 特殊處理CLK和Resource
                if param_name.upper() == 'CLK':
                    try:
                        current_node['fpga_constraints']['estimated_clk'] = int(param_value)
                    except:
                        pass
                elif param_name.lower() == 'resource':
                    current_node['fpga_constraints']['resource_usage'] = param_value
                else:
                    # 一般參數
                    try:
                        # 嘗試轉換為數字或列表
                        param_value = json.loads(param_value)
                    except:
                        pass
                    
                    current_node['parameters'][param_name] = {'default': param_value}
        
        # 加入最後一個節點
        if current_node:
            pipeline.append(current_node)
        
        # 從資料庫補全資訊
        for node in pipeline:
            self._enrich_node_from_library(node)
        
        return pipeline
    
    def _enrich_node_from_library(self, node: Dict):
        """
        從tech_lib.json補全節點資訊
        
        Args:
            node: 節點字典（會被修改）
        """
        func_name = node.get('function', '')
        
        # 嘗試查找
        algo_data = None
        
        # 策略1: 直接匹配
        normalized = func_name.lower().replace(' ', '_')
        algo_data = self.lib_manager.get_algorithm(normalized)
        
        # 策略2: CamelCase轉snake_case
        if not algo_data:
            snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', func_name).lower()
            algo_data = self.lib_manager.get_algorithm(snake_case)
        
        # 策略3: 搜尋
        if not algo_data:
            results = self.lib_manager.search_by_name(func_name)
            if results:
                algo_data = results[0]
        
        # 補全資訊
        if algo_data:
            node['name'] = algo_data.get('name', node['name'])
            node['category'] = algo_data.get('category', node['category'])
            node['description'] = algo_data.get('description', '')
            
            # 合併參數（保留匯入的值）
            lib_params = algo_data.get('parameters', {})
            for param_name, param_info in lib_params.items():
                if param_name not in node['parameters']:
                    node['parameters'][param_name] = param_info
            
            # 補全FPGA約束（僅當未設定時）
            if node['fpga_constraints']['estimated_clk'] == 999:
                lib_fpga = algo_data.get('fpga_constraints', {})
                node['fpga_constraints'] = lib_fpga
            
            node['source'] = algo_data.get('_library_type', 'official')


# ==================== 測試與範例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Import Parser Test Suite")
    print("=" * 60)
    
    parser = ImportParser()
    
    # Test 1: DOT格式
    print("\n[Test 1] Parsing DOT format:")
    dot_content = '''
    digraph pipeline {
        node_0 [label="GaussianBlur\\nksize=9\\nsigma=2.0", clk=150]
        node_1 [label="Canny\\nthreshold1=50\\nthreshold2=150", clk=450]
        node_2 [label="HoughCircles\\nparam2=50", clk=8500]
        
        node_0 -> node_1 -> node_2
    }
    '''
    
    result = parser.parse(dot_content)
    if result['success']:
        print(f"  [OK] Parsed {result['node_count']} nodes from DOT")
        for node in result['pipeline']:
            print(f"    - {node['name']}: {node['fpga_constraints']['estimated_clk']} clk")
    else:
        print(f"  [FAIL] {result['error']}")
    
    # Test 2: 文字工單格式
    print("\n[Test 2] Parsing Text Workflow format:")
    txt_content = '''
# Coin Detection Pipeline

1. GaussianBlur
   - ksize: 9
   - sigma: 2.0
   - CLK: 150

2. Canny
   - threshold1: 50
   - threshold2: 150
   - CLK: 450

3. HoughCircles
   - param2: 50
   - minDist: 50
   - CLK: 8500
    '''
    
    result = parser.parse(txt_content)
    if result['success']:
        print(f"  [OK] Parsed {result['node_count']} nodes from TXT")
        for node in result['pipeline']:
            params_str = ', '.join([f"{k}={v['default']}" for k, v in list(node['parameters'].items())[:2]])
            print(f"    - {node['name']}: [{params_str}], {node['fpga_constraints']['estimated_clk']} clk")
    else:
        print(f"  [FAIL] {result['error']}")
    
    # Test 3: 自動偵測
    print("\n[Test 3] Auto-detect format:")
    test_cases = [
        ('{"pipeline": []}', 'json'),
        ('digraph { a -> b }', 'dot'),
        ('1. GaussianBlur\n  - ksize: 5', 'txt')
    ]
    
    for content, expected in test_cases:
        detected = parser.detect_format(content)
        status = "[OK]" if detected == expected else "[FAIL]"
        print(f"  {status} Expected {expected}, got {detected}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
