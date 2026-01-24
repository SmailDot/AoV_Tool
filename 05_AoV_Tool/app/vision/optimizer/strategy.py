import random
import copy
from typing import List, Dict, Any, Tuple
from .params import ParameterRules

class OptimizationStrategy:
    """
    優化策略基類
    """
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict], last_improved: bool = False) -> List[Dict]:
        raise NotImplementedError

class RandomSearch(OptimizationStrategy):
    """
    純隨機搜尋 (Baseline)
    """
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict], last_improved: bool = False) -> List[Dict]:
        new_pipeline = copy.deepcopy(current_pipeline)
        
        # Randomly select a parameter to mutate
        target = random.choice(tunable_params)
        node = new_pipeline[target['node_idx']]
        
        min_val, max_val = target['range']
        
        if target['type'] == int:
            new_val = random.randint(int(min_val), int(max_val))
            if ParameterRules.is_odd_required(target['param_name']) and new_val % 2 == 0:
                new_val += 1
        else:
            new_val = random.uniform(min_val, max_val)
            
        node['parameters'][target['param_name']]['default'] = new_val
        return new_pipeline

class HillClimbing(OptimizationStrategy):
    """
    爬山演算法 (Hill Climbing)
    針對每個參數維護一個 'step' (步伐) 和 'direction' (方向)
    """
    def __init__(self):
        # 記憶體: 紀錄每個參數的上次變動狀態
        # Key: (node_idx, param_name), Value: {'step': float, 'last_change': float}
        self.memory = {}
        
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict], last_improved: bool = False) -> List[Dict]:
        new_pipeline = copy.deepcopy(current_pipeline)
        
        # 策略：每次只動一個參數，輪流動
        # 這裡簡化為隨機選一個，但長期來看會覆蓋所有參數
        if not tunable_params: return new_pipeline
        
        target = random.choice(tunable_params)
        key = (target['node_idx'], target['param_name'])
        
        # 初始化記憶
        if key not in self.memory:
            # 初始步伐設為範圍的 10%
            rng = target['range'][1] - target['range'][0]
            if rng == 0: rng = 1.0 # Prevent division by zero
            self.memory[key] = {'step': rng * 0.1, 'direction': 1}
            
        state = self.memory[key]
        
        # 如果上次變動導致進步 (last_improved=True)，保持方向，稍微加大步伐
        # 如果沒進步 (last_improved=False)，反轉方向，並縮小步伐
        if not last_improved:
            state['direction'] *= -1
            state['step'] *= 0.5
        else:
            state['step'] *= 1.1 # Momentum
            
        # 計算新值
        change = state['step'] * state['direction']
        current_val = target['current_val']
        new_val = current_val + change
        
        # 邊界檢查 (Clip)
        min_val, max_val = target['range']
        new_val = max(min_val, min(new_val, max_val))
        
        # 型態轉換與約束
        if target['type'] == int:
            new_val = int(round(new_val))
            if ParameterRules.is_odd_required(target['param_name']) and new_val % 2 == 0:
                new_val += 1
                
        # 寫入 Pipeline
        node = new_pipeline[target['node_idx']]
        node['parameters'][target['param_name']]['default'] = new_val
        
        return new_pipeline

class GeneticAlgorithm(OptimizationStrategy):
    """
    基因演算法 (Genetic Algorithm) - 支援結構突變 (Structural Mutation)
    模擬生物演化，支援：
    1. 參數突變 (Parameter Mutation)
    2. 結構突變 (Add/Remove/Swap Node) [NEW]
    """
    def __init__(self, mutation_rate=0.3, structure_mutation_rate=0.2, lib_manager=None):
        self.mutation_rate = mutation_rate
        self.structure_mutation_rate = structure_mutation_rate
        self.lib_manager = lib_manager
        
        # Valid nodes for insertion
        self.valid_ops = [
            'gaussian_blur', 'median_blur', 'bilateral_filter', 
            'dilate', 'erode', 'morph_open', 'morph_close',
            'threshold_binary', 'adaptive_threshold',
            'canny_edge', 'sobel', 'laplacian',
            'bitwise_not', 'equalize_hist'
        ]
        
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict], last_improved: bool = False) -> List[Dict]:
        """
        (1+1)-ES with Structural Mutation
        """
        new_pipeline = copy.deepcopy(current_pipeline)
        
        # Randomly select a parameter to mutate
        target = random.choice(tunable_params)
        node = new_pipeline[target['node_idx']]
        
        min_val, max_val = target['range']
        
        if target['type'] == int:
            new_val = random.randint(int(min_val), int(max_val))
            if ParameterRules.is_odd_required(target['param_name']) and new_val % 2 == 0:
                new_val += 1
        else:
            new_val = random.uniform(min_val, max_val)
            
        node['parameters'][target['param_name']]['default'] = new_val
        return new_pipeline

class HillClimbing(OptimizationStrategy):
    """
    爬山演算法 (Hill Climbing)
    針對每個參數維護一個 'step' (步伐) 和 'direction' (方向)
    """
    def __init__(self):
        # 記憶體: 紀錄每個參數的上次變動狀態
        # Key: (node_idx, param_name), Value: {'step': float, 'last_change': float}
        self.memory = {}
        
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict], last_improved: bool = False) -> List[Dict]:
        new_pipeline = copy.deepcopy(current_pipeline)
        
        # 策略：每次只動一個參數，輪流動
        # 這裡簡化為隨機選一個，但長期來看會覆蓋所有參數
        if not tunable_params: return new_pipeline
        
        target = random.choice(tunable_params)
        key = (target['node_idx'], target['param_name'])
        
        # 初始化記憶
        if key not in self.memory:
            # 初始步伐設為範圍的 10%
            rng = target['range'][1] - target['range'][0]
            if rng == 0: rng = 1.0 # Prevent division by zero
            self.memory[key] = {'step': rng * 0.1, 'direction': 1}
            
        state = self.memory[key]
        
        # 如果上次變動導致進步 (last_improved=True)，保持方向，稍微加大步伐
        # 如果沒進步 (last_improved=False)，反轉方向，並縮小步伐
        if not last_improved:
            state['direction'] *= -1
            state['step'] *= 0.5
        else:
            state['step'] *= 1.1 # Momentum
            
        # 計算新值
        change = state['step'] * state['direction']
        current_val = target['current_val']
        new_val = current_val + change
        
        # 邊界檢查 (Clip)
        min_val, max_val = target['range']
        new_val = max(min_val, min(new_val, max_val))
        
        # 型態轉換與約束
        if target['type'] == int:
            new_val = int(round(new_val))
            if ParameterRules.is_odd_required(target['param_name']) and new_val % 2 == 0:
                new_val += 1
                
        # 寫入 Pipeline
        node = new_pipeline[target['node_idx']]
        node['parameters'][target['param_name']]['default'] = new_val
        
        return new_pipeline

class GeneticAlgorithm(OptimizationStrategy):
    """
    基因演算法 (Genetic Algorithm) - 支援結構突變 (Structural Mutation)
    模擬生物演化，支援：
    1. 參數突變 (Parameter Mutation)
    2. 結構突變 (Add/Remove/Swap Node) [NEW]
    """
    def __init__(self, mutation_rate=0.3, structure_mutation_rate=0.2, lib_manager=None):
        self.mutation_rate = mutation_rate
        self.structure_mutation_rate = structure_mutation_rate
        self.lib_manager = lib_manager
        
        # Valid nodes for insertion
        self.valid_ops = [
            'gaussian_blur', 'median_blur', 'bilateral_filter', 
            'dilate', 'erode', 'morph_open', 'morph_close',
            'threshold_binary', 'adaptive_threshold',
            'canny_edge', 'sobel', 'laplacian',
            'bitwise_not', 'equalize_hist'
        ]
        
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict], last_improved: bool = False) -> List[Dict]:
        """
        (1+1)-ES with Structural Mutation
        """
        new_pipeline = copy.deepcopy(current_pipeline)
        
        # --- 1. Structural Mutation (Low Probability) ---
        if self.lib_manager and random.random() < self.structure_mutation_rate and not last_improved:
            action = random.choice(['add', 'remove', 'swap'])
            
            if action == 'add' and len(new_pipeline) < 10:
                # Insert a random node
                algo_name = random.choice(self.valid_ops)
                algo_data = self.lib_manager.get_algorithm(algo_name, 'official')
                
                if algo_data:
                    new_node = {
                        'id': f"auto_{random.randint(1000,9999)}",
                        'name': algo_data['name'],
                        'function': algo_data.get('opencv_function', algo_name),
                        'category': algo_data['category'],
                        'parameters': copy.deepcopy(algo_data.get('parameters', {})),
                        'fpga_constraints': copy.deepcopy(algo_data.get('fpga_constraints', {})),
                        '_enabled': True
                    }
                    insert_pos = random.randint(0, len(new_pipeline))
                    new_pipeline.insert(insert_pos, new_node)
                    # print(f"[GA] Structural Mutation: Added {algo_name} at {insert_pos}")
                    return new_pipeline # Return immediately after structural change
                    
            elif action == 'remove' and len(new_pipeline) > 1:
                # Remove a random node
                idx = random.randint(0, len(new_pipeline)-1)
                # print(f"[GA] Structural Mutation: Removed node at {idx}")
                new_pipeline.pop(idx)
                return new_pipeline
                
            elif action == 'swap' and len(new_pipeline) > 1:
                # Swap two adjacent nodes
                idx = random.randint(0, len(new_pipeline)-2)
                new_pipeline[idx], new_pipeline[idx+1] = new_pipeline[idx+1], new_pipeline[idx]
                # print(f"[GA] Structural Mutation: Swapped {idx} <-> {idx+1}")
                return new_pipeline

        # --- 2. Parameter Mutation (If no structural change) ---
        if not tunable_params: return new_pipeline
        
        # 決定突變強度：如果最近沒進步，加大突變力度 (Exploration)；如果有進步，微調 (Exploitation)
        mutation_strength = 0.5 if not last_improved else 0.1
        
        # 針對每個可調參數，都有機率發生突變
        mutated_count = 0
        
        for target in tunable_params:
            # Check if index is still valid (in case structure changed but tunable_params is stale)
            # Actually AutoTuner re-extracts tunable_params every iter, so it's fine.
            if target['node_idx'] >= len(new_pipeline): continue
            
            if random.random() < self.mutation_rate:
                mutated_count += 1
                
                min_val, max_val = target['range']
                current_val = target['current_val']
                val_range = max_val - min_val
                if val_range == 0: continue
                
                # Gaussian Mutation
                change = random.gauss(0, val_range * mutation_strength) 
                new_val = current_val + change
                
                # Clip
                new_val = max(min_val, min(new_val, max_val))
                
                # Type Constraint
                if target['type'] == int:
                    new_val = int(round(new_val))
                    if ParameterRules.is_odd_required(target['param_name']) and new_val % 2 == 0:
                        # Ensure odd
                        if new_val >= max_val: new_val -= 1
                        else: new_val += 1
                
                # Write back
                node = new_pipeline[target['node_idx']]
                node['parameters'][target['param_name']]['default'] = new_val
        
        # 如果隨機沒選到任何參數突變，強制選一個突變
        if mutated_count == 0:
             # Filter valid targets
             valid_targets = [t for t in tunable_params if t['node_idx'] < len(new_pipeline)]
             if not valid_targets: return new_pipeline
             
             target = random.choice(valid_targets)
             min_val, max_val = target['range']
             new_val = random.uniform(min_val, max_val)
             if target['type'] == int:
                 new_val = int(new_val)
                 if ParameterRules.is_odd_required(target['param_name']) and new_val % 2 == 0: new_val += 1
             
             node = new_pipeline[target['node_idx']]
             node['parameters'][target['param_name']]['default'] = new_val
             
        return new_pipeline
