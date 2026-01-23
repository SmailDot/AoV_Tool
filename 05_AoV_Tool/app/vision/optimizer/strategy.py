import random
import copy
from typing import List, Dict, Any, Tuple
from .params import ParameterRules

class OptimizationStrategy:
    """
    優化策略基類
    """
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict]) -> List[Dict]:
        raise NotImplementedError

class RandomSearch(OptimizationStrategy):
    """
    純隨機搜尋 (Baseline)
    """
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict]) -> List[Dict]:
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
        
    def generate_candidate(self, current_pipeline: List[Dict], tunable_params: List[Dict], last_improved: bool) -> List[Dict]:
        new_pipeline = copy.deepcopy(current_pipeline)
        
        # 策略：每次只動一個參數，輪流動
        # 這裡簡化為隨機選一個，但長期來看會覆蓋所有參數
        target = random.choice(tunable_params)
        key = (target['node_idx'], target['param_name'])
        
        # 初始化記憶
        if key not in self.memory:
            # 初始步伐設為範圍的 10%
            rng = target['range'][1] - target['range'][0]
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
