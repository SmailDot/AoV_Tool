import cv2
import numpy as np
import time
from copy import deepcopy
from typing import List, Dict, Tuple

# Internal Modules
from .strategy import HillClimbing, RandomSearch
from .evaluator import PipelineEvaluator
from .params import ParameterRules

# External Dependencies (Assume running from root)
from processor import ImageProcessor

class AutoTuner:
    """
    目標驅動優化器 (Target-Driven Optimizer) V2
    
    支援演算法: Hill Climbing
    支援指標: IoU + Shape Similarity
    """
    
    def __init__(self):
        self.processor = ImageProcessor()
        self.strategy = HillClimbing() # Default Strategy
        self.evaluator = PipelineEvaluator()
        
    def tune_pipeline(
        self, 
        image: np.ndarray, 
        target_mask: np.ndarray, 
        initial_pipeline: List[Dict], 
        max_iterations: int = 50,
        time_limit: int = 15
    ) -> Tuple[List[Dict], float]:
        """
        自動調參主迴圈
        """
        print(f"[AutoTuner] Starting optimization loop ({max_iterations} iters / {time_limit}s)...")
        
        # 1. 識別可優化參數
        tunable_params = self._extract_tunable_params(initial_pipeline)
        
        if not tunable_params:
            print("[AutoTuner] No tunable parameters found.")
            return initial_pipeline, 0.0
            
        # 2. 初始評估
        best_pipeline = deepcopy(initial_pipeline)
        
        try:
            # 執行並評分
            # 注意: execute_pipeline 需要 return 一個結果圖
            result = self.processor.execute_pipeline(image, best_pipeline, debug_mode=False)
            best_score = self.evaluator.calculate_score(result, target_mask)
        except Exception as e:
            print(f"[AutoTuner] Initial eval failed: {e}")
            best_score = 0.0
            
        print(f"[AutoTuner] Initial Score: {best_score:.4f}")
        
        start_time = time.time()
        current_pipeline = best_pipeline
        last_improved = False
        
        # 3. 優化迴圈
        for i in range(max_iterations):
            if time.time() - start_time > time_limit:
                print("[AutoTuner] Time limit reached.")
                break
                
            # 更新參數
            # 注意：每次重新提取 tunable_params 以確保 current_val 是最新的
            current_tunable = self._extract_tunable_params(current_pipeline)
            candidate_pipeline = self.strategy.generate_candidate(current_pipeline, current_tunable, last_improved)
            
            # 執行與評估
            try:
                result = self.processor.execute_pipeline(image, candidate_pipeline, debug_mode=False)
                score = self.evaluator.calculate_score(result, target_mask)
            except Exception:
                score = 0.0
                
            # 比較
            if score > best_score:
                print(f"  [Iter {i}] New Best! Score: {score:.4f} (was {best_score:.4f})")
                best_score = score
                best_pipeline = candidate_pipeline
                current_pipeline = candidate_pipeline
                last_improved = True
            else:
                last_improved = False
                # Hill Climbing 失敗時通常不回退，而是嘗試改變方向 (由 strategy 控制)
                # 但如果 score 變得極差 (e.g. 0.0)，可能需要回退?
                # 這裡簡單起見，我們接受 "Bad Move" 讓 strategy 下次調整方向
                current_pipeline = candidate_pipeline
            
        print(f"[AutoTuner] Optimization finished. Best Score: {best_score:.4f}")
        return best_pipeline, best_score

    def _extract_tunable_params(self, pipeline: List[Dict]) -> List[Dict]:
        """
        找出 Pipeline 中所有數值型的參數 (Snapshotted)
        """
        tunable = []
        for node_idx, node in enumerate(pipeline):
            params = node.get('parameters', {})
            for param_name, param_info in params.items():
                val = param_info.get('default')
                
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    param_range = ParameterRules.get_range(node['function'], param_name, val)
                    # 只有範圍不為 0 的才調
                    if param_range != (0, 0):
                        tunable.append({
                            "node_idx": node_idx,
                            "param_name": param_name,
                            "current_val": val,
                            "type": type(val),
                            "range": param_range
                        })
        return tunable
