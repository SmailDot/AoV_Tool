import cv2
import numpy as np
import time
import random
from copy import deepcopy
from typing import List, Dict, Tuple

# Internal Modules
from .strategy import HillClimbing, RandomSearch, GeneticAlgorithm
from .evaluator import PipelineEvaluator
from .params import ParameterRules

# External Dependencies (Assume running from root)
from app.core.processor import ImageProcessor
from app.core.library_manager import LibraryManager

class AutoTuner:
    """
    目標驅動優化器 (Target-Driven Optimizer) V2
    
    支援演算法: Hill Climbing, Genetic Algorithm (w/ Structure Mutation)
    支援指標: IoU + Shape Similarity
    """
    
    def __init__(self, method='ga'):
        self.method = method # Store method
        self.processor = ImageProcessor()
        self.lib_manager = LibraryManager() # Load library for structural mutation
        
        if method == 'ga':
            # Pass lib_manager to support structural mutation (add/remove nodes)
            self.strategy = GeneticAlgorithm(lib_manager=self.lib_manager)
        elif method == 'llm':
            self.strategy = None # Strategy handled by LLM direct calls
        else:
            self.strategy = HillClimbing()
            
        self.evaluator = PipelineEvaluator()

    def _tune_with_llm(self, image, target_mask, initial_pipeline, max_iters, time_limit, target_score):
        """
        Implementation of LLM-in-the-Loop Optimization
        """
        import streamlit as st
        from app.core.logic_engine import LogicEngine # Import here to avoid circular dependency
        
        print("[AutoTuner] Starting LLM Vision Feedback Loop...")
        best_pipeline = deepcopy(initial_pipeline)
        
        # Initial Eval
        try:
            result = self.processor.execute_pipeline(image, best_pipeline, debug_mode=False)
            best_score = self.evaluator.calculate_score(result, target_mask)
        except:
            best_score = 0.0
            
        status_container = st.empty()
        
        # We use a mocked logic if no key, but prompt engineering will handle the "vision" part 
        # by describing the image stats (histogram, edges) since we can't upload images to GPT-4o without key.
        # IF user has key, we use GPT-4o-mini with image.
        
        engine = st.session_state.get('engine')
        if not engine:
            engine = LogicEngine(self.lib_manager)
            
        # Optimization Loop
        for i in range(min(max_iters, 10)): # LLM calls are slow/expensive, limit to 10 rounds
            status_container.markdown(f"**LLM Thinking... Round {i+1}** (Current Best: {best_score:.4f})")
            
            # 1. Analyze current result
            # Since we might not have a real VLM key, we extract "textual features" from the image
            # to describe it to the LLM.
            stats = self._analyze_image_stats(result, target_mask)
            
            # 2. Ask LLM for suggestions
            prompt = f"""
            You are a Computer Vision Expert Optimizing an OpenCV Pipeline.
            
            Current Status:
            - IoU Score: {best_score:.4f} (Goal: {target_score})
            - Pipeline: {[n['name'] for n in best_pipeline]}
            
            Image Analysis:
            - Prediction Fill Ratio: {stats['pred_fill']:.2f}
            - Target Mask Fill Ratio: {stats['target_fill']:.2f}
            - Noise Level: {stats['noise_level']}
            - Edges Detected: {stats['edge_count']}
            
            Problem:
            The current result does not match the target mask. 
            If Pred Fill << Target Fill: We have edges but need a solid mask. Suggest 'find_contours' or 'dilate' or 'morph_close'.
            If Noise Level is High: Suggest 'gaussian_blur' or 'median_blur'.
            If Edges are weak: Lower 'canny_threshold'.
            
            RESPONSE FORMAT (JSON ONLY):
            {{
                "action": "modify_param" | "add_node" | "remove_node",
                "target_node_index": 0,
                "param_name": "threshold1", 
                "new_value": 50,
                "node_name": "dilate" (if adding)
            }}
            """
            
            # Call LLM (Mock or Real)
            # Use LogicEngine if API Key is available, otherwise use simulation
            if engine and engine.prompt_master.llm_available:
                suggestion = engine.prompt_master.get_optimization_suggestion(stats, best_pipeline, best_score, target_score)
            else:
                suggestion = self._simulate_llm_expert(stats, best_pipeline)
            
            # 3. Apply Suggestion
            new_pipeline = deepcopy(best_pipeline)
            self._apply_suggestion(new_pipeline, suggestion)
            
            # 4. Evaluate
            try:
                new_result = self.processor.execute_pipeline(image, new_pipeline, debug_mode=False)
                new_score = self.evaluator.calculate_score(new_result, target_mask)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pipeline = new_pipeline
                    result = new_result
                    st.toast(f"LLM Improved Score! ({new_score:.4f})", icon="🧠")
                else:
                    # Revert if worse
                    pass
            except Exception as e:
                print(f"LLM Suggestion Failed: {e}")
                
            if best_score >= target_score:
                break
                
        return best_pipeline, best_score

    def _analyze_image_stats(self, image, mask):
        # Helper to extract features for "Text-based VLM"
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape)==3 else mask
        
        return {
            "pred_fill": np.count_nonzero(img_gray) / img_gray.size,
            "target_fill": np.count_nonzero(mask_gray) / mask_gray.size,
            "noise_level": "High" if cv2.Laplacian(img_gray, cv2.CV_64F).var() > 500 else "Low",
            "edge_count": np.count_nonzero(cv2.Canny(img_gray, 100, 200))
        }

    def _simulate_llm_expert(self, stats, pipeline):
        # A "Rule-Based Expert" simulating what an LLM would say
        # This implements the "Diagnosis Logic" we discussed.
        
        pred_fill = stats['pred_fill']
        target_fill = stats['target_fill']
        
        # Case 1: Empty Result
        if pred_fill < 0.01:
             # Need to lower thresholds or add source
             # Find Canny
             for idx, node in enumerate(pipeline):
                 if 'canny' in node['name'].lower():
                     return {"action": "modify_param", "target_node_index": idx, "param_name": "threshold1", "new_value": 10}
             return {"action": "add_node", "node_name": "adaptive_threshold", "target_node_index": 0}
             
        # Case 2: Hollow vs Solid (The Coin Problem)
        if pred_fill < target_fill * 0.5:
             # We have edges, need solid
             # Check if we have dilation
             for idx, node in enumerate(pipeline):
                 if 'dilate' in node['name'].lower():
                     return {"action": "modify_param", "target_node_index": idx, "param_name": "iterations", "new_value": 5}
             
             # Add Morph Close or Fill Holes
             return {"action": "add_node", "node_name": "morph_close", "target_node_index": len(pipeline)}
             
        return {"action": "none"}

    def _apply_suggestion(self, pipeline, suggestion):
        action = suggestion.get('action')
        if action == 'modify_param':
            idx = suggestion.get('target_node_index')
            if idx is not None and idx < len(pipeline):
                param_name = suggestion.get('param_name')
                new_val = suggestion.get('new_value')
                # Ensure parameter exists
                if 'parameters' in pipeline[idx] and param_name in pipeline[idx]['parameters']:
                    pipeline[idx]['parameters'][param_name]['default'] = new_val
                    
        elif action == 'add_node':
            node_name = suggestion.get('node_name')
            idx = suggestion.get('target_node_index', len(pipeline))
            
            # Use LibraryManager to get node data
            if self.lib_manager:
                algo_data = self.lib_manager.get_algorithm(node_name, 'official')
                # Try fallback search
                if not algo_data:
                     algo_data = self.lib_manager.get_algorithm(node_name, 'contributed')
                
                if algo_data:
                    new_node = {
                        'id': f"auto_{random.randint(1000,9999)}",
                        'name': algo_data['name'],
                        'function': algo_data.get('opencv_function', node_name),
                        'category': algo_data['category'],
                        'parameters': deepcopy(algo_data.get('parameters', {})),
                        'fpga_constraints': deepcopy(algo_data.get('fpga_constraints', {})),
                        '_enabled': True
                    }
                    if idx > len(pipeline): idx = len(pipeline)
                    pipeline.insert(idx, new_node)
        
        elif action == 'remove_node':
             idx = suggestion.get('target_node_index')
             if idx is not None and idx < len(pipeline):
                 pipeline.pop(idx)
        
    def tune_pipeline(
        self, 
        image: np.ndarray, 
        target_mask: np.ndarray, 
        initial_pipeline: List[Dict], 
        max_iterations: int = 50,
        time_limit: int = 15,
        target_score: float = 0.95
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

        # [NEW] Check for LLM Vision Feedback mode
        if self.method == 'llm':
             print("[AutoTuner] Using LLM Vision Feedback Strategy")
             # Use a specialized loop for LLM feedback
             return self._tune_with_llm(image, target_mask, best_pipeline, max_iterations, time_limit, target_score)
            
        # [Fix] Pre-process Pipeline to remove Fixed Resize distortion
        # ... (rest of the file)

        h_src, w_src = image.shape[:2]
        ar_src = w_src / (h_src + 1e-6)
        
        # Make a deep copy first so we don't mutate input in place unexpectedly
        best_pipeline = deepcopy(initial_pipeline)
        
        # Import streamlit for notifications
        import streamlit as st
        
        for node in best_pipeline:
            # Check for Resize nodes
            if 'resize' in node.get('name', '').lower() or 'resize' in node.get('function', '').lower():
                params = node.get('parameters', {})
                w_node = params.get('width', {}).get('default', 0)
                h_node = params.get('height', {}).get('default', 0)
                
                if w_node > 0 and h_node > 0:
                     ar_node = w_node / (h_node + 1e-6)
                     # If mismatch > 5%, force fix (e.g., set height to -1 for auto)
                     if abs(ar_node - ar_src) / ar_src > 0.05:
                         print(f"[AutoTuner] Fixing distorted Resize node: {w_node}x{h_node}")
                         # Set height to 0 (which triggers smart resize logic in basic.py to use AR)
                         node['parameters']['height']['default'] = 0
                         # Update display
                         st.toast(f"已自動修正 Resize 節點以符合原圖比例 ({w_src}x{h_src})", icon="🔧")

        # 2. 初始評估
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
        
        # [Visual Feedback]
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        stop_reason = "Max iterations reached"
        
        # 3. 優化迴圈
        for i in range(max_iterations):
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                print("[AutoTuner] Time limit reached.")
                stop_reason = "Time limit reached"
                break
                
            progress_bar.progress((i + 1) / max_iterations)
            
        if self.method == 'llm':
            # This path is actually unreachable due to early return above, 
            # but kept for safety/logic flow integrity if logic changes.
            pass
        else:
            # 更新參數
            current_tunable = self._extract_tunable_params(current_pipeline)
            candidate_pipeline = self.strategy.generate_candidate(current_pipeline, current_tunable, last_improved)
            
            # [Fix] Resize Protection
            # Ensure candidate pipeline doesn't introduce distortion via Resize nodes
            # If width/height are set to fixed values that distort aspect ratio, reset them.
            h_src, w_src = image.shape[:2]
            ar_src = w_src / (h_src + 1e-6)
            
            for node in candidate_pipeline:
                 if 'resize' in node.get('name', '').lower():
                     p = node.get('parameters', {})
                     w = p.get('width', {}).get('default', 0)
                     h = p.get('height', {}).get('default', 0)
                     if w > 0 and h > 0:
                         ar_node = w / (h + 1e-6)
                         if abs(ar_node - ar_src) / ar_src > 0.05:
                             # Distortion detected in candidate, force fix
                             node['parameters']['height']['default'] = 0
            
            # 執行與評估
            try:
                # Debug: Print what we are running
                # print(f"  [Iter {i}] Running pipeline...")
                result = self.processor.execute_pipeline(image, candidate_pipeline, debug_mode=False)
                score = self.evaluator.calculate_score(result, target_mask)
            except Exception as e:
                # print(f"  [Iter {i}] Failed: {e}")
                score = 0.0
                    
                # 更新狀態顯示
                status_container.markdown(f"""
                **Optimization Status**: Iteration {i+1}/{max_iterations}
                - Current Best IoU: `{best_score:.4f}`
                - Last Attempt IoU: `{score:.4f}`
                - Time Elapsed: `{elapsed:.1f}s / {time_limit}s`
                """)

                # 比較
                if score > best_score:
                    print(f"  [Iter {i}] New Best! Score: {score:.4f} (was {best_score:.4f})")
                    best_score = score
                    best_pipeline = candidate_pipeline
                    current_pipeline = candidate_pipeline
                    last_improved = True
                    
                    # Early Exit if Good Enough
                    if best_score >= target_score:
                        status_container.success(f"Reached Target Accuracy! ({best_score:.4f})")
                        stop_reason = "Target accuracy reached"
                        return best_pipeline, best_score
                else:
                    last_improved = False
                    # GA/Hill Climbing Logic
                    current_pipeline = best_pipeline
            
        print(f"[AutoTuner] Optimization finished. Reason: {stop_reason}. Best Score: {best_score:.4f}")
        st.toast(f"優化結束: {stop_reason}", icon="🏁")
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
