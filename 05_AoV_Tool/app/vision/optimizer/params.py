from typing import Dict, Tuple, Any

class ParameterRules:
    """
    定義各演算法參數的可調整範圍
    """
    
    @staticmethod
    def get_range(func_name: str, param_name: str, current_val: Any) -> Tuple[float, float]:
        """
        獲取參數範圍 (min, max)
        """
        func_lower = func_name.lower()
        param_lower = param_name.lower()
        
        # --- Canny Edge ---
        if 'canny' in func_lower:
            if 'threshold' in param_lower: return (10, 250)
            
        # --- Hough Circles ---
        if 'hough' in func_lower:
            if 'dp' in param_lower: return (1.0, 2.0)
            if 'dist' in param_lower: return (10, 100)
            if 'param1' in param_lower: return (20, 150) # Canny high
            if 'param2' in param_lower: return (10, 60)  # Accumulator
            if 'radius' in param_lower: return (0, 200)
            
        # --- Blur (Gaussian, Median) ---
        if 'blur' in func_lower:
            if 'ksize' in param_lower: return (1, 31) # 需為奇數
            if 'sigma' in param_lower: return (0.1, 10.0)
            
        # --- Threshold ---
        if 'thresh' in func_lower:
            if 'thresh' in param_lower: return (0, 255)
            if 'maxval' in param_lower: return (255, 255) # 通常不調
            
        # --- Bilateral Filter ---
        if 'bilateral' in func_lower:
            if 'd' in param_lower: return (5, 15)
            if 'sigma' in param_lower: return (10, 150)
            
        # --- Morphology ---
        if 'morph' in func_lower or 'dilate' in func_lower or 'erode' in func_lower:
            if 'ksize' in param_lower: return (3, 11)
            if 'iter' in param_lower: return (1, 5)

        # Default: Heuristic Range based on current value
        try:
            val = float(current_val)
            if val == 0: return (0, 10)
            return (val * 0.5, val * 1.5)
        except:
            return (0, 0) # Non-numeric, don't tune

    @staticmethod
    def is_odd_required(param_name: str) -> bool:
        """某些參數 (如 kernel size) 必須是奇數"""
        return 'ksize' in param_name.lower()
