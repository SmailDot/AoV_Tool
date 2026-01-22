
import cv2
import numpy as np
from typing import Dict, List

def prepare_merge_inputs(inputs: List[np.ndarray]) -> List[np.ndarray]:
    """ Helper: Ensure inputs have same size for merge ops """
    if not inputs: return []
    
    base_h, base_w = inputs[0].shape[:2]
    resized_inputs = [inputs[0]]
    
    for i in range(1, len(inputs)):
        img = inputs[i]
        h, w = img.shape[:2]
        if h != base_h or w != base_w:
            img = cv2.resize(img, (base_w, base_h))
        
        # Type matching (Color vs Gray)
        if len(inputs[0].shape) == 3 and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(inputs[0].shape) == 2 and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        resized_inputs.append(img)
        
    return resized_inputs

def op_add_weighted(inputs: List[np.ndarray], params: Dict, debug: bool) -> np.ndarray:
    if len(inputs) < 2: return inputs[0] if inputs else None
    inputs = prepare_merge_inputs(inputs)
    
    alpha = float(params.get('alpha', {}).get('default', 0.5))
    beta = float(params.get('beta', {}).get('default', 0.5))
    gamma = float(params.get('gamma', {}).get('default', 0.0))
    
    return cv2.addWeighted(inputs[0], alpha, inputs[1], beta, gamma)

def op_bitwise_and(inputs: List[np.ndarray], params: Dict, debug: bool) -> np.ndarray:
    if len(inputs) < 2: return inputs[0] if inputs else None
    inputs = prepare_merge_inputs(inputs)
    return cv2.bitwise_and(inputs[0], inputs[1])

def op_add(inputs: List[np.ndarray], params: Dict, debug: bool) -> np.ndarray:
    if len(inputs) < 2: return inputs[0] if inputs else None
    inputs = prepare_merge_inputs(inputs)
    return cv2.add(inputs[0], inputs[1])

def op_bitwise_or(inputs: List[np.ndarray], params: Dict, debug: bool) -> np.ndarray:
    if len(inputs) < 2: return inputs[0] if inputs else None
    inputs = prepare_merge_inputs(inputs)
    return cv2.bitwise_or(inputs[0], inputs[1])

def op_bitwise_xor(inputs: List[np.ndarray], params: Dict, debug: bool) -> np.ndarray:
    if len(inputs) < 2: return inputs[0] if inputs else None
    inputs = prepare_merge_inputs(inputs)
    return cv2.bitwise_xor(inputs[0], inputs[1])

def op_absdiff(inputs: List[np.ndarray], params: Dict, debug: bool) -> np.ndarray:
    if len(inputs) < 2: return inputs[0] if inputs else None
    inputs = prepare_merge_inputs(inputs)
    return cv2.absdiff(inputs[0], inputs[1])
