"""
Processor Module for NKUST AoV Tool
負責人：Preview_Artist (Visual_Feedback_Engineer)

職責：
1. 執行 Pipeline（將 JSON 轉換為實際的 OpenCV 操作）
2. 提供即時預覽功能
3. 錯誤處理與異常保護
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Callable, Optional
import traceback

# Import new Ops modules
from app.vision.ops import basic, edge, morph, filter, transform, feature, detect, merge

class ImageProcessor:
    """
    Pipeline Execution Engine (Refactored)
    Delegates actual processing to 'app.vision.ops' modules.
    """
    
    def __init__(self):
        self.operation_map: Dict[str, Callable] = {
            # --- Basic ---
            "gaussian_blur": basic.op_gaussian_blur,
            "median_blur": basic.op_median_blur,
            "resize": basic.op_resize,
            "bgr2gray": basic.op_bgr2gray,
            "bgr2hsv": basic.op_bgr2hsv,
            
            # --- Edge ---
            "canny_edge": edge.op_canny,
            "sobel": edge.op_sobel,
            "laplacian": edge.op_laplacian,
            
            # --- Filter ---
            "bilateral_filter": filter.op_bilateral_filter,
            "box_filter": filter.op_box_filter,
            "threshold_binary": filter.op_threshold,
            "adaptive_threshold": filter.op_adaptive_threshold,
            "in_range": filter.op_in_range,
            "bitwise_not": filter.op_bitwise_not,
            "equalize_hist": filter.op_equalize_hist,
            "clahe_enhance": filter.op_clahe,
            
            # --- Morphology ---
            "dilate": morph.op_dilate,
            "erode": morph.op_erode,
            "morph_open": morph.op_morph_open,
            "morph_close": morph.op_morph_close,
            "morph_gradient": morph.op_morph_gradient,
            
            # --- Feature ---
            "harris_corner": feature.op_harris_corner,
            "fast_detector": feature.op_fast_detector,
            
            # --- Detection ---
            "advanced_coin_detection": detect.op_advanced_coin_logic,
            "hough_circles": detect.op_hough_circles,
            "find_contours": detect.op_find_contours,
            "cascade_classifier": detect.op_cascade_classifier,
            "hog_descriptor": detect.op_hog_descriptor,
            
            # --- Transform ---
            "flip": transform.op_flip,
            "rotate": transform.op_rotate,
            
            # --- Merge (Graph Ops) ---
            "add": merge.op_add,
            "addWeighted": merge.op_add_weighted,
            "bitwise_and": merge.op_bitwise_and,
            "bitwise_or": merge.op_bitwise_or,
            "bitwise_xor": merge.op_bitwise_xor,
            "absdiff": merge.op_absdiff,
            
            # --- Legacy / Stateful (Kept here for now) ---
            "background_subtractor": self._op_mog2,
            "optical_flow": self._op_optical_flow,
            
            # --- Aliases (Backward Compatibility) ---
            "GaussianBlur": basic.op_gaussian_blur,
            "Canny": edge.op_canny,
            "Dilate": morph.op_dilate,
            "Erode": morph.op_erode,
            "HoughCircles": detect.op_hough_circles,
        }

    def _normalize_func_name(self, func_name: str) -> str:
        """
        Normalize function name to match operation_map keys.
        Handles cases like:
        - "cv2.GaussianBlur" -> "gaussian_blur"
        - "cv2.Canny" -> "canny_edge" (mapped)
        - "GaussianBlur" -> "gaussian_blur"
        """
        if not func_name or func_name == 'Unknown':
            return ''
        
        # Remove cv2. prefix
        if func_name.startswith('cv2.'):
            func_name = func_name[4:]
        
        # Direct match check
        if func_name in self.operation_map:
            return func_name
        
        # Try case-insensitive match
        func_lower = func_name.lower()
        for key in self.operation_map.keys():
            if key.lower() == func_lower:
                return key
        
        # Common name mappings (cv2 name -> our key)
        cv2_mappings = {
            'gaussianblur': 'gaussian_blur',
            'medianblur': 'median_blur',
            'canny': 'canny_edge',
            'threshold': 'threshold_binary',
            'adaptivethreshold': 'adaptive_threshold',
            'bilateralfilter': 'bilateral_filter',
            'boxfilter': 'box_filter',
            'inrange': 'in_range',
            'bitwisenot': 'bitwise_not',
            'equalizehist': 'equalize_hist',
            'dilate': 'dilate',
            'erode': 'erode',
            'morphologyex_open': 'morph_open',
            'morphologyex_close': 'morph_close',
            'morphologyex_gradient': 'morph_gradient',
            'houghcircles': 'hough_circles',
            'findcontours': 'find_contours',
            'flip': 'flip',
            'rotate': 'rotate',
            'resize': 'resize',
            'cvtcolor_bgr2gray': 'bgr2gray',
            'cvtcolor_bgr2hsv': 'bgr2hsv',
            'addweighted': 'addWeighted',
            'add': 'add',
            'bitwise_and': 'bitwise_and',
            'bitwise_or': 'bitwise_or',
            'bitwise_xor': 'bitwise_xor',
            'absdiff': 'absdiff',
        }
        
        if func_lower in cv2_mappings:
            mapped = cv2_mappings[func_lower]
            if mapped in self.operation_map:
                return mapped
        
        # Try removing underscores and compare
        func_compact = func_lower.replace('_', '')
        if func_compact in cv2_mappings:
            mapped = cv2_mappings[func_compact]
            if mapped in self.operation_map:
                return mapped
        
        return func_name

    def execute_pipeline(
        self, 
        image_bgr: np.ndarray, 
        pipeline_json: List[Dict[str, Any]],
        debug_mode: bool = False,
        context: Optional[Dict[str, Any]] = None # [NEW] State Context
    ) -> np.ndarray:
        """
        Execute Graph Pipeline
        """
        if image_bgr is None or image_bgr.size == 0:
            print("[Error] Invalid input image")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Init context if not provided (Stateless mode)
        if context is None:
            context = {}
        
        results_cache = {"source": image_bgr.copy()}
        last_valid_output = image_bgr.copy()  # Track last enabled node's output
        last_valid_node_id = "source"  # Track last enabled node's id
        
        print(f"\n{'=' * 60}")
        print(f"Executing Graph Pipeline ({len(pipeline_json)} nodes)")
        print(f"{'=' * 60}")
        
        for idx, node in enumerate(pipeline_json):
            node_id = node.get('id', f'node_{idx}')
            is_enabled = node.get('_enabled', True)
            
            # [Fix] Handle disabled/skipped nodes - pass through last valid output
            if not is_enabled:
                print(f"\n[{node_id}] SKIPPED (disabled)")
                # Disabled node passes through the last valid output
                results_cache[node_id] = last_valid_output
                continue
            
            try:
                raw_func_name = node.get('function', node.get('name', 'Unknown'))
                func_name = self._normalize_func_name(raw_func_name)
                params = node.get('parameters', {})
                input_ids = node.get('inputs', [])
                
                print(f"\n[{node_id}] {raw_func_name} -> {func_name} (Inputs: {input_ids})")
                
                # Resolve Inputs - use last valid output if no explicit inputs
                input_images = []
                if not input_ids:
                    if idx == 0:
                        input_images = [results_cache["source"]]
                    else:
                        # Use last valid node's output instead of just previous node
                        input_images = [last_valid_output]
                else:
                    for inp_id in input_ids:
                        if inp_id in results_cache:
                            input_images.append(results_cache[inp_id])
                        else:
                            print(f"  [WARN] Input '{inp_id}' not found. Using last valid output.")
                            input_images.append(last_valid_output)
                
                # Dispatch
                if func_name in self.operation_map:
                    op_func = self.operation_map[func_name]
                    
                    # Inspect function signature to see if it accepts context
                    import inspect
                    sig = inspect.signature(op_func)
                    accepts_context = 'context' in sig.parameters
                    
                    # Merge vs Single Dispatch
                    if func_name in ["add", "addWeighted", "bitwise_and", "bitwise_or", "bitwise_xor", "absdiff"]:
                        output_image = op_func(input_images, params, debug_mode)
                    else:
                        src_img = input_images[0] if input_images else results_cache["source"]
                        
                        if accepts_context:
                            output_image = op_func(src_img, params, debug_mode, context=context)
                        else:
                            output_image = op_func(src_img, params, debug_mode)
                    
                    if output_image is None:
                        print(f"  [ERROR] Node {node_id} returned None.")
                        output_image = np.zeros_like(results_cache["source"])
                    
                    results_cache[node_id] = output_image
                    last_valid_output = output_image  # Update last valid output
                    last_valid_node_id = node_id
                    print(f"  [OK] Output shape: {output_image.shape}")
                    
                else:
                    # [Fix] Unknown operation - raise error instead of silent pass-through
                    error_msg = f"Unknown operation '{func_name}'. Not found in operation_map. Available operations: {list(self.operation_map.keys())[:10]}..."
                    print(f"  [ERROR] {error_msg}")
                    raise ValueError(error_msg)
                
            except Exception as e:
                print(f"  [ERROR] Node {node_id} failed: {e}")
                # On error, pass through last valid output so pipeline can continue
                results_cache[node_id] = last_valid_output
                if debug_mode: traceback.print_exc()
                results_cache[node_id] = results_cache["source"]
        
        print(f"\n{'=' * 60}\n")
        return last_output_image

    # ==================== Stateful Ops (Keep Here) ====================
    
    def _ensure_gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
        
    def _ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _op_mog2(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        if not hasattr(self, '_mog2_subtractor'):
            history = int(params.get('history', {}).get('default', 500))
            varThreshold = float(params.get('varThreshold', {}).get('default', 16))
            detectShadows = bool(params.get('detectShadows', {}).get('default', True))
            self._mog2_subtractor = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
        
        mask = self._mog2_subtractor.apply(img)
        return self._ensure_bgr(mask)

    def _op_optical_flow(self, img: np.ndarray, params: Dict, debug: bool) -> np.ndarray:
        gray = self._ensure_gray(img)
        if not hasattr(self, '_prev_gray_flow'):
            self._prev_gray_flow = gray
            return img
        
        # [Fix] Explicit flow handling for type safety
        flow = cv2.calcOpticalFlowFarneback(self._prev_gray_flow, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self._prev_gray_flow = gray
        
        h, w = gray.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # [Fix] Normalize properly
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def create_thumbnail(self, image: np.ndarray, max_width: int = 640) -> np.ndarray:
        return basic.op_resize(image, {'width': {'default': max_width}, 'height': {'default': int(image.shape[0]*max_width/image.shape[1])}}, False)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        pipeline_json: List[Dict[str, Any]],
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a video file frame by frame using the defined pipeline.
        
        Args:
            input_path: Path to source video
            output_path: Path to save processed video
            pipeline_json: Algorithm pipeline
            max_frames: Optional limit for testing
            
        Returns:
            Dict containing processing stats
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        # Get Video Properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0: fps = 30.0
        
        # [Fix] Logic Bug: Initialize writer based on PROCESSED frame size, not original size
        # This allows the pipeline to resize the video (e.g. 1080p -> 480p) correctly.
        
        out = None
        context = {} # State container for the entire video session
        frame_count = 0
        
        print(f"\n[Video] Input: {width}x{height} @ {fps}fps")
        
        # Init output dimensions to avoid unbound local error
        out_w, out_h = width, height 
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                    
                # Execute Pipeline for this frame
                processed_frame = self.execute_pipeline(frame, pipeline_json, debug_mode=False, context=context)
                
                # Initialize VideoWriter on the first frame
                if out is None:
                    out_h, out_w = processed_frame.shape[:2]
                    print(f"[Video] Output Resolution initialized to: {out_w}x{out_h}")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
                
                # Double check dimension consistency (in case dynamic resize changes mid-stream)
                # If frame size changes mid-stream, we must force it to match the writer
                if processed_frame.shape[1] != out_w or processed_frame.shape[0] != out_h:
                    processed_frame = cv2.resize(processed_frame, (out_w, out_h))
                
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"  Processed {frame_count}/{total_frames} frames...")
                    
        finally:
            cap.release()
            if out is not None:
                out.release()
            print(f"[Video] Finished. Saved to {output_path}")
            
        return {
            "total_frames": frame_count,
            "fps": fps,
            "resolution": f"{out_w}x{out_h}" if 'out_w' in locals() else "0x0"
        }

    print("Processor loaded.")
