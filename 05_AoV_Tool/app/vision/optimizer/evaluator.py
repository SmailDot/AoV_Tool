import cv2
import numpy as np
from typing import Dict, Optional

class PipelineEvaluator:
    """
    負責評估 Pipeline 執行結果的品質
    """
    
    @staticmethod
    def calculate_score(prediction: np.ndarray, target_mask: np.ndarray) -> float:
        """
        綜合評分 (0.0 ~ 1.0)
        
        Score = w1 * IoU + w2 * ShapeSimilarity
        """
        # 1. 預處理：確保格式一致
        pred_bin = PipelineEvaluator._to_binary(prediction)
        target_bin = PipelineEvaluator._to_binary(target_mask)
        
        # 尺寸對齊
        if pred_bin.shape != target_bin.shape:
            pred_bin = cv2.resize(pred_bin, (target_bin.shape[1], target_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # 2. 計算 IoU (重疊率) - 權重 0.7
        iou = PipelineEvaluator._calculate_iou(pred_bin, target_bin)
        
        # 3. 計算形狀相似度 (Hu Moments) - 權重 0.3
        # 只有當 IoU > 0 (有重疊) 時才計算形狀，避免對全黑圖算形狀
        shape_score = 0.0
        if iou > 0.05:
            shape_score = PipelineEvaluator._calculate_shape_similarity(pred_bin, target_bin)
            
        final_score = (iou * 0.7) + (shape_score * 0.3)
        return float(final_score)

    @staticmethod
    def _to_binary(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def _calculate_iou(pred: np.ndarray, target: np.ndarray) -> float:
        intersection = np.logical_and(pred, target)
        union = np.logical_or(pred, target)
        iou = np.sum(intersection) / (np.sum(union) + 1e-6)
        return float(iou)

    @staticmethod
    def _calculate_shape_similarity(pred: np.ndarray, target: np.ndarray) -> float:
        """
        使用 Hu Moments 計算形狀相似度。
        cv2.matchShapes 回傳值越小越相似 (0=完全一樣)。
        我們將其轉換為 0~1 的分數 (1=完全一樣)。
        """
        try:
            # 尋找輪廓
            # 注意: OpenCV 版本差異，findContours 回傳值數量可能不同
            contours_pred, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_target, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours_pred or not contours_target:
                return 0.0
                
            # 取最大輪廓代表物體
            c_pred = max(contours_pred, key=cv2.contourArea)
            c_target = max(contours_target, key=cv2.contourArea)
            
            # matchShapes: result is lower for better match
            diff = cv2.matchShapes(c_pred, c_target, cv2.CONTOURS_MATCH_I1, 0)
            
            # 轉換為 0~1 分數 (Diff > 1 視為完全不像)
            score = max(0.0, 1.0 - diff)
            return score
            
        except Exception:
            return 0.0
