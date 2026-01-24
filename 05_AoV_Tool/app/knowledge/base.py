import os
import json
import torch
import clip
import numpy as np
import cv2
import faiss
from PIL import Image
from typing import List, Dict, Tuple, Optional

class KnowledgeBase:
    """
    老師傅知識庫 (The Master's Library) - FAISS Enhanced
    功能：
    1. 儲存案例：(圖片特徵, Pipeline, 描述)
    2. 檢索案例：使用 FAISS 進行高效向量相似度搜尋
    """
    
    def __init__(self, db_path="knowledge_db.json"):
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.db = self._load_db()
        
        # Initialize FAISS Index
        self.dimension = 512 # ViT-B/32 output dimension
        self.index = faiss.IndexFlatIP(self.dimension) # Inner Product (Cosine Similarity if normalized)
        self._build_index()
        
        print(f"[KnowledgeBase] Loaded {len(self.db)} cases. Device: {self.device}. Engine: FAISS")

    def _build_index(self):
        """
        Rebuild FAISS index from DB
        """
        if not self.db:
            return
            
        vectors = []
        for entry in self.db:
            vec = np.array(entry['embedding'], dtype=np.float32)
            # Normalize for Cosine Similarity
            faiss.normalize_L2(vec.reshape(1, -1))
            vectors.append(vec)
            
        if vectors:
            matrix = np.vstack(vectors)
            self.index.reset()
            self.index.add(matrix)

    def _load_db(self) -> List[Dict]:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Error] Failed to load DB: {e}")
                return []
        return []

    def save_db(self):
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, indent=2, ensure_ascii=False)

    def add_case(self, image_path: str, pipeline: List[Dict], description: str = ""):
        """
        新增一個成功案例
        """
        # 1. 提取特徵
        features = self._extract_features(image_path)
        if features is None:
            print("[Error] Feature extraction failed.")
            return

        # 2. 建立案例紀錄
        case_id = f"case_{len(self.db) + 1:03d}"
        entry = {
            "id": case_id,
            "description": description,
            "pipeline": pipeline,
            "image_path": image_path, # In real app, might want to store relative path or hash
            "embedding": features.tolist() # Convert numpy/tensor to list for JSON
        }
        
        self.db.append(entry)
        self.save_db()
        
        # 3. Update Index
        vec = features.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        
        print(f"[KnowledgeBase] Saved case {case_id}: {description}")

    def find_similar_cases(self, image_path: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        以圖搜圖：找到最像的案例 (使用 FAISS)
        """
        if not self.db or self.index.ntotal == 0:
            return []
            
        # 1. 提取查詢圖特徵
        query_features = self._extract_features(image_path)
        if query_features is None:
            return []
            
        # 2. 搜尋 (FAISS)
        query_vec = query_features.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        # Ensure k doesn't exceed total vectors
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.db[idx], float(score)))
            
        return results

    def _extract_features(self, image_input) -> Optional[np.ndarray]:
        """
        使用 CLIP 提取影像特徵向量
        """
        try:
            # Handle path or numpy array
            if isinstance(image_input, str):
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                return None
                
            image_preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(image_preprocessed)
                
            return features.cpu().numpy()[0]
            
        except Exception as e:
            print(f"[Error] CLIP feature extraction failed: {e}")
            return None

# --- Demo Usage ---
if __name__ == "__main__":
    # Initialize
    kb = KnowledgeBase()
    
    # 1. Simulate saving a case (Use test1.jpg if exists)
    test_img = "test1.jpg"
    if os.path.exists(test_img):
        # Mock successful pipeline for coin detection
        coin_pipeline = [
            {"name": "Resize", "function": "resize", "parameters": {"width": {"default": 640}}},
            {"name": "Gaussian Blur", "function": "gaussian_blur", "parameters": {"ksize": {"default": [9, 9]}}},
            {"name": "Canny", "function": "canny_edge", "parameters": {"threshold1": {"default": 50}, "threshold2": {"default": 150}}}
        ]
        
        kb.add_case(test_img, coin_pipeline, "Coin detection on keyboard (noisy background)")
        
    # 2. Simulate querying with the same image (Should match 100%)
    if os.path.exists(test_img):
        print("\n[Query] Searching for similar cases...")
        matches = kb.find_similar_cases(test_img)
        for case, score in matches:
            print(f"  - Match: {case['description']} (Score: {score:.4f})")
            print(f"    Suggested Pipeline: {[node['name'] for node in case['pipeline']]}")
