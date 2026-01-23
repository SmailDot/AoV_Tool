import cv2
import numpy as np
import os
import shutil
from app.vision.optimizer import AutoTuner

def create_test_data():
    """
    產生測試用的合成圖片
    - 原始圖: 黑色背景 + 白色圓形 + 噪點
    - 目標遮罩: 乾淨的白色圓形
    """
    width, height = 400, 400
    
    # 1. Ground Truth Mask (完美的圓)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (200, 200), 50, 255, -1)
    
    # 2. Source Image (加點料)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # 畫圓
    cv2.circle(img, (200, 200), 50, (200, 200, 200), -1) 
    # 加噪點
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    cv2.imwrite("test_source.png", img)
    cv2.imwrite("test_mask.png", mask)
    print("[Test] Generated test_source.png and test_mask.png")
    
    return img, mask

def test_optimization():
    print("="*60)
    print("Testing AutoTuner (Hill Climbing)")
    print("="*60)
    
    # 1. 準備資料
    img, mask = create_test_data()
    
    # 2. 定義一個「不完美」的 Pipeline
    # 故意把參數設錯：minDist 太小，minRadius/maxRadius 範圍不對
    initial_pipeline = [
        {
            "id": "blur",
            "function": "gaussian_blur",
            "parameters": {
                "ksize": {"default": [5, 5]},
                "sigmaX": {"default": 0}
            }
        },
        {
            "id": "detect",
            "function": "hough_circles",
            "parameters": {
                "dp": {"default": 1.2},
                "minDist": {"default": 10},      # 初始值
                "param1": {"default": 50},
                "param2": {"default": 30},
                "minRadius": {"default": 10},    # 初始值 (太小)
                "maxRadius": {"default": 30}     # 初始值 (太小，目標是 50)
            }
        }
    ]
    
    print("\n[Initial State] Parameters:")
    print(f"  - minRadius: {initial_pipeline[1]['parameters']['minRadius']['default']}")
    print(f"  - maxRadius: {initial_pipeline[1]['parameters']['maxRadius']['default']}")
    
    # 3. 執行優化
    tuner = AutoTuner()
    
    # 設定較短的時間限制以便快速測試
    best_pipeline, best_score = tuner.tune_pipeline(
        img, 
        mask, 
        initial_pipeline, 
        max_iterations=100, 
        time_limit=10
    )
    
    # 4. 顯示結果
    print("\n" + "="*60)
    print(f"Final Score: {best_score:.4f}")
    print("Optimized Parameters:")
    
    hough_params = best_pipeline[1]['parameters']
    print(f"  - minRadius: {hough_params['minRadius']['default']} (Target: ~50)")
    print(f"  - maxRadius: {hough_params['maxRadius']['default']} (Target: ~50)")
    print("="*60)
    
    # 簡單驗證
    final_r_max = hough_params['maxRadius']['default']
    if final_r_max > 40:
        print("\n✅ 測試成功！參數已自動修正向正確方向 (半徑變大)。")
    else:
        print("\n⚠️ 測試未達預期。可能需要更多迭代次數。")

    # 清理
    if os.path.exists("test_source.png"): os.remove("test_source.png")
    if os.path.exists("test_mask.png"): os.remove("test_mask.png")

if __name__ == "__main__":
    test_optimization()
