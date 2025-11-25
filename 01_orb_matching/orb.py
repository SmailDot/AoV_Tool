import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取影像 (請換成您的 CCTV 截圖路徑)
# 為了模擬兩張圖，我們手動把同一張圖做一點旋轉和縮放
img1 = cv.imread('test_image.png', cv.IMREAD_GRAYSCALE)  # QueryImage
if img1 is None:
    print("找不到圖片，請確認路徑！")
    exit()

# 模擬第二張圖：旋轉 15 度並縮小
rows, cols = img1.shape
M = cv.getRotationMatrix2D((cols/2, rows/2), 15, 0.8) # 旋轉15度, 縮放0.8
img2 = cv.warpAffine(img1, M, (cols, rows)) # TrainImage

# 2. 初始化 ORB 檢測器
orb = cv.ORB_create(nfeatures=1000) # 假設只找 10 個點

# 3. 尋找關鍵點 (Keypoints) 與 計算描述子 (Descriptors)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 建立匹配器 (Brute-Force Matcher)
# ORB 是二進制描述子，使用 Hamming 距離比較快且準
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# 5. 執行匹配
matches = bf.match(des1, des2)

# 6. 根據距離排序，只取前 20 個最好的匹配點 (距離越小越好)
matches = sorted(matches, key = lambda x:x.distance)
good_matches = matches[:20]

# 7. 繪製匹配結果
print("正在繪製並儲存圖片...")
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# --- 新增：儲存圖片 ---
#這行會把結果存成 'result_matches.jpg'，解析度是原始像素等級，沒有任何截圖誤差
output_filename = 'result_matches.jpg'
cv.imwrite(output_filename, img_matches)
print(f"成功！圖片已儲存為：{output_filename}")

# --- 新增：如果您想要看單張圖上的特徵點 (跟原圖一樣大) ---
img_kp1 = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
cv.imwrite('result_keypoints_original.jpg', img_kp1)
print("成功！原圖特徵點已儲存為：result_keypoints_original.jpg")