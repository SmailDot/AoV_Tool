import cv2 as cv
import numpy as np

# 1. 讀取原始影像
img1 = cv.imread('test_image.png', cv.IMREAD_GRAYSCALE)
if img1 is None:
    print("找不到圖片")
    exit()

# 2. 製作「旋轉後」的影像 (作為對照組)
rows, cols = img1.shape
# 以中心點為軸，旋轉 45 度，並不縮放 (scale=1)
M = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1) 
img2 = cv.warpAffine(img1, M, (cols, rows))

print("正在執行 SIFT 特徵檢測與匹配...")

# 3. 初始化 SIFT 檢測器
sift = cv.SIFT_create()

# 4. 檢測特徵點與計算描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 5. 特徵匹配 (Feature Matching)
# 注意！SIFT 的描述子是浮點數向量，必須使用 NORM_L2 (歐式距離)
# 之前 ORB 是二進制，所以用 NORM_HAMMING，這裡不一樣喔！
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# 執行匹配
matches = bf.match(des1, des2)

# 6. 篩選優良匹配 (只取前 50 個距離最近的點，避免畫面太亂)
matches = sorted(matches, key = lambda x:x.distance)
good_matches = matches[:50]

# 7. 繪製匹配連線
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 8. 顯示並存檔
output_file = 'sift_rotation_proof.jpg'
cv.imwrite(output_file, img_matches)
cv.imshow('SIFT Rotation Invariance Proof', img_matches)

print(f"驗證完成！結果已儲存為 {output_file}")
print("觀察重點：")
print("請看圖中的連線，是否準確連接了「旋轉前」與「旋轉後」的同一個物體？")
print("如果是，即證明 SIFT 描述子具有旋轉不變性。")

cv.waitKey(0)
cv.destroyAllWindows()