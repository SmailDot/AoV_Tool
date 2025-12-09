import cv2 as cv
import numpy as np

# === 1. 準備實驗素材 ===
# 讀取原始影像
img_full = cv.imread('test_image.png')
if img_full is None:
    print("找不到 test_image.png，請確認路徑！")
    exit()

# 為了模擬尺度變化，我們裁切一塊區域，然後放大它
h, w = img_full.shape[:2]
# 裁切中間區域 (您可以根據需要調整範圍)
roi_small = img_full[int(h/3):int(2*h/3), int(w/3):int(2*h/3)]

# 製造「放大版」影像 (放大 3 倍)
scale_factor = 3
roi_zoomed = cv.resize(roi_small, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)

print(f"圖1(小)尺寸: {roi_small.shape}, 圖2(大)尺寸: {roi_zoomed.shape}")
print("正在進行 Harris 特徵檢測與匹配實驗...")

# === 2. 定義 Harris 檢測並轉換為 KeyPoint 的函數 ===
def get_harris_keypoints(img_bgr):
    """
    使用 Harris 原理找到角點，並轉換為 OpenCV 可用的 KeyPoint 物件
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    
    # 使用 goodFeaturesToTrack，並設定 useHarrisDetector=True
    # 這是 OpenCV 中使用 Harris 最方便的方法
    corners = cv.goodFeaturesToTrack(
        gray,
        maxCorners=200,       # 最多找幾個點
        qualityLevel=0.01,    # 角點品質門檻
        minDistance=10,       # 點與點之間的最小距離
        useHarrisDetector=True, # 【重點】開啟 Harris 模式
        k=0.04
    )
    
    keypoints = []
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            # 將座標轉換為 KeyPoint 物件，size 設為固定值 (因為 Harris 沒有尺度資訊)
            keypoints.append(cv.KeyPoint(x=float(x), y=float(y), size=10))
            
    return keypoints, gray

# === 3. 執行檢測 ===
# 分別在小圖和大圖上找 Harris 點
kp1, gray1 = get_harris_keypoints(roi_small)
kp2, gray2 = get_harris_keypoints(roi_zoomed)

print(f"小圖找到 {len(kp1)} 個 Harris 點")
print(f"大圖找到 {len(kp2)} 個 Harris 點 (注意：很多角點可能因為放大而變成了邊緣，導致找不到)")

# === 4. 計算描述子 (借用 SIFT) ===
# 因為 Harris 沒有描述子，我們借用 SIFT 來計算，才能進行匹配
sift = cv.SIFT_create()
# 注意這裡是用 .compute() 而不是 detectAndCompute()，因為點已經找到了
kp1, des1 = sift.compute(gray1, kp1)
kp2, des2 = sift.compute(gray2, kp2)

# === 5. 執行匹配 (Matching) ===
# 使用暴力匹配器 (BFMatcher)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 根據距離排序，只看前 30 個最好的匹配
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:30]

# === 6. 繪製結果 ===
# 繪製匹配連線
img_matches = cv.drawMatches(roi_small, kp1, roi_zoomed, kp2, good_matches, None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                             matchColor=(0, 255, 0), # 綠色連線
                             singlePointColor=(0, 0, 255)) # 紅色點

# 顯示並存檔
output_file = 'harris_matching_fail.jpg'
cv.imwrite(output_file, img_matches)
cv.imshow('Harris Matching Failure Demo', img_matches)
print(f"\n實驗完成！結果已儲存為 {output_file}")
print("=== 觀察重點 ===")
print("請觀察連線是否正確？")
print("預期結果：由於尺度差異太大，Harris 在兩張圖找到的點無法對應，連線應該會非常雜亂(錯誤匹配)，證明其不具備尺度不變性。")

cv.waitKey(0)
cv.destroyAllWindows()