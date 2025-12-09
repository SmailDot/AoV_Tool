import cv2 as cv
import numpy as np

# === 1. 準備實驗素材 ===
img_full = cv.imread('test_image.png')
if img_full is None:
    print("找不到圖片")
    exit()

# 裁切中間一塊區域來做實驗 (避免旋轉後黑邊太多干擾)
h, w = img_full.shape[:2]
img1 = img_full[int(h/4):int(3*h/4), int(w/4):int(3*w/4)]

# === 製造「旋轉版」影像 (旋轉 45 度) ===
h, w = img1.shape[:2]
center = (w // 2, h // 2)
# 取得旋轉矩陣 (旋轉 45 度，縮放 1.0)
M = cv.getRotationMatrix2D(center, 45, 1.0)
img2 = cv.warpAffine(img1, M, (w, h))

print("正在進行 Harris 旋轉匹配實驗...")

# === 2. 定義檢測函數 ===
def get_harris_keypoints_and_descriptors(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 使用 Harris 找角點
    corners = cv.goodFeaturesToTrack(
        gray, maxCorners=200, qualityLevel=0.01, minDistance=10, 
        useHarrisDetector=True, k=0.04
    )
    
    keypoints = []
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            # 【關鍵】Harris 找出來的點，angle 預設是 -1 (沒有方向)
            keypoints.append(cv.KeyPoint(x=float(x), y=float(y), size=10))
            
    # 借用 SIFT 算描述子
    sift = cv.SIFT_create()
    # 因為 keypoints 沒有角度資訊，SIFT 無法進行「旋轉校正」
    kps, des = sift.compute(gray, keypoints)
    return kps, des

# === 3. 執行檢測與計算 ===
kp1, des1 = get_harris_keypoints_and_descriptors(img1)
kp2, des2 = get_harris_keypoints_and_descriptors(img2)

print(f"原圖找到 {len(kp1)} 個點")
print(f"旋轉圖找到 {len(kp2)} 個點 (數量應該差不多，證明 Harris 檢測本身不怕旋轉)")

# === 4. 執行匹配 ===
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 篩選前 30 個匹配
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:30]

# === 5. 繪製結果 ===
# 這裡我們開啟 DRAW_RICH_KEYPOINTS 來看看 Harris 的「空心」指針
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                             matchColor=(0, 255, 0))

output_file = 'harris_rotation_fail.jpg'
cv.imwrite(output_file, img_matches)
cv.imshow('Harris Rotation Failure', img_matches)

print(f"實驗完成！結果已儲存為 {output_file}")
print("觀察重點：")
print("1. 點的位置：Harris 還是有抓到角點 (這是它厲害的地方)。")
print("2. 連線：但是連線應該會亂七八糟 (因為沒有方向資訊，描述子無法對齊)。")

cv.waitKey(0)
cv.destroyAllWindows()