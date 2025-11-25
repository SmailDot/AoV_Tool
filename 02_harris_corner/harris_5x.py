import cv2 as cv
import numpy as np

# 1. 讀取原始影像
img = cv.imread('test_image.png')
if img is None:
    print("找不到圖片")
    exit()

# 為了方便觀察，我們只裁切一小塊區域 (例如鍵盤或螢幕角)
# 您可能需要根據圖片調整這個範圍，這裡是大概抓個中間區域
h, w = img.shape[:2]
roi = img[int(h/2):int(h/2)+200, int(w/2):int(w/2)+200] 

# --- 測試 A: 原始尺度 (Success) ---
gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
gray_roi = np.float32(gray_roi)
dst_a = cv.cornerHarris(gray_roi, 2, 3, 0.04)
dst_a = cv.dilate(dst_a, None)
roi_display = roi.copy()
roi_display[dst_a > 0.01 * dst_a.max()] = [0, 0, 255] # 標紅點

# --- 測試 B: 放大 5 倍 (Failure) ---
# 將同一塊區域放大 5 倍
scale = 5
roi_zoomed = cv.resize(roi, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
gray_zoomed = cv.cvtColor(roi_zoomed, cv.COLOR_BGR2GRAY)
gray_zoomed = np.float32(gray_zoomed)

# 跑一樣的 Harris 參數
dst_b = cv.cornerHarris(gray_zoomed, 2, 3, 0.04)
dst_b = cv.dilate(dst_b, None)

# 標記紅點
roi_zoomed_display = roi_zoomed.copy()
roi_zoomed_display[dst_b > 0.01 * dst_b.max()] = [0, 0, 255]

# --- 儲存結果 ---
cv.imwrite('harris_success_small.jpg', roi_display)
cv.imwrite('harris_fail_zoomed.jpg', roi_zoomed_display)

print("實驗完成！請查看：")
print("1. harris_success_small.jpg (小圖有紅點)")
print("2. harris_fail_zoomed.jpg (放大後紅點消失或是變很少)")