import cv2 as cv
import numpy as np
import os
import time
import pickle
import threading
from PIL import Image, ImageDraw, ImageFont

# === 1. 設定區 ===
RTSP_URL = "rtsp://192.168.178.214:554/"
PROCESS_WIDTH = 1280 
MEMORY_FILE = "cctv_memory_v7.pkl" # 新檔名
FONT_PATH = "msjh.ttc" 

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# === 2. 顏色定義 ===
COLOR_NORMAL = (0, 255, 0)      # 綠 (SIFT 確認)
COLOR_COLOR_OK = (0, 200, 255)  # 橘 (SIFT 失敗，但顏色確認)
COLOR_WARNING = (255, 255, 0)   # 黃 (遮蔽/不穩)
COLOR_DANGER = (255, 0, 0)      # 紅 (遺失)

# === 3. 多執行緒攝影機 ===
class CameraStream:
    def __init__(self, src):
        self.capture = cv.VideoCapture(src, cv.CAP_FFMPEG)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.capture.read()
        self.stopped = False
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
    def update(self):
        while not self.stopped:
            ret, frame = self.capture.read()
            if ret: self.ret, self.frame = ret, frame
    def read(self): return self.ret, self.frame
    def stop(self): self.stopped = True; self.capture.release()

# === 4. 核心演算法區 ===

def cv2_add_chinese_text(img, text, position, textColor, textSize=20):
    if (isinstance(img, np.ndarray)): img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try: fontStyle = ImageFont.truetype(FONT_PATH, textSize, encoding="utf-8")
    except: fontStyle = ImageFont.load_default()
    draw.text(position, text, textColor, font=fontStyle, stroke_width=2, stroke_fill=(0,0,0))
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

def resize_frame(frame, width):
    h, w = frame.shape[:2]
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA), ratio

# [新增] 計算顏色直方圖 (HSV 空間)
def calc_color_hist(img, mask=None):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 計算 Hue (色調) 和 Saturation (飽和度) 的直方圖
    # 忽略 Value (亮度)，這樣受光影影響較小
    hist = cv.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    cv.normalize(hist, hist, 0, 1, cv.NORM_MINMAX)
    return hist

# [新增] 比較兩個直方圖的相似度
def compare_histograms(hist1, hist2):
    # 使用巴氏距離 (Bhattacharyya)，結果越小越相似 (0=完全一樣, 1=完全不同)
    # 轉成相似度分數 (0~1)，1 代表完全一樣
    score = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    return score

# ... (PolygonEditor 和 precise_polygon_select 與 V6 相同，省略以節省篇幅，請直接沿用 V6 的代碼) ...
# 為了確保程式碼完整性，我還是把這段放進來，如果您已經有 V6，可以直接複製 V6 的 class PolygonEditor 到這裡
# --- 這裡插入 V6 的 PolygonEditor 類別與 precise_polygon_select 函式 ---
class PolygonEditor:
    def __init__(self, window_name, img):
        self.window_name = window_name
        self.original_img = img.copy()
        self.display_img = img.copy()
        self.points = []
        self.dragging_idx = -1
        self.hover_idx = -1
        self.done = False
    def mouse_callback(self, event, x, y, flags, param):
        self.hover_idx = -1
        for i, p in enumerate(self.points):
            if np.linalg.norm(np.array(p) - np.array((x, y))) < 10:
                self.hover_idx = i
                break
        if event == cv.EVENT_LBUTTONDOWN:
            if self.hover_idx != -1: self.dragging_idx = self.hover_idx
            else: self.points.append((x, y))
        elif event == cv.EVENT_MOUSEMOVE:
            if self.dragging_idx != -1: self.points[self.dragging_idx] = (x, y)
        elif event == cv.EVENT_LBUTTONUP: self.dragging_idx = -1
        elif event == cv.EVENT_RBUTTONDOWN:
            if self.hover_idx != -1: self.points.pop(self.hover_idx)
    def run(self):
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.mouse_callback)
        while not self.done:
            self.display_img = self.original_img.copy()
            if len(self.points) > 0:
                pts_np = np.array(self.points, np.int32).reshape((-1, 1, 2))
                is_closed = len(self.points) > 2
                cv.polylines(self.display_img, [pts_np], is_closed, (0, 255, 255), 2)
                for i, p in enumerate(self.points):
                    cv.circle(self.display_img, p, 5, (0, 0, 255) if i!=self.hover_idx else (0, 255, 0), -1)
            cv.imshow(self.window_name, self.display_img)
            key = cv.waitKey(10) & 0xFF
            if key == 13 or key == 32: 
                if len(self.points) >= 3: self.done = True
            elif key == 27: self.points = []; self.done = True
        cv.destroyWindow(self.window_name)
        return self.points

def precise_polygon_select(cap, frame_width):
    ret, frame = cap.read()
    if not ret: return None
    small_frame, _ = resize_frame(frame, frame_width)
    roi_rough = cv.selectROI("1. Rough Select", small_frame, showCrosshair=True)
    cv.destroyWindow("1. Rough Select")
    if roi_rough[2] == 0: return None
    rx, ry, rw, rh = [int(v) for v in roi_rough]
    crop = small_frame[ry:ry+rh, rx:rx+rw]
    zoom_scale = frame_width / float(rw)
    zoomed_img = cv.resize(crop, None, fx=zoom_scale, fy=zoom_scale)
    editor = PolygonEditor("2. Fine Polygon Edit", zoomed_img)
    poly_points_zoomed = editor.run()
    if not poly_points_zoomed: return None
    final_pts = []
    for px, py in poly_points_zoomed:
        real_x = int(rx + px / zoom_scale)
        real_y = int(ry + py / zoom_scale)
        final_pts.append((real_x, real_y))
    return final_pts
# ---------------------------------------------------------------

def save_memory(db):
    data = []
    for obj in db:
        kp_data = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in obj['kp']]
        # 新增 hist 欄位
        data.append({
            "name": obj['name'], "poly_pts": obj['poly_pts'], "des": obj['des'], 
            "kp_data": kp_data, "hist": obj['hist'], "threshold": obj['threshold']
        })
    with open(MEMORY_FILE, 'wb') as f: pickle.dump(data, f)
    print(f"💾 設定已儲存！")

def load_memory():
    if not os.path.exists(MEMORY_FILE): return []
    try:
        with open(MEMORY_FILE, 'rb') as f: loaded_data = pickle.load(f)
        db = []
        for item in loaded_data:
            kps = [cv.KeyPoint(x=k[0][0], y=k[0][1], size=k[1], angle=k[2], response=k[3], octave=k[4], class_id=k[5]) for k in item['kp_data']]
            # 相容性檢查：如果舊存檔沒有 hist，設為 None
            hist = item.get('hist', None)
            thresh = item.get('threshold', 4)
            db.append({
                "name": item['name'], "poly_pts": item['poly_pts'], "des": item['des'], 
                "kp": kps, "hist": hist, "threshold": thresh,
                "status": "Init", "missing_timer": 0
            })
        return db
    except: return []

# === 5. 主程式邏輯 ===
objects_db = []
stable_counter = 0

cam = CameraStream(RTSP_URL).start()
time.sleep(1.0)

# --- 註冊流程 ---
objects_db = load_memory()
if len(objects_db) > 0:
    print(f"已載入 {len(objects_db)} 個舊設定。按 'n' 重新標記...")
    # (省略 input 邏輯以簡化)
else:
    print("【模式：物品註冊 + 顏色採樣】")
    sift = cv.SIFT_create()
    while True:
        poly_pts = precise_polygon_select(cam, PROCESS_WIDTH)
        if poly_pts is None: break
        
        ret, frame = cam.read()
        small_frame, _ = resize_frame(frame, PROCESS_WIDTH)
        
        # 建立 Mask
        mask = np.zeros(small_frame.shape[:2], dtype=np.uint8)
        cv.fillPoly(mask, [np.array(poly_pts)], 255)
        
        # 1. SIFT 特徵
        kp, des = sift.detectAndCompute(small_frame, mask)
        
        # 2. 顏色直方圖 (Color Histogram)
        hist = calc_color_hist(small_frame, mask)
        
        if des is not None:
            name = input("輸入物品名稱: ")
            if name == "": name = f"Item_{len(objects_db)}"
            
            # [自適應門檻] 
            # 如果初始特徵點很少 (<10)，門檻降到 3，否則維持 5
            # 這能解決「眼鏡/衛生紙」特徵不足的問題
            adaptive_thresh = 3 if len(kp) < 20 else 5
            print(f"   > 初始特徵點: {len(kp)}，設定匹配門檻: {adaptive_thresh}")
            
            objects_db.append({
                "name": name, "kp": kp, "des": des, "poly_pts": poly_pts,
                "hist": hist, "threshold": adaptive_thresh,
                "status": "Init", "missing_timer": 0
            })
            save_memory(objects_db)
        else:
            print("⚠️ 無法提取特徵！")

if len(objects_db) == 0:
    cam.stop()
    exit()

print(f"🚀 開始監控 (SIFT + Color 雙重驗證)...")
bg_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
bf = cv.BFMatcher()
sift = cv.SIFT_create()

while True:
    ret, frame = cam.read()
    if not ret: continue

    proc_frame, _ = resize_frame(frame, PROCESS_WIDTH)
    display_frame = proc_frame.copy()
    
    # 動態偵測
    fg_mask = bg_subtractor.apply(proc_frame)
    motion = np.count_nonzero(fg_mask) / (proc_frame.shape[0]*proc_frame.shape[1])
    
    if motion > 0.05:
        stable_counter = 0
        display_frame = cv2_add_chinese_text(display_frame, "動態偵測中...", (10, 30), COLOR_WARNING, 25)
    else:
        stable_counter += 1
    
    draw_list = []
    if stable_counter > 5:
        display_frame = cv2_add_chinese_text(display_frame, "監控中", (10, 30), COLOR_NORMAL, 25)
        kp_scene, des_scene = sift.detectAndCompute(proc_frame, None)
        
        # 即使 SIFT 沒找到任何點，我們也要跑迴圈檢查「顏色」
        if True: 
            for obj in objects_db:
                # 1. SIFT 檢查
                sift_ok = False
                match_count = 0
                if des_scene is not None and obj['des'] is not None:
                    matches = bf.knnMatch(obj["des"], des_scene, k=2)
                    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                    match_count = len(good)
                    if match_count >= obj['threshold']:
                        sift_ok = True

                # 2. 顏色檢查 (救援機制)
                # 針對原本的位置，切一塊出來算顏色
                pts_arr = np.array(obj['poly_pts'])
                rect = cv.boundingRect(pts_arr)
                x, y, w, h = rect
                
                # 安全邊界檢查
                y1, y2 = max(0, y), min(proc_frame.shape[0], y+h)
                x1, x2 = max(0, x), min(proc_frame.shape[1], x+w)
                roi_curr = proc_frame[y1:y2, x1:x2]
                
                # 建立局部 mask (為了濾掉背景)
                mask_curr = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                # 將多邊形平移到 ROI 座標系
                poly_shifted = pts_arr - [x1, y1]
                cv.fillPoly(mask_curr, [poly_shifted], 255)
                
                color_score = 0
                if roi_curr.size > 0:
                    hist_curr = calc_color_hist(roi_curr, mask_curr)
                    color_score = compare_histograms(obj['hist'], hist_curr)
                
                # 判定：顏色相關度 > 0.6 就算顏色吻合
                color_ok = color_score > 0.6
                
                # --- 綜合判定邏輯 ---
                status_msg = ""
                box_color = COLOR_DANGER
                
                if sift_ok:
                    obj["status"] = "正常"
                    obj["missing_timer"] = 0
                    box_color = COLOR_NORMAL
                    status_msg = f"SIFT:{match_count}"
                
                elif color_ok:
                    # SIFT 失敗，但顏色還在 -> 判定為「特徵不足但仍在位」
                    obj["status"] = "正常(Color)"
                    obj["missing_timer"] = 0
                    box_color = COLOR_COLOR_OK # 橘色
                    status_msg = f"Color:{color_score:.2f}"
                
                else:
                    # 兩者都失敗
                    obj["missing_timer"] += 1
                    if obj["missing_timer"] > 40:
                        obj["status"] = "遺失"
                        box_color = COLOR_DANGER
                    else:
                        obj["status"] = "遮蔽"
                        box_color = COLOR_WARNING
                    status_msg = f"Lost..{obj['missing_timer']}"

                # 繪圖
                cv.polylines(display_frame, [pts_arr], True, box_color, 2)
                
                text_y = y - 30 if y - 30 > 0 else y + h + 10
                draw_list.append({
                    "text": f"{obj['name']}: {obj['status']} ({status_msg})",
                    "pos": (x, text_y),
                    "color": box_color
                })

    for item in draw_list:
        display_frame = cv2_add_chinese_text(display_frame, item['text'], item['pos'], item['color'], 16)

    cv.imshow('Smart CCTV V7 (Hybrid)', display_frame)
    if cv.waitKey(1) & 0xFF == 27: break
    if cv.waitKey(1) & 0xFF == ord('r'):
        objects_db = []
        os.remove(MEMORY_FILE)
        cv.destroyAllWindows()

cam.stop()
cv.destroyAllWindows()