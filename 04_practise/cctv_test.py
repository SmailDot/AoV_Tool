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
MEMORY_FILE = "cctv_memory_v6.pkl" # 改個檔名避免格式衝突
FONT_PATH = "msjh.ttc" 

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# === 2. 顏色定義 (RGB) ===
COLOR_NORMAL = (0, 255, 0)      # 綠
COLOR_WARNING = (255, 255, 0)   # 黃
COLOR_DANGER = (255, 0, 0)      # 紅

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
            if ret:
                self.ret, self.frame = ret, frame
    def read(self):
        return self.ret, self.frame
    def stop(self):
        self.stopped = True
        self.capture.release()

# === 4. 輔助函式區 ===
def cv2_add_chinese_text(img, text, position, textColor, textSize=20):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        fontStyle = ImageFont.truetype(FONT_PATH, textSize, encoding="utf-8")
    except:
        fontStyle = ImageFont.load_default()
    # 描邊效果
    draw.text(position, text, textColor, font=fontStyle, stroke_width=2, stroke_fill=(0,0,0))
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

def resize_frame(frame, width):
    h, w = frame.shape[:2]
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA), ratio

# === [核心升級] 可編輯多邊形編輯器 ===
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
        # 1. 檢查滑鼠是否懸停在某個點上 (用於拖拉或刪除)
        self.hover_idx = -1
        for i, p in enumerate(self.points):
            if np.linalg.norm(np.array(p) - np.array((x, y))) < 10: # 感應距離 10px
                self.hover_idx = i
                break

        # 2. 左鍵按下：新增點 或 開始拖拉
        if event == cv.EVENT_LBUTTONDOWN:
            if self.hover_idx != -1:
                self.dragging_idx = self.hover_idx # 抓住了！開始拖拉
            else:
                self.points.append((x, y)) # 沒抓到東西，就新增一個點

        # 3. 滑鼠移動：拖拉中
        elif event == cv.EVENT_MOUSEMOVE:
            if self.dragging_idx != -1:
                self.points[self.dragging_idx] = (x, y) # 更新點的位置

        # 4. 左鍵放開：結束拖拉
        elif event == cv.EVENT_LBUTTONUP:
            self.dragging_idx = -1

        # 5. 右鍵點擊：刪除該點
        elif event == cv.EVENT_RBUTTONDOWN:
            if self.hover_idx != -1:
                self.points.pop(self.hover_idx)

    def run(self):
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.mouse_callback)
        
        while not self.done:
            self.display_img = self.original_img.copy()
            
            # 畫出多邊形連線
            if len(self.points) > 0:
                # 畫線
                pts_np = np.array(self.points, np.int32).reshape((-1, 1, 2))
                is_closed = len(self.points) > 2
                cv.polylines(self.display_img, [pts_np], is_closed, (0, 255, 255), 2)
                
                # 畫點 (節點)
                for i, p in enumerate(self.points):
                    color = (0, 0, 255) # 紅色 (一般)
                    radius = 5
                    if i == self.hover_idx or i == self.dragging_idx:
                        color = (0, 255, 0) # 綠色 (選中/拖拉中)
                        radius = 8
                    cv.circle(self.display_img, p, radius, color, -1)
                    # 畫個外框增加對比
                    cv.circle(self.display_img, p, radius, (0, 0, 0), 1)

            # 提示文字
            info = f"Points: {len(self.points)} | Enter: Finish | Right Click: Delete Point"
            cv.putText(self.display_img, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(self.display_img, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            cv.imshow(self.window_name, self.display_img)
            
            key = cv.waitKey(10) & 0xFF
            if key == 13 or key == 32: # Enter/Space 完成
                if len(self.points) >= 3:
                    self.done = True
                else:
                    print("⚠️ 至少需要 3 個點才能構成多邊形！")
            elif key == 27: # ESC 取消
                self.points = []
                self.done = True

        cv.destroyWindow(self.window_name)
        return self.points

# === 整合：兩段式放大 + 多邊形編輯器 ===
def precise_polygon_select(cap, frame_width):
    # 1. 抓圖
    ret, frame = cap.read()
    if not ret: return None
    small_frame, _ = resize_frame(frame, frame_width)
    
    # 2. 第一階段：粗略框選 (使用內建矩形)
    print(">> 步驟 1/2: 請框選大致區域以放大 (Space 確認)")
    roi_rough = cv.selectROI("1. Rough Select (Rectangle)", small_frame, showCrosshair=True)
    cv.destroyWindow("1. Rough Select (Rectangle)")
    
    if roi_rough[2] == 0 or roi_rough[3] == 0: return None

    # 3. 放大
    rx, ry, rw, rh = [int(v) for v in roi_rough]
    crop = small_frame[ry:ry+rh, rx:rx+rw]
    zoom_scale = frame_width / float(rw)
    zoomed_img = cv.resize(crop, None, fx=zoom_scale, fy=zoom_scale)
    
    # 4. 第二階段：多邊形編輯 (使用自定義編輯器)
    print(">> 步驟 2/2: 點擊新增點，按住拖拉點，右鍵刪除點 (Enter 完成)")
    editor = PolygonEditor("2. Fine Polygon Edit (Draggable)", zoomed_img)
    poly_points_zoomed = editor.run()
    
    if not poly_points_zoomed: return None

    # 5. 座標還原
    final_pts = []
    for px, py in poly_points_zoomed:
        real_x = int(rx + px / zoom_scale)
        real_y = int(ry + py / zoom_scale)
        final_pts.append((real_x, real_y))
        
    return final_pts

# === 存檔/讀檔 ===
def save_memory(db):
    data = []
    for obj in db:
        kp_data = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in obj['kp']]
        data.append({"name": obj['name'], "poly_pts": obj['poly_pts'], "des": obj['des'], "kp_data": kp_data})
    with open(MEMORY_FILE, 'wb') as f: pickle.dump(data, f)
    print(f"💾 設定已儲存！")

def load_memory():
    if not os.path.exists(MEMORY_FILE): return []
    try:
        with open(MEMORY_FILE, 'rb') as f: loaded_data = pickle.load(f)
        db = []
        for item in loaded_data:
            kps = [cv.KeyPoint(x=k[0][0], y=k[0][1], size=k[1], angle=k[2], response=k[3], octave=k[4], class_id=k[5]) for k in item['kp_data']]
            db.append({"name": item['name'], "poly_pts": item['poly_pts'], "des": item['des'], "kp": kps, "status": "Init", "missing_timer": 0})
        return db
    except: return []

# === 5. 主程式邏輯 ===
objects_db = []
min_match_count = 4
stable_counter = 0

cam = CameraStream(RTSP_URL).start()
time.sleep(1.0)

# --- 初始化流程 ---
objects_db = load_memory()
if len(objects_db) > 0:
    print(f"已載入 {len(objects_db)} 個舊設定。按 'n' 重新標記，其他鍵繼續...")
    # 若要實作按鍵選擇，可在此處加入 input 或 waitKey 邏輯
else:
    print("【模式：多邊形物品註冊】")
    sift = cv.SIFT_create()
    while True:
        poly_pts = precise_polygon_select(cam, PROCESS_WIDTH)
        if poly_pts is None: break
        
        # 取得最新畫面切圖
        ret, frame = cam.read()
        small_frame, _ = resize_frame(frame, PROCESS_WIDTH)
        
        # 建立 Mask
        mask = np.zeros(small_frame.shape[:2], dtype=np.uint8)
        cv.fillPoly(mask, [np.array(poly_pts)], 255)
        
        kp, des = sift.detectAndCompute(small_frame, mask)
        
        if des is not None and len(des) > 0:
            name = input("輸入物品名稱: ")
            if name == "": name = f"Item_{len(objects_db)}"
            objects_db.append({
                "name": name, "kp": kp, "des": des, "poly_pts": poly_pts,
                "status": "Init", "missing_timer": 0
            })
            save_memory(objects_db)
        else:
            print("⚠️ 特徵不足！")

if len(objects_db) == 0:
    cam.stop()
    exit()

print(f"🚀 開始監控...")
bg_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
bf = cv.BFMatcher()
sift = cv.SIFT_create()

while True:
    ret, frame = cam.read()
    if not ret: continue # 這裡不會阻塞，因為是多執行緒

    proc_frame, _ = resize_frame(frame, PROCESS_WIDTH)
    display_frame = proc_frame.copy()
    
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
        
        if des_scene is not None:
            for obj in objects_db:
                matches = bf.knnMatch(obj["des"], des_scene, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                
                # 計算多邊形中心與邊框
                pts_arr = np.array(obj['poly_pts'])
                rect = cv.boundingRect(pts_arr)
                x, y, w, h = rect
                
                if len(good) >= min_match_count:
                    obj["status"] = "正常"
                    obj["missing_timer"] = 0
                    color = COLOR_NORMAL
                else:
                    obj["missing_timer"] += 1
                    if obj["missing_timer"] > 40:
                        obj["status"] = "遺失"
                        color = COLOR_DANGER
                    else:
                        obj["status"] = "遮蔽"
                        color = COLOR_WARNING
                
                # 畫多邊形
                cv.polylines(display_frame, [pts_arr], True, color, 2)
                
                # 文字避讓
                text_y = y - 30 if y - 30 > 0 else y + h + 10
                draw_list.append({
                    "text": f"{obj['name']}: {obj['status']}",
                    "pos": (x, text_y),
                    "color": color
                })

    for item in draw_list:
        display_frame = cv2_add_chinese_text(display_frame, item['text'], item['pos'], item['color'], 18)

    cv.imshow('Smart CCTV V6 (Editable Poly)', display_frame)
    
    key = cv.waitKey(1) & 0xFF
    if key == 27: break
    if key == ord('r') or key == ord('R'):
        objects_db = []
        os.remove(MEMORY_FILE)
        cv.destroyAllWindows()
        print("重置設定...")

cam.stop()
cv.destroyAllWindows()