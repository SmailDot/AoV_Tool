import cv2
import os

# IP 位址 (根據您的截圖)
IP = "192.168.178.214"
PORT = "554" # RTSP 預設埠號

# 設定環境變數：強制使用 TCP 傳輸 (對 H.265 很重要！)
# 如果不加這行，高解析度 (2880x1620) 的 H.265 很容易因為掉封包而連線失敗
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 常見的 RTSP 路徑列表 (包含 Hikvision, Dahua, TP-Link, 一般 ONVIF)
paths_to_test = [
    "",                         # 只有 IP
    "/stream1",                 # 通用
    "/stream2",                 # 通用 (子串流)
    "/live/ch0",                # 通用
    "/live/main",               # 通用
    "/h265",                    # H.265 專用
    "/h264",                    # H.264
    "/profile1",                # ONVIF
    "/onvif1",                  # ONVIF
    "/Streaming/Channels/101",  # Hikvision (海康威視) 主串流
    "/Streaming/Channels/102",  # Hikvision 子串流
    "/cam/realmonitor?channel=1&subtype=0", # Dahua (大華) 主串流
    "/cam/realmonitor?channel=1&subtype=1", # Dahua 子串流
    "/11",                      # 某些老舊機型
    "/12"
]

print(f"開始掃描 IP: {IP} 的 RTSP 路徑...")
print("-" * 50)

found = False

for path in paths_to_test:
    # 組合出完整的 URL
    if path == "":
        url = f"rtsp://{IP}:{PORT}/"
    else:
        url = f"rtsp://{IP}:{PORT}{path}"
    
    print(f"嘗試連線: {url} ... ", end="", flush=True)
    
    # 嘗試開啟
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if cap.isOpened():
        print("✅ 成功！")
        print("-" * 50)
        print(f"🎉 找到正確網址了：\n{url}")
        print("-" * 50)
        
        # 讀一幀來確認真的有畫面
        ret, frame = cap.read()
        if ret:
            print(f"畫面解析度: {frame.shape[1]}x{frame.shape[0]}")
            cv2.imshow('Camera Test', frame)
            cv2.waitKey(0) # 按任意鍵關閉
            cv2.destroyAllWindows()
        else:
            print("⚠️ 連線成功但讀不到畫面 (可能是解碼問題)")
        
        cap.release()
        found = True
        break # 找到就停止
    else:
        print("❌ 失敗")

if not found:
    print("-" * 50)
    print("掃描結束，未找到可用路徑。")
    print("建議：")
    print("1. 查看攝影機機身上的『品牌』(如 Hikvision, D-Link)")
    print("2. 詢問實驗室管理員是否有設定『帳號密碼』")