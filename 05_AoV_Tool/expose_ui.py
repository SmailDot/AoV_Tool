import os
import sys
import time
import subprocess
from threading import Thread

try:
    from pyngrok import ngrok, conf
except ImportError:
    print("錯誤: 找不到 pyngrok 套件。")
    print("請執行: pip install pyngrok")
    sys.exit(1)

def start_streamlit():
    """啟動 Streamlit"""
    print("[Launcher] Starting Streamlit...")
    # 使用 subprocess 啟動 streamlit，並讓它在背景執行
    subprocess.call([sys.executable, "-m", "streamlit", "run", "aov_app.py", "--server.port=8501", "--server.headless=true"])

def start_tunnel():
    """啟動 ngrok"""
    # 如果需要 token，請在這裡設定或使用 'ngrok config add-authtoken'
    # ngrok.set_auth_token("YOUR_TOKEN")
    
    # 稍微等待 Streamlit 啟動
    time.sleep(3)
    
    try:
        # 建立 HTTP 隧道指向 8501
        public_url = ngrok.connect(8501).public_url
        print("\n" + "="*60)
        print(f"🚀 UI 已上線！請在實驗室電腦瀏覽器開啟以下網址：")
        print(f"\n    {public_url}")
        print("\n" + "="*60)
    except Exception as e:
        print(f"\n[Error] ngrok 啟動失敗: {e}")
        print("如果是 Auth 錯誤，請執行: ngrok config add-authtoken <TOKEN>")

if __name__ == "__main__":
    # 使用 Thread 同時啟動 ngrok 和 Streamlit
    # 因為 Streamlit 會佔用主執行緒，所以我們先啟動 ngrok 監聽
    tunnel_thread = Thread(target=start_tunnel)
    tunnel_thread.daemon = True
    tunnel_thread.start()
    
    # 在主執行緒啟動 Streamlit
    start_streamlit()
