import os
import sys
import threading
import time

try:
    from pyngrok import ngrok, conf
except ImportError:
    print("錯誤: 找不到 pyngrok 套件。")
    print("請執行: pip install pyngrok")
    sys.exit(1)

# Import the existing app
from app_server import app

def start_tunnel():
    """
    啟動 ngrok 隧道並印出公開網址
    """
    # 如果您有 ngrok authtoken，可以在這裡設定 (選填，但建議使用以獲得更穩定的連線)
    # ngrok.set_auth_token("YOUR_AUTHTOKEN_HERE")
    
    # Open a HTTP tunnel on the default port 5000
    try:
        public_url = ngrok.connect(5000).public_url
        print("\n" + "="*60)
        print(f"🚀 隧道已建立！請在實驗室電腦使用以下網址：")
        print(f"\n    {public_url}")
        print(f"\n    API Endpoint: {public_url}/process")
        print("="*60 + "\n")
    except Exception as e:
        print(f"ngrok 連線失敗: {e}")
        print("提示: 如果您看到 ERR_NGROK_4018，代表您需要註冊 ngrok 帳號並設定 Authtoken。")
        print("前往 https://dashboard.ngrok.com/get-started/your-authtoken 取得 Token")
        print("然後在程式碼中取消註解: ngrok.set_auth_token(...)")

if __name__ == "__main__":
    # Start ngrok in a separate brief delay or just before app run
    # pyngrok starts a background process, so we can just call it
    start_tunnel()
    
    print("[Server] Starting Flask App on port 5000...")
    # Disable reloader to prevent double-starting ngrok
    app.run(host='0.0.0.0', port=5000, use_reloader=False)
