import requests
import base64
import json
import os
import sys
import time

# ================= 設定區 =================
# 請填入您的 Server 網址 (本地測試用 localhost，遠端用 ngrok)
API_URL = "http://localhost:5000/process" 
# API_URL = "https://xxxx.ngrok-free.app/process" 

# 測試用的檔案 (圖片或影片) - 僅在 execution_mode='full' 時需要
INPUT_FILE = "test1.jpg" 

# 查詢指令
QUERY = "偵測這張圖的邊緣"

# 執行模式: 'full' (運算+圖片) 或 'plan_only' (只產 JSON)
EXECUTION_MODE = 'plan_only'
# ==========================================

def encode_file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def main():
    payload = {
        "query": QUERY,
        "use_mock": True,
        "execution_mode": EXECUTION_MODE
    }

    # 只有在完整模式才需要讀檔
    if EXECUTION_MODE == 'full':
        if not os.path.exists(INPUT_FILE):
            print(f"錯誤: 找不到檔案 '{INPUT_FILE}'")
            return
        print(f"1. 讀取檔案: {INPUT_FILE}")
        payload["image_base64"] = encode_file_to_base64(INPUT_FILE)
        payload["return_base64"] = True
    else:
        print(f"1. 模式: {EXECUTION_MODE} (跳過檔案讀取)")

    print(f"2. 發送請求至: {API_URL}")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 成功! (耗時: {time.time() - start_time:.2f}s)")
            print(f"   - 狀態: {result['status']}")
            
            # Print Pipeline JSON Structure
            if result.get('pipeline_json'):
                print(f"   - Pipeline JSON: ✅ (包含 {len(result['pipeline_json'])} 個節點)")
                print("     [節點列表]:")
                for node in result['pipeline_json']:
                    print(f"       -> {node.get('name')} ({node.get('function')})")
            
            print(f"   - FPGA: {result.get('fpga_estimated_clk')} clk")
            
            if result.get('generated_code'):
                print(f"   - 程式碼生成: ✅ (長度: {len(result['generated_code'])} chars)")
            
            # Save full JSON response for inspection
            with open("response_debug.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   - 完整回應已存為: response_debug.json")

            # 處理回傳的檔案 (僅限 full 模式)
            if EXECUTION_MODE == 'full' and result.get('output_base64'):
                output_filename = f"output_{int(time.time())}.png"
                with open(output_filename, "wb") as f:
                    f.write(base64.b64decode(result['output_base64']))
                print(f"   - 結果圖片已存為: {output_filename}")
                
        else:
            print(f"\n❌ 失敗 (Status {response.status_code})")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ 連線錯誤: {e}")

if __name__ == "__main__":
    # 檢查依賴
    try:
        import requests
    except ImportError:
        print("請先安裝 requests: pip install requests")
        sys.exit(1)
        
    main()
