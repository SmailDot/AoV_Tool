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

# 測試用的檔案 (圖片或影片)
INPUT_FILE = "test1.jpg" 

# 查詢指令
QUERY = "偵測這張圖的邊緣"
# ==========================================

def encode_file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"錯誤: 找不到檔案 '{INPUT_FILE}'")
        print("請修改腳本中的 INPUT_FILE 變數，或將圖片放在同目錄下。")
        return

    print(f"1. 讀取檔案: {INPUT_FILE}")
    base64_str = encode_file_to_base64(INPUT_FILE)
    
    payload = {
        "query": QUERY,
        "image_base64": base64_str,
        "return_base64": True,  # 告訴 Server 我想要把結果圖片也傳回來
        "use_mock": True        # 測試時用 Mock LLM 省錢，正式用可改 False
    }

    print(f"2. 發送請求至: {API_URL}")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300) # 影片處理要設長一點 timeout
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 成功! (耗時: {time.time() - start_time:.2f}s)")
            print(f"   - 狀態: {result['status']}")
            print(f"   - 類型: {result.get('type', 'unknown')}")
            print(f"   - 流程: {result.get('pipeline_summary')}")
            print(f"   - FPGA: {result.get('fpga_estimated_clk')} clk")
            
            if result.get('generated_code'):
                print(f"   - 程式碼生成: ✅ (長度: {len(result['generated_code'])} chars)")
                # Print first 3 lines preview
                code_preview = "\n".join(result['generated_code'].split('\n')[:3])
                print(f"     預覽:\n{code_preview}...")
            
            # Save full JSON response for inspection
            with open("response_debug.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   - 完整回應已存為: response_debug.json")

            # 處理回傳的檔案
            if result.get('output_base64'):
                output_filename = f"output_{int(time.time())}.png"
                with open(output_filename, "wb") as f:
                    f.write(base64.b64decode(result['output_base64']))
                print(f"   - 結果圖片已存為: {output_filename}")
            else:
                # 如果是影片，通常不回傳 base64 (太大)，只回傳路徑
                print(f"   - 結果路徑 (Server端): {result.get('output_path')}")
                
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
