# NKUST AoV Tool 使用手冊

> **國立高雄科技大學 視覺實驗室**  
> FPGA-aware Computer Vision Pipeline Generator  
> Version: 1.0 | 最後更新：2026-01-13

---

## 📖 系統簡介

**AoV Tool** 是一套專為 FPGA 開發設計的視覺演算法流程生成工具。你只需用**自然語言**描述需求，系統就會自動產生對應的 OpenCV 演算法流程，並即時顯示 **FPGA 硬體資源估算**。

### ✨ 核心特色
- 🤖 **AI 驅動**：支援自然語言轉演算法（目前使用智慧關鍵字匹配）
- ⚡ **FPGA 感知**：每個節點顯示預估時脈、資源消耗
- 📊 **視覺化流程圖**：Graphviz 自動生成，邊上顯示 CLK
- 🖼️ **即時預覽**：調整參數立即看到影像處理結果
- 💾 **知識傳承**：專案可匯出/匯入，學長姐經驗不流失

---

## 🚀 快速開始

### 環境需求
- Python 3.8 或更高版本
- Windows / macOS / Linux

### 安裝步驟

#### 1. 安裝系統依賴（僅 Windows 需要）
```bash
# 如果你使用 Chocolatey
choco install graphviz

# 或者從官網下載安裝：https://graphviz.org/download/
```

#### 2. 安裝 Python 套件
```bash
# 進入專案目錄
cd E:\LAB_DATA\ORB\experiments\05_AoV_Tool

# （推薦）啟動虛擬環境
.venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

#### 3. 啟動應用程式
```bash
# 使用虛擬環境
.venv\Scripts\python.exe -m streamlit run aov_app.py

# 或者直接執行（如果 streamlit 在 PATH 中）
streamlit run aov_app.py
```

#### 4. 開啟瀏覽器
系統會自動開啟瀏覽器，若沒有請手動訪問：
```
http://localhost:8501
```

---

## 📋 使用流程（4 步驟）

### 步驟 1️⃣：上傳影像 & 描述需求
1. 在左側「上傳影像」區塊選擇一張圖片
2. 在「描述你的需求」輸入框填寫需求，例如：
   - `偵測硬幣`
   - `找出邊緣`
   - `降噪處理`
3. 點擊 **🚀 產生 Pipeline**

### 步驟 2️⃣：調整參數與 CLK
1. 系統會自動產生演算法流程
2. 在「調整參數」區塊，每個節點都可展開編輯
3. **重點**：修改「CLK 數值」可調整 FPGA 時脈估計
4. 右側流程圖會即時更新

### 步驟 3️⃣：執行 & 查看結果
1. 點擊 **▶️ 執行 Pipeline**
2. 右側「處理結果預覽」會顯示處理後的影像
3. 可點擊「下載結果」儲存影像

### 步驟 4️⃣：匯出專案（傳承學習）
1. 在「專案存取」區塊，展開「💾 匯出專案」
2. 填寫：
   - **作者姓名**：例如 `C111XXXX 王小明`
   - **實驗備註**：記錄此專案的目的與心得
   - **專案名稱**：例如 `硬幣辨識_V1`
3. 點擊「下載專案檔」
4. 檔案會儲存為 `.json` 格式，可分享給學弟妹

---

## 📂 檔案結構說明

### 核心檔案
```
05_AoV_Tool/
├── aov_app.py              # Streamlit 主應用程式
├── logic_engine.py         # LLM 整合與 Pipeline 生成引擎
├── processor.py            # 影像執行引擎（OpenCV 操作）
├── project_manager.py      # 專案匯入/匯出管理器
├── library_manager.py      # 演算法資料庫管理器
├── tech_lib.json           # 演算法資料庫（FPGA 約束）
├── requirements.txt        # Python 依賴清單
└── sample_coin_detection.json  # 範例專案檔
```

### 重要資料檔案

#### `tech_lib.json`
**用途：** 演算法資料庫（Single Source of Truth）

**內容：**
- 官方演算法（GaussianBlur, Canny, Dilate...）
- 學生貢獻演算法（CLAHE, 自訂演算法...）
- 每個演算法的：
  - FPGA 時脈估計（`estimated_clk`）
  - 資源消耗（`resource_usage`）
  - 參數定義（`parameters`）

**如何新增演算法：**
```python
from library_manager import LibraryManager

manager = LibraryManager()
manager.add_algorithm("my_algorithm", {
    "name": "我的演算法",
    "category": "preprocessing",
    "fpga_constraints": {...}
}, "contributed")
```

#### `nkust_project_*.json`
**用途：** 專案存檔（匯出後的檔案）

**內容：**
- `meta`：作者、時間、備註
- `pipeline`：完整的節點列表（含調整後的參數與 CLK）
- `hardware_summary`：總時脈、複雜度評估

**如何使用：**
1. 在「專案存取」→「匯入專案」上傳此檔案
2. 系統會自動載入流程圖並可執行

---

## 🛠️ 疑難排解

### ❌ 問題 1：流程圖無法顯示
**症狀：** 右側「演算法流程圖」區域空白

**原因：** 系統未安裝 Graphviz

**解決方案（Windows）：**
```bash
# 方法 1: 使用 Chocolatey
choco install graphviz

# 方法 2: 手動安裝
# 1. 前往 https://graphviz.org/download/
# 2. 下載 Windows 安裝檔
# 3. 安裝後，將 Graphviz/bin 加入系統 PATH
```

**驗證是否安裝成功：**
```bash
dot -version
# 應顯示版本號，例如：dot - graphviz version 2.50.0
```

### ❌ 問題 2：執行 Pipeline 時出現錯誤
**症狀：** 點擊「執行 Pipeline」後顯示紅色錯誤訊息

**可能原因：**
- 影像格式不支援
- 某個演算法參數設定錯誤
- OpenCV 未正確安裝

**解決方案：**
1. 確認圖片格式為 JPG/PNG/BMP
2. 檢查終端機的詳細錯誤訊息
3. 重新安裝 opencv-python：
   ```bash
   pip uninstall opencv-python
   pip install opencv-python
   ```

### ❌ 問題 3：匯入專案失敗
**症狀：** 上傳 JSON 檔案後顯示「載入失敗」

**可能原因：**
- JSON 格式錯誤
- 檔案不是 AoV Tool 匯出的

**解決方案：**
1. 確認檔案是由「匯出專案」功能產生的
2. 用文字編輯器打開檔案，檢查是否包含 `meta` 和 `pipeline` 欄位
3. 若要手動編輯，請參考 `sample_coin_detection.json`

### ❌ 問題 4：繁體中文顯示亂碼（Windows）
**症狀：** UI 或 JSON 檔案中的中文變成問號

**解決方案：**
1. 確認系統地區設定為「台灣」
2. 在 PowerShell 中設定正確編碼：
   ```powershell
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   ```

---

## 📚 進階功能

### 匯入範例專案
系統附帶一個範例專案 `sample_coin_detection.json`，展示經典的邊緣檢測流程。

**使用方法：**
1. 開啟「專案存取」→「匯入專案」
2. 上傳 `sample_coin_detection.json`
3. 點擊「載入此專案」
4. 上傳一張影像並執行

### 自訂演算法
若內建演算法不足，可自行新增：

1. **更新資料庫**（`tech_lib.json`）
2. **實作執行邏輯**（`processor.py`）
3. **重啟應用程式**

詳細步驟請參考 `walkthrough.md`。

---

## 🎓 學習資源

### 推薦學習路徑
1. **第一次使用**：匯入 `sample_coin_detection.json` 並執行
2. **練習調參**：修改 Canny 的 threshold，觀察結果變化
3. **自己建流程**：用自然語言生成新的 Pipeline
4. **匯出分享**：填寫完整備註，匯出給學弟妹

### 相關文件
- `walkthrough.md`：完整系統架構說明
- `tech_lib.json`：演算法資料庫參考
- `.cursorrules/agents_manifest.json`：團隊協作架構

---

## 👥 開發團隊

**實驗室：** 國立高雄科技大學 視覺實驗室  
**開發團隊：** 10 位 AI Agents 協作完成
- Captain_Vision (專案經理)
- Streamlit_Wizard (前端工程師)
- Pipeline_Core (後端工程師)
- Prompt_Master (LLM 整合)
- Verilog_Guru (FPGA 專家)
- Legacy_Keeper (資料庫管理)
- Preview_Artist (影像處理)
- Bridge_Builder (系統整合)
- Bug_Hunter (測試工程師)
- Professor_Doc (技術文件)

---

## 📞 技術支援

若遇到問題，請依序嘗試：
1. 查看本手冊的「疑難排解」章節
2. 檢查終端機的錯誤訊息
3. 查閱 `walkthrough.md` 取得更詳細的技術說明
4. 聯繫實驗室助教或指導教授

---

## 📝 版本資訊

**目前版本：** 1.0  
**發布日期：** 2026-01-13  
**授權方式：** MIT License（僅供 NKUST 內部使用）

---

**祝你使用愉快！** 🎉
