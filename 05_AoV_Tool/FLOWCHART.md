graph TD
    %% 節點定義
    User([使用者])
    
    subgraph Frontend [前端介面 - Streamlit UI]
        UI_Input[影像上傳 & 文字指令]
        UI_Editor[節點編輯器 UI]
        UI_Viz[Pipeline 視覺化流程圖]
        UI_Results[執行結果預覽]
    end

    subgraph Core_Logic [後端核心]
        Logic[邏輯引擎<br/>指令解析]
        Processor[影像處理器<br/>OpenCV 執行]
        LibMgr[函式庫管理器<br/>演算法分類查詢]
        ProjMgr[專案管理器]
    end

    subgraph Data [資料層]
        TechLib[(tech_lib.json<br/>演算法資料庫)]
        KB_DB[(knowledge_db.json<br/>歷史案例庫)]
        Images[影像檔案]
    end

    subgraph Intelligence [AI 模組]
        AutoTuner[自動調參器<br/>遺傳演算法]
        KB_Engine[知識庫引擎<br/>CLIP + FAISS]
    end

    subgraph KB_Tree [知識庫分類樹 - 分支樹功能]
        Tree_UI[知識樹瀏覽 UI<br/>tab_tree 區塊]
        Tree_Builder[樹狀結構建構器<br/>Category → Algorithm → Case]
        D3_Viz[D3.js 互動樹狀圖<br/>可展開/收合/縮放]
        Tree_Dialog[過去經驗對話框<br/>Modal Dialog]
        Tree_Ops[案例操作<br/>載入/刪除]
    end

    %% 資料流向
    User -->|1. 上傳影像| UI_Input
    User -->|2. 輸入需求描述| UI_Input
    
    %% 生成流程
    UI_Input -->|3. 請求生成 Pipeline| Logic
    Logic -->|4. 解析意圖| TechLib
    Logic -->|5. 回傳 Pipeline JSON| UI_Editor
    
    %% 執行流程
    UI_Editor -->|6. 執行/更新參數| Processor
    Processor -->|7. 讀取演算法實作| TechLib
    Processor -->|8. 處理影像| UI_Results
    
    %% 視覺化流程
    UI_Editor -->|更新圖表數據| UI_Viz
    UI_Viz -.->|渲染| User
    
    %% AI 智慧流程
    User -->|智慧推薦| KB_Engine
    KB_Engine -->|向量搜尋| KB_DB
    KB_Engine -->|相似案例| UI_Editor
    
    User -->|最佳化參數| AutoTuner
    AutoTuner -->|迭代測試| Processor
    AutoTuner -->|最佳參數| UI_Editor
    
    %% 專案管理
    UI_Editor -->|匯出專案| ProjMgr
    ProjMgr -->|儲存 JSON| User
    User -->|匯入 JSON| ProjMgr
    ProjMgr -->|載入 Pipeline| UI_Editor
    
    %% 知識庫分類樹流程（分支樹功能）
    User -->|點擊"過去經驗"按鈕| Tree_Dialog
    Tree_Dialog -->|顯示| D3_Viz
    
    Tree_Builder -->|讀取所有案例| KB_DB
    Tree_Builder -->|查詢演算法分類| LibMgr
    Tree_Builder -->|構建三層結構| D3_Viz
    
    D3_Viz -->|類別層級| Tree_UI
    D3_Viz -->|演算法層級| Tree_UI
    D3_Viz -->|案例層級| Tree_UI
    
    Tree_UI -->|選擇案例| Tree_Ops
    Tree_Ops -->|載入 Pipeline| UI_Editor
    Tree_Ops -->|刪除案例| KB_DB
    Tree_Ops -->|更新索引| KB_Engine

    %% 樣式設定
    classDef ui fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef logic fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef data fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef ai fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef tree fill:#fff8e1,stroke:#f57c00,stroke-width:2px;
    
    class UI_Input,UI_Editor,UI_Viz,UI_Results ui;
    class Logic,Processor,LibMgr,ProjMgr logic;
    class TechLib,KB_DB,Images data;
    class AutoTuner,KB_Engine ai;
    class Tree_UI,Tree_Builder,D3_Viz,Tree_Dialog,Tree_Ops tree;
