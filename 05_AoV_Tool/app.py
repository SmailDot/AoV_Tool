import streamlit as st
import cv2
import numpy as np
import json
import graphviz

# --- 1. 技術庫：僅作為參數說明的參考字典 ---
TECH_DESC = {
    "Setup_Specs": "定義硬幣規格：載入選定國家(如 TWD, USD)的硬幣尺寸與顏色資料。",
    "Preprocessing": "預處理：務必先將圖像 Resize 至寬度 640px 以統一比例，再轉灰階、高斯模糊。",
    "Morphological_Filter": "形態學過濾：使用 Morphological Open (開運算) 消除細小的雜訊與電線干擾。",
    "Hough_Circle_Transform": "圓形偵測：在 640px 寬度的基準下，使用針對性的參數偵測圓形。",
    "Color_Analysis": "色彩分析：在圓形區域內計算 HSV/LAB 直方圖以特徵化硬幣。",
    "Size_Normalization": "尺寸歸一化：使用『色彩輔助多重假設檢定』。分別假設最大圓是 50元(金)、10元(銀)等。若最大圓顏色與假設不符(如假設50元卻是銀色)，則剔除該假設。再取 RMSE 最小者。",
    "Classification": "分類邏輯：根據歸一化後的真實尺寸(mm)與色彩判定最接近的標準硬幣面額。",
    "Result_Rendering": "結果繪製：在原圖標記圓框、面額文字與總金額。"
}

# --- 1.1 硬幣資料庫 ---
COIN_DB = {
    "TWD (Taiwan)": {
        "1": {"diameter_mm": 20.0, "color": "copper", "value": 1},
        "5": {"diameter_mm": 22.0, "color": "silver", "value": 5},
        "10": {"diameter_mm": 26.0, "color": "silver", "value": 10},
        "20": {"diameter_mm": 26.85, "color": "gold_silver", "value": 20},
        "50": {"diameter_mm": 28.0, "color": "gold", "value": 50}
    },
    "USD (USA)": {
        "0.01": {"diameter_mm": 19.05, "color": "copper", "value": 0.01},
        "0.05": {"diameter_mm": 21.21, "color": "silver", "value": 0.05},
        "0.10": {"diameter_mm": 17.91, "color": "silver", "value": 0.10},
        "0.25": {"diameter_mm": 24.26, "color": "silver", "value": 0.25}
    },
    "JPY (Japan)": {
        "1": {"diameter_mm": 20.0, "color": "silver", "value": 1},
        "5": {"diameter_mm": 22.0, "color": "gold", "value": 5},
        "10": {"diameter_mm": 23.5, "color": "copper", "value": 10},
        "50": {"diameter_mm": 21.0, "color": "silver", "value": 50},
        "100": {"diameter_mm": 22.6, "color": "silver", "value": 100},
        "500": {"diameter_mm": 26.5, "color": "gold", "value": 500}
    }
}

# --- 2. 硬幣辨識標準模板 (Perfect Template) ---
COIN_TEMPLATE = [
    {
        "id": "S0", "op": "Setup_Specs",
        "params": {
            "coins": {
                "1": {"diameter_mm": 20.0, "color": "copper", "value": 1},
                "5": {"diameter_mm": 22.0, "color": "silver", "value": 5},
                "10": {"diameter_mm": 26.0, "color": "silver", "value": 10},
                "20": {"diameter_mm": 26.85, "color": "gold_silver", "value": 20},
                "50": {"diameter_mm": 28.0, "color": "gold", "value": 50}
            },
            "pixel_per_mm_strategy": "max_coin_assumption"
        }
    },
    {
        "id": "S1", "op": "Preprocessing",
        "params": {
            "resize_width": 640,
            "blur_ksize": 7, 
            "method": "median_blur"
        }
    },
    {
        "id": "S2", "op": "Morphological_Filter",
        "params": {
            "op": "open",
            "kernel_size": 15,
            "shape": "ellipse"
        }
    },
    {
        "id": "S3", "op": "Hough_Circle_Transform",
        "params": {
            "dp": 1.2, "minDist": 30,
            "param1": 50, "param2": 44,
            "minRadius": 15, "maxRadius": 60
        }
    },
    {
        "id": "S3", "op": "Color_Analysis",
        "params": {
            "copper_hue_range": [[0, 20], [170, 180]],
            "silver_saturation_max": 60,
            "center_sample_ratio": 0.4
        }
    },
    {
        "id": "S4", "op": "Classification",
        "params": {"tolerance_mm": 2.0}
    },
    {
        "id": "S5", "op": "Result_Rendering",
        "params": {"font_scale": 0.8, "thickness": 2}
    }
]

# --- 3. 核心功能：狀態管理與導入導出 ---
if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = COIN_TEMPLATE

def save_to_json():
    # Dynamic Prompt based on current pipeline
    region_hint = "TWD"
    try:
         region_hint = st.session_state['pipeline'][0]['params'].get('region', "TWD")
    except: pass

    final_output = {
        "metadata": {"author": "NKUST_Vision_Lab", "task": "Coin_Recognition", "currency": region_hint},
        "logic_prompt": f"此程式需要識別 {region_hint} 硬幣。Step 1: 執行 'Preprocessing' (Resize 640px -> Gray -> Gaussian Blur) 與 'Morphological_Filter' (Open, k=15)。 Step 2: 使用 'Hough_Circle_Transform' 偵測所有圓形 (param2=44, minDist=30)。 Step 3: 執行 'Size_Normalization'。最大圓可能是 50元、20元或 10元。請檢查最大圓的中心顏色：若是銀色，則不可能是 50元(金)或 20元(金銀)，應優先假設為 10元。確認假設後，計算 Scale Factor。 Step 4: 根據 Scale Factor 計算每個圓的直徑(mm)並匹配面額。 Step 5: 繪製結果。",
        "steps": st.session_state['pipeline']
    }
    return json.dumps(final_output, indent=4, ensure_ascii=False)

# --- 4. Streamlit UI (2026 Ready) ---
st.set_page_config(layout="wide", page_title="NKUST AoV Architect")

st.title("🛠️ NKUST AoV 演算法劇本編輯器")
st.caption("核心目標：產出完美 JSON，讓 AI 能根據此劇本撰寫出高品質程式碼。")

# Sidebar: 導入與保存
with st.sidebar:
    st.header("💾 劇本管理")
    
    # 導入功能
    uploaded_json = st.file_uploader("導入自定義 AoV JSON", type=['json'])
    if uploaded_json:
        try:
            data = json.load(uploaded_json)
            st.session_state['pipeline'] = data.get('steps', [])
            st.success("自定義 AoV 已成功導入")
        except:
            st.error("JSON 格式不符")

    st.divider()
    
    # Currency Selection
    selected_currency = st.selectbox("🌎 選擇目標貨幣 (Coin Currency)", list(COIN_DB.keys()), index=0)
    
    if st.button("🔄 載入硬幣辨識模板", width='stretch'):
        # Deep Copy Template
        new_pipeline = json.loads(json.dumps(COIN_TEMPLATE))
        # Inject Specs
        new_pipeline[0]['params']['coins'] = COIN_DB[selected_currency]
        # Inject Region Strategy
        new_pipeline[0]['params']['region'] = selected_currency
        
        st.session_state['pipeline'] = new_pipeline
        st.rerun()

# Main Layout
col_aov, col_params = st.columns([1.2, 0.8])

with col_aov:
    st.subheader("📊 AoV 邏輯流程圖")
    if st.session_state['pipeline']:
        dot = graphviz.Digraph()
        dot.attr(rankdir='TB', bgcolor='transparent')
        for i, s in enumerate(st.session_state['pipeline']):
            dot.node(s['id'], f"Node {i}: {s['op']}", shape="box", style="filled,rounded", fillcolor="#E3F2FD")
            if i > 0:
                dot.edge(st.session_state['pipeline'][i-1]['id'], s['id'])
        st.graphviz_chart(dot, width='stretch')
    else:
        st.info("目前無流程，請點選左側載入模板或手動增加節點。")

with col_params:
    st.subheader("⚙️ 節點參數微調")
    for i, s in enumerate(st.session_state['pipeline']):
        with st.expander(f"Step {i}: {s['op']}", expanded=False):
            st.caption(TECH_DESC.get(s['op'], "自定義技術節點"))
            for pk, pv in s['params'].items():
                # 動態型別處理，避免 Streamlit 報錯
                if isinstance(pv, list):
                    s['params'][pk] = json.loads(st.text_input(f"{pk}", value=json.dumps(pv), key=f"{i}_{pk}"))
                elif isinstance(pv, (int, float)):
                    s['params'][pk] = st.number_input(f"{pk}", value=pv, key=f"{i}_{pk}")
                else:
                    s['params'][pk] = st.text_input(f"{pk}", value=str(pv), key=f"{i}_{pk}")

# --- Step 5: JSON 輸出與複製區 ---
st.divider()
st.subheader("📄 最終演算法劇本 (貼給 AI 使用)")

final_recipe = save_to_json()

c_code, c_copy = st.columns([2, 1])
with c_code:
    st.code(final_recipe, language="json")

with c_copy:
    st.write("💡 如何使用這份 JSON？")
    st.markdown("""
    1. 複製左側 JSON 內容。
    2. 開啟 ChatGPT / Claude / GitHub Copilot。
    3. 輸入指令：
    > 『這是我的視覺演算法 AoV 劇本，請根據這份 JSON 邏輯寫出 Python OpenCV 程式碼。請注意劇本中的技術說明與幾何校驗策略，並確保使用 PIL 渲染標籤。』
    """)
    st.download_button("💾 下載 JSON 檔案", final_recipe, file_name="vision_recipe.json", width='stretch')