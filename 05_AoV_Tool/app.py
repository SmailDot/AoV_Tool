import streamlit as st
import graphviz

def render_aov_tool():
    st.title("AoV (Activity on Vertex) 視覺化產生器")
    st.markdown("### 流程圖生成工具")

    # --- 側邊欄：輸入區 ---
    st.sidebar.header("設定節點與關係")
    
    # 1. 定義節點 (Activities)
    nodes_input = st.sidebar.text_area(
        "輸入節點 (用逗號分隔)", 
        "輸入問題, 資料前處理, 模型推論, 格式化輸出, 最終結果"
    )
    nodes = [n.strip() for n in nodes_input.split(',') if n.strip()]

    # 2. 定義連線 (Dependencies)
    edges_input = st.sidebar.text_area(
        "輸入連線關係 (格式: 起點->終點, 每行一個)",
        "輸入問題->資料前處理\n資料前處理->模型推論\n模型推論->格式化輸出\n格式化輸出->最終結果"
    )

    # --- 核心邏輯：繪製圖形 ---
    if nodes:
        # 創建 Graphviz 物件
        # 'digraph' 代表有向圖 (Directed Graph)
        dot = graphviz.Digraph(comment='AoV Chart')
        
        # 設定圖形屬性 (由左至右 LR，或由上至下 TB)
        dot.attr(rankdir='LR') 
        
        # 加入節點
        for node in nodes:
            # shape='box' 比較像流程圖的 Activity
            dot.node(node, node, shape='box', style='filled', fillcolor='lightblue')

        # 加入邊
        try:
            for line in edges_input.split('\n'):
                if '->' in line:
                    parts = line.split('->')
                    src = parts[0].strip()
                    dst = parts[1].strip()
                    if src in nodes and dst in nodes:
                        dot.edge(src, dst)
        except Exception as e:
            st.error(f"解析連線時發生錯誤: {e}")

        # --- 顯示結果 ---
        st.graphviz_chart(dot)
        
        st.info("提示：這是標準的 AoV/DAG 結構，適用於工作流或神經網路架構展示。")
    else:
        st.warning("請先在左側輸入節點。")

if __name__ == "__main__":
    render_aov_tool()