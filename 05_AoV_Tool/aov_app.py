
"""
NKUST AoV Tool - Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
import json

from logic_engine import LogicEngine
from processor import ImageProcessor
from library_manager import LibraryManager
from project_manager import ProjectManager
from import_parser import ImportParser
from code_generator import CodeGenerator
from templates import get_default_templates

# Import Refactored Components
from components.node_editor import render_parameter_editor
from components.sidebar import render_sidebar
from components.visualizer import render_pipeline_graph
from components.style import apply_custom_style, render_hero_section

# ==================== Main App ====================

st.set_page_config(
    page_title="NKUST AoV Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Theme
apply_custom_style()

# 1. Init Library Manager (Global Cache)
@st.cache_resource
def init_lib_manager():
    return LibraryManager()

# 2. Init Session State (User Scope)
if 'engine' not in st.session_state:
    st.session_state.engine = LogicEngine(lib_manager=init_lib_manager())

if 'processor' not in st.session_state:
    st.session_state.processor = ImageProcessor()

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Aliases
engine = st.session_state.engine
processor = st.session_state.processor

# st.title("NKUST AoV 演算法視覺化工具") # Replaced by Hero Section
render_hero_section()

col_left, col_right = st.columns([1, 1.5], gap="large") # Added gap

with col_left:
    st.markdown("### 🛠️ Pipeline 編輯器")
    
    with st.container():
        st.caption("1. 上傳影像")
        uploaded_file = st.file_uploader(
            "選擇影像/影片檔案",
            type=['jpg', 'jpeg', 'png', 'bmp', 'mp4', 'avi', 'mov'],
            label_visibility="collapsed"
        )

    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type in ['mp4', 'avi', 'mov']:
            # Video Handling
            st.session_state.is_video = True
            st.session_state.video_path = f"temp_input.{file_type}"
            with open(st.session_state.video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.video(st.session_state.video_path)
            st.info(f"已載入影片: {uploaded_file.name}")
            
            # Use first frame for preview/FPGA calculation
            cap = cv2.VideoCapture(st.session_state.video_path)
            ret, first_frame = cap.read()
            if ret:
                st.session_state.uploaded_image = first_frame # Keep for preview context
                h, w = first_frame.shape[:2]
                if st.session_state.pipeline:
                    engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
            cap.release()
            
        else:
            # Image Handling (Existing Logic)
            st.session_state.is_video = False
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                st.error("無法解碼影像，請確認檔案格式是否正確。")
            else:
                st.session_state.uploaded_image = img_bgr
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="原始影像")

                # [NEW] Dynamic FPGA Estimation
                if st.session_state.pipeline:
                    h, w = img_bgr.shape[:2]
                    engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                    st.toast(f"FPGA 資源已根據解析度 ({w}x{h}) 更新", icon="⚡")
    
    st.divider()
    
    st.subheader("2. 描述需求")
    user_query = st.text_input(
        "輸入需求描述",
        placeholder="例如：偵測硬幣、找出邊緣、降噪處理"
    )
    
    col_gen1, col_gen2 = st.columns(2)
    
    with col_gen1:
        if st.button("Generate Pipeline", type="primary", use_container_width=True):
            if user_query:
                # Update API Key if provided
                if st.session_state.get('llm_api_key'):
                    engine.prompt_master.api_key = st.session_state.llm_api_key
                    engine.prompt_master.base_url = st.session_state.llm_base_url
                    engine.prompt_master.model = st.session_state.get('llm_model_name', 'gpt-4o')
                    engine.prompt_master.llm_available = True
                
                with st.spinner("Processing..."):
                    result = engine.process_user_query(
                        user_query, 
                        use_mock_llm=st.session_state.get('use_mock_llm', True)
                    )
                    
                    # Handle Error
                    if result.get("error"):
                        st.error(f"AI 生成失敗: {result['error']}")
                        with st.expander("詳細錯誤資訊"):
                            st.code(result.get('reasoning', 'No details'))
                    else:
                        # Success
                        st.session_state.pipeline = result["pipeline"]
                        st.session_state.last_reasoning = result.get("reasoning", "")
                        
                        # Initial calculation if image exists
                        if st.session_state.uploaded_image is not None:
                            h, w = st.session_state.uploaded_image.shape[:2]
                            engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                            
                        st.success(f"已產生 {len(st.session_state.pipeline)} 個節點")
                        st.rerun()
            else:
                st.warning("請先輸入需求描述")
    
    with col_gen2:
        if st.button("Reset", use_container_width=True):
            st.session_state.pipeline = []
            st.session_state.processed_image = None
            st.session_state.last_reasoning = ""
            st.rerun()
    
    st.divider()

    # [NEW] AI Reasoning Display
    if st.session_state.get('last_reasoning'):
        with st.expander("AI Reasoning", expanded=True):
            st.info(st.session_state.last_reasoning)

    # ========== 快速模板 (Templates) ==========
    st.subheader("2.5 快速模板")
    templates = get_default_templates()
    selected_template = st.selectbox("選擇預設場景", list(templates.keys()))
    
    if st.button("Load Template", use_container_width=True):
        raw_nodes = templates[selected_template]
        hydrated_pipeline = []
        
        # Hydrate nodes with library data
        for idx, raw_node in enumerate(raw_nodes):
            func_name = raw_node['function']
            # Try to find in library
            algo_data = engine.lib_manager.get_algorithm(func_name, 'official')
            if not algo_data:
                algo_data = engine.lib_manager.get_algorithm(func_name, 'contributed')
            
            node_to_add = raw_node.copy()
            if algo_data:
                node_to_add['category'] = algo_data.get('category', 'unknown')
                node_to_add['description'] = algo_data.get('description', '')
                node_to_add['fpga_constraints'] = algo_data.get('fpga_constraints', {}).copy() # Copy!
            else:
                node_to_add['fpga_constraints'] = {"estimated_clk": 0, "resource_usage": "Unknown", "latency_type": "Software"}
                
            node_to_add['id'] = f"node_{idx}"
            node_to_add['_enabled'] = True
            hydrated_pipeline.append(node_to_add)
            
        st.session_state.pipeline = hydrated_pipeline
        st.session_state.processed_image = None
        
        # Recalc if image exists
        if st.session_state.uploaded_image is not None:
            h, w = st.session_state.uploaded_image.shape[:2]
            engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)

        st.success(f"已載入: {selected_template}")
        st.rerun()

    st.divider()
    
    if st.session_state.pipeline:
        st.subheader("3. Pipeline Editor")
        
        # ========== 新增節點功能 ==========
        with st.expander("Add Node", expanded=False):
            all_algos = engine.lib_manager.list_algorithms()
            
            algo_options = {}
            for a in all_algos:
                cn_name = a.get('name_zh', a['name'])
                display_name = f"{a['name']} / {cn_name} ({a['category']})"
                algo_options[display_name] = a
            
            selected_algo_name = st.selectbox("選擇演算法", list(algo_options.keys()), key="new_algo_select")
            insert_position = st.number_input("插入位置", min_value=0, max_value=len(st.session_state.pipeline), value=len(st.session_state.pipeline), key="insert_pos")
            
            if st.button("Add Node", type="primary", use_container_width=True):
                algo_data = algo_options[selected_algo_name]
                algo_id = algo_data['_algo_id']
                
                new_node = {
                    'id': f'node_{insert_position}',
                    'name': algo_data['name'],
                    'function': algo_data.get('opencv_function', algo_id),
                    'category': algo_data['category'],
                    'description': algo_data.get('description', ''),
                    'parameters': algo_data.get('parameters', {}).copy(),
                    'fpga_constraints': algo_data['fpga_constraints'].copy(),
                    'source': 'manual_add',
                    '_enabled': True
                }
                
                st.session_state.pipeline.insert(insert_position, new_node)
                
                for i, n in enumerate(st.session_state.pipeline):
                    n['id'] = f"node_{i}"
                
                if st.session_state.uploaded_image is not None:
                    h, w = st.session_state.uploaded_image.shape[:2]
                    engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                    
                st.success(f"已新增 {algo_data['name']}")
                st.rerun()
        
        for idx, node in enumerate(st.session_state.pipeline):
            node_id = node.get('id', f'node_{idx}')
            node_name = node.get('name', '未知節點')
            fpga = node.get('fpga_constraints', {})
            is_enabled = node.get('_enabled', True)
            
            with st.expander(f"[{idx}] {node_name}" + (" (Disabled)" if not is_enabled else ""), expanded=False):
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                
                with col_btn1:
                    if st.button("Up", key=f"up_{idx}", disabled=(idx == 0)):
                        st.session_state.pipeline[idx], st.session_state.pipeline[idx-1] = \
                            st.session_state.pipeline[idx-1], st.session_state.pipeline[idx]
                        for i, n in enumerate(st.session_state.pipeline):
                            n['id'] = f"node_{i}"
                        st.rerun()
                
                with col_btn2:
                    if st.button("Down", key=f"down_{idx}", disabled=(idx == len(st.session_state.pipeline)-1)):
                        st.session_state.pipeline[idx], st.session_state.pipeline[idx+1] = \
                            st.session_state.pipeline[idx+1], st.session_state.pipeline[idx]
                        for i, n in enumerate(st.session_state.pipeline):
                            n['id'] = f"node_{i}"
                        st.rerun()
                
                with col_btn3:
                    move_to = st.number_input("Move to", min_value=0, max_value=len(st.session_state.pipeline)-1, value=idx, key=f"moveto_{idx}", label_visibility="collapsed")
                    if move_to != idx and st.button("GO", key=f"move_{idx}"):
                        node_to_move = st.session_state.pipeline.pop(idx)
                        st.session_state.pipeline.insert(move_to, node_to_move)
                        for i, n in enumerate(st.session_state.pipeline):
                            n['id'] = f"node_{i}"
                        st.rerun()
                
                with col_btn4:
                    skip_label = "Enable" if not is_enabled else "Skip"
                    if st.button(skip_label, key=f"skip_{idx}"):
                        st.session_state.pipeline[idx]['_enabled'] = not is_enabled
                        st.rerun()
                
                with col_btn5:
                    if st.button("Delete", key=f"del_{idx}"):
                        st.session_state.pipeline.pop(idx)
                        for i, n in enumerate(st.session_state.pipeline):
                            n['id'] = f"node_{i}"
                        st.rerun()
                
                st.divider()
                
                st.caption(f"類別: {node.get('category', 'N/A')}")
                st.caption(f"函數: {node.get('function', 'N/A')}")
                st.caption(f"狀態: {'啟用' if is_enabled else '已停用'}")
                
                st.markdown("**FPGA 時脈估計**")
                
                current_clk = fpga.get('estimated_clk', 0)
                new_clk = st.number_input(
                    "CLK (Auto-calculated)",
                    min_value=0,
                    value=current_clk,
                    disabled=True, # Lock explicit edit if we use formulas, but maybe allow override?
                    key=f"clk_{node_id}"
                )
                if 'clk_formula' in fpga:
                    st.caption(f"Formula: `{fpga['clk_formula']}`")
                
                st.caption(f"資源: {fpga.get('resource_usage', 'Unknown')}")
                st.caption(f"延遲: {fpga.get('latency_type', 'Unknown')}")
                
                # Render Parameters using Component
                render_parameter_editor(node, idx, node_id)
                
                if '_warning' in node:
                    st.warning(node['_warning'])
    
    st.divider()
    
    st.subheader("4. Project Management")
    
    with st.expander("Export Project", expanded=False):
        st.caption("確保所有調整過的參數都會被保存")
        
        export_author = st.text_input("作者", placeholder="學號/姓名", key="export_author")
        export_notes = st.text_area("備註", placeholder="說明此Pipeline的用途", height=80, key="export_notes")
        export_project_name = st.text_input("專案名稱", placeholder="例如：硬幣偵測_V1", key="export_project_name")
        
        if st.session_state.pipeline:
            json_str = ProjectManager.export_project_to_json(
                st.session_state.pipeline,
                author=export_author or "Unknown",
                notes=export_notes,
                project_name=export_project_name or "Untitled_Project"
            )
            
            filename = ProjectManager.generate_project_filename(export_author or "Unknown")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                    use_container_width=True,
                    type="primary"
                )
            
            with col_dl2:
                if st.button("Save to Folder", use_container_width=True):
                    try:
                        save_path = f"e:/LAB_DATA/ORB/experiments/05_AoV_Tool/{filename}"
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(json_str)
                        st.success(f"已儲存: {filename}")
                    except Exception as e:
                        st.error(f"儲存失敗: {e}")
            
            st.markdown("### Code Generation")
            tab_py, tab_vhdl, tab_verilog = st.tabs(["Python", "VHDL", "Verilog"])
            
            with tab_py:
                py_code = CodeGenerator.generate_python_script(st.session_state.pipeline)
                st.code(py_code, language="python")
                st.download_button("Download Python", py_code, "pipeline.py")
                
            with tab_vhdl:
                vhdl_code = CodeGenerator.generate_vhdl(st.session_state.pipeline)
                st.code(vhdl_code, language="vhdl")
                st.download_button("Download VHDL", vhdl_code, "pipeline.vhd")

            with tab_verilog:
                verilog_code = CodeGenerator.generate_verilog(st.session_state.pipeline)
                st.code(verilog_code, language="verilog")
                st.download_button("Download Verilog", verilog_code, "pipeline.v")
        else:
            st.info("請先產生 Pipeline")
    
    with st.expander("Import Project", expanded=False):
        uploaded_project = st.file_uploader(
            "選擇檔案",
            type=['json', 'dot', 'txt', 'md'],
            key="project_uploader"
        )
        
        if uploaded_project is not None:
            try:
                json_content = uploaded_project.read().decode('utf-8')
                file_extension = uploaded_project.name.split('.')[-1].lower()
                
                if file_extension == 'json':
                    result = ProjectManager.load_project_from_json(json_content)
                    if result["success"]:
                        if st.button("Load Project", type="primary", use_container_width=True, key="load_project_btn"):
                            st.session_state.pipeline = result["pipeline"]
                            st.session_state.processed_image = None
                            
                            if st.session_state.uploaded_image is not None:
                                h, w = st.session_state.uploaded_image.shape[:2]
                                engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                                
                            st.success("載入成功")
                            st.rerun()
                    else:
                        st.error(f"失敗: {result['error']}")
                # Simplified for brevity - import parser logic remains same
            except Exception as e:
                st.error(f"錯誤: {str(e)}")
    
    st.divider()
    
    if st.session_state.pipeline and (st.session_state.uploaded_image is not None or st.session_state.get('is_video')):
        if st.button("Run Pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    active_pipeline = [n for n in st.session_state.pipeline if n.get('_enabled', True)]
                    
                    if st.session_state.get('is_video'):
                        # Video Execution
                        output_path = "temp_output.mp4"
                        stats = processor.process_video(
                            st.session_state.video_path,
                            output_path,
                            active_pipeline
                        )
                        st.session_state.processed_video_path = output_path
                        st.success(f"Video Complete: {stats['resolution']} @ {stats['fps']}fps")
                    else:
                        # Image Execution
                        result = processor.execute_pipeline(
                            st.session_state.uploaded_image,
                            active_pipeline
                        )
                        st.session_state.processed_image = result
                        st.success("Complete")
                        
                except Exception as e:
                    st.error(f"Failed: {e}")


with col_right:
    st.header("視覺化預覽")
    
    st.subheader("流程圖")
    # Use Refactored Component
    render_pipeline_graph(st.session_state.pipeline)
    
    st.divider()
    
    st.subheader("處理結果")
    
    # Video Result Display
    if st.session_state.get('is_video') and st.session_state.get('processed_video_path'):
        st.video(st.session_state.processed_video_path)
        with open(st.session_state.processed_video_path, "rb") as f:
            st.download_button("下載影片", f, "result.mp4", "video/mp4", use_container_width=True)
            
    # Image Result Display
    elif st.session_state.processed_image is not None:
        zoom_level = st.slider("縮放", 25, 200, 100, 25, format="%d%%")
        result_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
        
        if zoom_level != 100:
            h, w = result_rgb.shape[:2]
            new_w = int(w * zoom_level / 100)
            new_h = int(h * zoom_level / 100)
            result_rgb = cv2.resize(result_rgb, (new_w, new_h))
        
        st.image(result_rgb, caption=f"({zoom_level}%)")
        
        is_success, buffer = cv2.imencode(".png", st.session_state.processed_image)
        if is_success:
            st.download_button("下載結果", buffer.tobytes(), "result.png", "image/png", use_container_width=True)
    else:
        st.info("執行後顯示結果")

# Render Sidebar using Component
render_sidebar(engine)

if __name__ == "__main__":
    pass
