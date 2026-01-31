
"""
NKUST AoV Tool - Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import time

from app.core.logic_engine import LogicEngine
from app.core.processor import ImageProcessor
from app.core.library_manager import LibraryManager
from app.core.project_manager import ProjectManager
from app.core.import_parser import ImportParser
from app.core.code_generator import CodeGenerator
from app.core.templates import get_default_templates

# Import Refactored Components
from components.node_editor import render_parameter_editor
from components.sidebar import render_sidebar
from components.visualizer import render_pipeline_graph
from components.style import apply_custom_style, render_hero_section
from components.knowledge_tree import render_d3_tree

# [NEW] Import AutoTuner & KnowledgeBase
from app.vision.optimizer import AutoTuner
from app.knowledge import get_knowledge_base
from PIL import Image
import numpy as np
import cv2
import time

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

# Initialize Knowledge Base (Lazy Load)
if 'kb' not in st.session_state:
    with st.spinner("載入知識庫 (Knowledge Base)..."):
        st.session_state.kb = get_knowledge_base()

# Aliases
engine = st.session_state.engine
processor = st.session_state.processor
kb = st.session_state.kb

# st.title("NKUST AoV 演算法視覺化工具") # Replaced by Hero Section
render_hero_section()

col_left, col_right = st.columns([1, 1.5], gap="large") # Added gap

with col_left:
    # [Auto-execution] Function to execute pipeline automatically on changes
    def execute_pipeline_auto():
        """Auto-execute pipeline when changes are made"""
        if st.session_state.pipeline and (st.session_state.uploaded_image is not None or st.session_state.get('is_video')):
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
                else:
                    # Image Execution
                    if st.session_state.uploaded_image is not None:
                        result = processor.execute_pipeline(
                            st.session_state.uploaded_image,
                            active_pipeline
                        )
                        st.session_state.processed_image = result
                
                # Show update toast
                st.toast("✅ Pipeline Updated", icon="🔄")
                
            except Exception as e:
                st.toast(f"❌ Pipeline Error: {str(e)[:50]}", icon="⚠️")
    
    # ================= 1. Global Input (Uploader) =================
    st.subheader("1. 影像輸入")
    
    with st.container():
        st.caption("上傳影像/影片")
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
                
                # ================= Auto-Tune (File Upload Mode) =================
                enable_tuning = st.checkbox("啟用目標驅動優化 (Auto-Tune)", value=False)
                
                if enable_tuning:
                    st.info("請上傳一張與原圖大小相同的「目標遮罩 (Ground Truth Mask)」。\n(黑白圖片，白色代表目標區域)")
                    
                    mask_file = st.file_uploader("上傳遮罩圖片", type=['png', 'jpg', 'bmp'], key="mask_uploader")
                    
                    col_preview1, col_preview2 = st.columns(2)
                    with col_preview1:
                        st.image(img_rgb, caption="原始影像")
                        
                    if mask_file is not None:
                        # Load Mask
                        mask_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
                        mask_raw = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
                        
                        if mask_raw is not None:
                            # Binarize and Resize if needed
                            _, mask_bin = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)
                            
                            # Auto-resize mask to match source if needed
                            if mask_bin.shape != img_bgr.shape[:2]:
                                st.warning(f"遮罩尺寸 ({mask_bin.shape[::-1]}) 與原圖 ({img_bgr.shape[1]}x{img_bgr.shape[0]}) 不符，將自動縮放。")
                                mask_bin = cv2.resize(mask_bin, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                            with col_preview2:
                                st.image(mask_bin, caption="目標遮罩 (Target)", clamp=True)
                            
                            # [NEW] Optimization Settings
                            with st.expander("優化設定 (Optimization Settings)", expanded=False):
                                opt_max_iters = st.slider("最大迭代次數 (Max Iterations)", 50, 2000, 500, step=50)
                                opt_time_limit = st.slider("時間限制 (Time Limit, seconds)", 30, 600, 180, step=30)
                                opt_target_score = st.slider("目標準確率 (Target IoU)", 0.5, 0.99, 0.92, step=0.01)

                            if st.session_state.pipeline:
                                if st.button("開始自動優化 (Auto-Tune)", type="primary"):
                                    # Save original pipeline for comparison
                                    original_pipeline = [n.copy() for n in st.session_state.pipeline]
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    with st.spinner("正在執行演化演算法優化參數 (Genetic Algorithm)..."):
                                        # Run Optimizer
                                        tuner = AutoTuner(method='ga')
                                        status_text.text("初始化優化引擎...")
                                        
                                        # Increase limits for better convergence
                                        best_pipeline, best_score = tuner.tune_pipeline(
                                            img_bgr,
                                            mask_bin,
                                            st.session_state.pipeline,
                                            max_iterations=opt_max_iters,
                                            time_limit=opt_time_limit,
                                            target_score=opt_target_score
                                        )
                                        
                                        progress_bar.progress(100)
                                        status_text.text("優化完成！")
                                        
                                        # Update Session
                                        st.session_state.pipeline = best_pipeline
                                        
                                        # [Fix] Force update Streamlit widgets
                                        # Update session_state keys for parameters to reflect new values in UI
                                        for node in best_pipeline:
                                            node_id = node.get('id')
                                            for param_name, param_info in node.get('parameters', {}).items():
                                                key = f"param_{node_id}_{param_name}"
                                                if key in st.session_state:
                                                    val = param_info['default']
                                                    # Handle types to match widget expectations
                                                    if isinstance(val, (list, dict, tuple)):
                                                        # Text Input expects string (JSON)
                                                        try:
                                                            st.session_state[key] = json.dumps(val)
                                                        except:
                                                            st.session_state[key] = str(val)
                                                    elif isinstance(val, (int, float, bool)):
                                                        # Number/Checkbox expect raw values
                                                        st.session_state[key] = val
                                                    else:
                                                        # Fallback to string
                                                        st.session_state[key] = str(val)
                                        
                                        # Force Re-execution for Preview
                                        try:
                                            result = processor.execute_pipeline(st.session_state.uploaded_image, best_pipeline)
                                            st.session_state.processed_image = result
                                        except:
                                            pass
                                        
                                        st.success(f"優化完成！IoU 分數提升至: {best_score:.4f}")
                                        
                                        # Show Diff
                                        with st.expander("參數變更報告", expanded=True):
                                            # [Fix] Handle structure changes (Add/Remove nodes)
                                            # If lengths differ, structural mutation happened.
                                            if len(best_pipeline) != len(original_pipeline):
                                                st.info(f"Pipeline 結構已變更：節點數 {len(original_pipeline)} -> {len(best_pipeline)}")
                                                # Simple list of current nodes
                                                st.markdown("### 新的 Pipeline 結構")
                                                for idx, node in enumerate(best_pipeline):
                                                    st.text(f"{idx}. {node['name']}")
                                            else:
                                                # Same length, check params
                                                for i, node in enumerate(best_pipeline):
                                                    # Safe access in case node structure is different even if length is same
                                                    if i >= len(original_pipeline): break
                                                    
                                                    old_node = original_pipeline[i]
                                                    
                                                    # Check if node name changed (Swap/Replace)
                                                    if node['name'] != old_node['name']:
                                                        st.warning(f"Node {i} Changed: {old_node['name']} -> {node['name']}")
                                                        continue
                                                        
                                                    node_name = node['name']
                                                    changed = False
                                                    diff_msg = []
                                                    
                                                    for param_key, param_info in node.get('parameters', {}).items():
                                                        # Safe access
                                                        if 'parameters' not in old_node or param_key not in old_node['parameters']:
                                                            continue
                                                            
                                                        new_val = param_info['default']
                                                        old_val = old_node['parameters'][param_key]['default']
                                                        
                                                        # Simple equality check
                                                        if new_val != old_val:
                                                            changed = True
                                                            diff_msg.append(f"{param_key}: {old_val} -> {new_val}")
                                                    
                                                    if changed:
                                                        st.markdown(f"**{node_name}**: " + ", ".join(diff_msg))
                                                    else:
                                                        st.caption(f"{node_name}: 無變更")
                                        
                                        st.balloons()
                                        time.sleep(1) # Let user see the balloons
                                        st.rerun()
                            else:
                                st.warning("請先在左側生成 Pipeline 才能進行優化。")
                        else:
                            st.error("無法讀取遮罩圖片。")
                else:
                    st.image(img_rgb, caption="原始影像")

                # [NEW] Dynamic FPGA Estimation
                if st.session_state.pipeline:
                    h, w = img_bgr.shape[:2]
                    engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                    st.toast(f"FPGA 資源已根據解析度 ({w}x{h}) 更新")

    st.divider()

    # ================= 2. Tabs =================
    tab_editor, tab_kb = st.tabs(["Pipeline 編輯", "知識庫"])

    # ----------------- Tab 1: Editor -----------------
    with tab_editor:
        st.subheader("2. 描述需求")
        user_query = st.text_input(
            "輸入需求描述",
            placeholder="例如：偵測硬幣、找出邊緣、降噪處理"
        )
        
        col_gen1, col_gen2 = st.columns(2)
        
        with col_gen1:
            if st.button("Generate Pipeline", type="primary", use_container_width=True):
                if user_query:
                    # [Fix] 先搜尋知識庫查看是否有類似案例
                    with st.spinner("🔍 正在知識庫搜尋類似案例..."):
                        kb_matches = kb.find_similar_cases_by_text(user_query, top_k=3)
                    
                    if kb_matches and kb_matches[0][1] > 0.85:  # 相似度 > 0.85
                        best_case, score = kb_matches[0]
                        st.success(f"✅ 從知識庫找到高相似案例 (相似度: {score:.2f})")
                        st.info(f"📚 案例描述: {best_case.get('description', '未命名')}")
                        
                        # 載入知識庫案例
                        import copy
                        st.session_state.pipeline = copy.deepcopy(best_case['pipeline'])
                        st.session_state.processed_image = None
                        st.session_state.last_reasoning = f"[知識庫推薦] 找到相似案例: {best_case.get('description', '')}"
                        
                        if st.session_state.uploaded_image is not None:
                            h, w = st.session_state.uploaded_image.shape[:2]
                            engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                        
                        st.rerun()
                    else:
                        # 知識庫沒有類似案例，使用 LLM 生成
                        if kb_matches:
                            st.info(f"💡 知識庫找到 {len(kb_matches)} 個案例，但相似度不足，改用 AI 生成...")
                        else:
                            st.info("💡 知識庫無類似案例，使用 AI 生成...")
                        
                        # Update API Key if provided
                        if st.session_state.get('llm_api_key'):
                            engine.prompt_master.api_key = st.session_state.llm_api_key
                            engine.prompt_master.base_url = st.session_state.llm_base_url
                            engine.prompt_master.model = st.session_state.get('llm_model_name', 'gpt-4o')
                            engine.prompt_master.llm_available = True
                        
                        with st.spinner("🤖 AI 正在生成 Pipeline..."):
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
        
        # Show Pipeline Editor section regardless of whether pipeline has nodes
        # This ensures Add Node functionality is always available
        st.subheader("3. Pipeline Editor")
        
        # ========== 新增節點功能 (Always visible) ==========
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
                st.session_state._auto_execute = True
                st.rerun()
        
        # Only show node list if there are nodes
        if st.session_state.pipeline:
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
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("Down", key=f"down_{idx}", disabled=(idx == len(st.session_state.pipeline)-1)):
                            st.session_state.pipeline[idx], st.session_state.pipeline[idx+1] = \
                                st.session_state.pipeline[idx+1], st.session_state.pipeline[idx]
                            for i, n in enumerate(st.session_state.pipeline):
                                n['id'] = f"node_{i}"
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    with col_btn3:
                        move_to = st.number_input("Move to", min_value=0, max_value=len(st.session_state.pipeline)-1, value=idx, key=f"moveto_{idx}", label_visibility="collapsed")
                        if move_to != idx and st.button("GO", key=f"move_{idx}"):
                            node_to_move = st.session_state.pipeline.pop(idx)
                            st.session_state.pipeline.insert(move_to, node_to_move)
                            for i, n in enumerate(st.session_state.pipeline):
                                n['id'] = f"node_{i}"
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    with col_btn4:
                        skip_label = "Enable" if not is_enabled else "Skip"
                        if st.button(skip_label, key=f"skip_{idx}"):
                            st.session_state.pipeline[idx]['_enabled'] = not is_enabled
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    with col_btn5:
                        if st.button("Delete", key=f"del_{idx}"):
                            st.session_state.pipeline.pop(idx)
                            for i, n in enumerate(st.session_state.pipeline):
                                n['id'] = f"node_{i}"
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    # [New] Reset node parameters to default
                    if st.button("↺ 重置", key=f"reset_{idx}", help="將所有參數恢復為預設值"):
                        # Get original parameters from library
                        func_name = node.get('function', node.get('name', ''))
                        algo_data = engine.lib_manager.get_algorithm(func_name, 'official')
                        if not algo_data:
                            algo_data = engine.lib_manager.get_algorithm(func_name, 'contributed')
                        
                        if algo_data and 'parameters' in algo_data:
                            # Reset parameters to library defaults
                            st.session_state.pipeline[idx]['parameters'] = algo_data['parameters'].copy()
                            # Clear cached widget values to force update
                            for param_name in algo_data['parameters'].keys():
                                key = f"param_{node_id}_{param_name}"
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.session_state._auto_execute = True
                            st.toast(f"✅ 節點 '{node_name}' 已重置為預設值", icon="🔄")
                            st.rerun()
                        else:
                            st.warning("無法找到原始預設值")
                    
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
                            import os
                            save_dir = "e:/LAB_DATA/ORB/experiments/05_AoV_Tool"
                            # [Fix] Auto-create directory if not exists
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, filename)
                            with open(save_path, 'w', encoding='utf-8') as f:
                                f.write(json_str)
                            st.success(f"✅ 已儲存: {save_path}")
                            
                            # [NEW] 詢問是否存入知識庫中心
                            if st.session_state.uploaded_image is not None:
                                st.session_state._show_kb_dialog = True
                            else:
                                st.info("💡 若要存入知識庫，請先上傳圖片")
                            
                        except Exception as e:
                            st.error(f"儲存失敗: {e}")
                
                # [NEW] 知識庫存入確認對話框
                if st.session_state.get('_show_kb_dialog'):
                    st.divider()
                    st.markdown("### 📚 存入知識庫中心？")
                    st.caption("將此案例存入知識庫，讓後人也能學習您的成果")
                    
                    kb_description = st.text_input(
                        "案例描述",
                        placeholder="例如：針對強反光的硬幣偵測、低光源下的邊緣檢測...",
                        key="kb_save_description"
                    )
                    
                    col_kb_yes, col_kb_no = st.columns(2)
                    
                    with col_kb_yes:
                        if st.button("✅ 存入知識庫", use_container_width=True, type="primary"):
                            if kb_description:
                                try:
                                    # 保存圖片到 uploads 目錄
                                    os.makedirs("uploads", exist_ok=True)
                                    timestamp = int(time.time())
                                    img_path = f"uploads/case_{timestamp}.jpg"
                                    cv2.imwrite(img_path, st.session_state.uploaded_image)
                                    
                                    # 存入知識庫
                                    kb.add_case(img_path, st.session_state.pipeline, kb_description)
                                    
                                    st.session_state._show_kb_dialog = False
                                    st.success(f"✅ 已成功存入知識庫！描述：{kb_description}")
                                    st.balloons()
                                except Exception as e:
                                    st.error(f"存入知識庫失敗: {e}")
                            else:
                                st.warning("請輸入案例描述，幫助他人理解此案例的用途")
                    
                    with col_kb_no:
                        if st.button("❌ 僅儲存檔案", use_container_width=True):
                            st.session_state._show_kb_dialog = False
                            st.info("已儲存 JSON 檔案，但未存入知識庫")
                
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
                            if st.session_state.uploaded_image is not None:
                                result = processor.execute_pipeline(
                                    st.session_state.uploaded_image,
                                    active_pipeline
                                )
                                st.session_state.processed_image = result
                                st.success("Complete")
                            else:
                                st.error("No image uploaded")
                            
                    except Exception as e:
                        st.error(f"Failed: {e}")
        
        # [Auto-execution] Check if auto-execution is triggered
        if st.session_state.get('_auto_execute'):
            st.session_state._auto_execute = False  # Reset flag
            execute_pipeline_auto()

    # ----------------- Tab 2: Knowledge Base -----------------
    with tab_kb:
        st.subheader("知識庫中心")
        st.caption("在這裡尋找靈感，或貢獻您的解決方案。")
        
        tab_text, tab_img, tab_tree, tab_contribute = st.tabs(["文字搜尋", "以圖搜圖", "知識樹瀏覽", "貢獻案例"])
        
        # Tab 1: Text Search (No Image Required)
        with tab_text:
            search_query = st.text_input("輸入需求關鍵字", placeholder="例如: 偵測硬幣, 去除雜訊...", key="kb_text_search")
            if st.button("搜尋方案", key="btn_text_search"):
                if search_query:
                    with st.spinner(f"正在搜尋 '{search_query}'..."):
                        matches = kb.find_similar_cases_by_text(search_query, top_k=3)
                        
                    if matches:
                        st.success(f"找到 {len(matches)} 個相關案例")
                        for i, (case, score) in enumerate(matches):
                            with st.container():
                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    st.markdown(f"**方案 {chr(65+i)}**: {case.get('description', '未命名')}")
                                    st.caption(f"相似度: {score:.2f}")
                                    st.code(" -> ".join([n['name'] for n in case['pipeline']]), language="text")
                                with c2:
                                    if st.button("載入", key=f"load_text_{i}"):
                                        # [Fix] 使用 deepcopy 確保完整複製 Pipeline 和參數
                                        import copy
                                        st.session_state.pipeline = copy.deepcopy(case['pipeline'])
                                        st.session_state.processed_image = None  # 清除舊結果
                                        
                                        # 重新計算 FPGA 統計（無論是否有圖片）
                                        if st.session_state.pipeline and st.session_state.uploaded_image is not None:
                                            h, w = st.session_state.uploaded_image.shape[:2]
                                            engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                                        
                                        st.success(f"✅ 已載入方案: {case.get('description', '未命名')[:30]}...")
                                        st.toast("Pipeline 已更新", icon="🔄")
                                        st.rerun()
                                st.divider()
                    else:
                        st.info("無相關結果。")
                else:
                    st.warning("請輸入關鍵字")
        
        # Tab 2: Image Search (Needs Image)
        with tab_img:
            if st.session_state.uploaded_image is not None:
                if st.button("分析目前圖片"):
                    matches = kb.find_similar_cases(st.session_state.uploaded_image)
                    if matches:
                        st.success(f"找到 {len(matches)} 個相似案例")
                        for i, (case, score) in enumerate(matches):
                                with st.container():
                                    st.markdown(f"**方案 {chr(65+i)}** ({score:.2f})")
                                    st.caption(case.get('description', ''))
                                    st.code(" -> ".join([n['name'] for n in case['pipeline']]), language="text")
                                    if st.button("套用此方案", key=f"apply_img_{i}"):
                                        # [Fix] 使用 deepcopy 確保完整複製 Pipeline 和參數
                                        import copy
                                        st.session_state.pipeline = copy.deepcopy(case['pipeline'])
                                        st.session_state.processed_image = None  # 清除舊結果
                                        
                                        # 重新計算 FPGA 統計
                                        if st.session_state.pipeline and st.session_state.uploaded_image is not None:
                                            h, w = st.session_state.uploaded_image.shape[:2]
                                            engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                                        
                                        st.success(f"✅ 已套用方案: {case.get('description', '未命名')[:30]}...")
                                        st.toast("Pipeline 已更新", icon="🔄")
                                        st.rerun()
                                    st.divider()
                    else:
                        st.info("無相似案例。")
            else:
                st.info("請先上傳圖片才能使用此功能。")

        # Tab 3: Knowledge Tree (UI Branch Tree)
        with tab_tree:
            st.markdown("### 📂 知識庫分類樹")
            st.caption("依據演算法成分檢視案例")
            
            # Category Translation Map
            CATEGORY_MAP = {
                "detection": "物件偵測 (Detection)",
                "preprocessing": "前處理 (Preprocessing)",
                "edge_detection": "邊緣檢測 (Edge)",
                "filtering": "濾波與降噪 (Filtering)",
                "morphology": "形態學 (Morphology)",
                "geometric": "幾何變換 (Geometric)",
                "thresholding": "二值化 (Thresholding)",
                "color": "色彩處理 (Color)",
                "feature_detection": "特徵提取 (Feature)",
                "arithmetic": "算術運算 (Arithmetic)",
                "enhancement": "影像增強 (Enhancement)",
                "analysis": "影像分析 (Analysis)",
                "shape_detection": "形狀檢測 (Shape)",
                "segmentation": "影像分割 (Segmentation)",
                "motion_analysis": "動態分析 (Motion)",
                "unknown": "未分類 (Unknown)",
                "Other": "其他 (Other)"
            }

            # Build Tree Data
            # Structure: Category -> Algorithm -> Cases
            tree_data = {}
            all_cases = kb.db
            
            if not all_cases:
                st.info("知識庫目前為空。")
            else:
                for case in all_cases:
                    case_id = case.get('id', 'unknown')
                    desc = case.get('description', 'Untitled')
                    pipeline = case.get('pipeline', [])
                    
                    # Extract algorithms used
                    for node in pipeline:
                        func_name = node.get('function')
                        node_name = node.get('name')
                        
                        # Lookup Category in Library Manager
                        # We use engine.lib_manager if available, else infer
                        algo_info = engine.lib_manager.get_algorithm(func_name, 'official')
                        if not algo_info:
                            algo_info = engine.lib_manager.get_algorithm(func_name, 'contributed')
                            
                        raw_category = algo_info.get('category', 'Other') if algo_info else 'Other'
                        
                        # Translate Category
                        category_zh = CATEGORY_MAP.get(raw_category, raw_category.title())

                        if category_zh not in tree_data:
                            tree_data[category_zh] = {}
                        if node_name not in tree_data[category_zh]:
                            tree_data[category_zh][node_name] = []
                        
                        # Avoid duplicates per algo branch
                        exists = any(c['id'] == case_id for c in tree_data[category_zh][node_name])
                        if not exists:
                            tree_data[category_zh][node_name].append({
                                'id': case_id,
                                'desc': desc,
                                'pipeline': pipeline
                            })
                
                # --- Visual D3 Tree ---
                d3_data = {"name": "Knowledge Base", "children": []}
                for cat_name in sorted(tree_data.keys()):
                    cat_node = {"name": cat_name, "children": []}
                    for algo_name in sorted(tree_data[cat_name].keys()):
                        algo_node = {"name": algo_name, "children": []}
                        for case in tree_data[cat_name][algo_name]:
                            # Leaf node
                            algo_node["children"].append({"name": case['desc'][:20]+"..."}) # Truncate for display
                        cat_node["children"].append(algo_node)
                    d3_data["children"].append(cat_node)
                
                # Button to trigger Modal/Dialog
                # Initialize session state for dialog control
                if "show_past_exp_dialog" not in st.session_state:
                    st.session_state.show_past_exp_dialog = False
                
                # Use a unique key for the button to prevent state conflicts
                if st.button("過去經驗 ", use_container_width=True, key="btn_past_exp_dialog"):
                    st.session_state.show_past_exp_dialog = True
                
                # Show dialog independently of button click to maintain tab state
                if st.session_state.show_past_exp_dialog:
                    @st.dialog("過去經驗 (Past Experience Tree)", width="large")
                    def show_tree_dialog():
                        st.caption("互動式探索：點擊節點展開/收合")
                        render_d3_tree(d3_data, height=700)
                    show_tree_dialog()
                    # Reset the flag after dialog is shown
                    st.session_state.show_past_exp_dialog = False

                st.divider()

                # Render List Tree with Delete Button
                for cat_name in sorted(tree_data.keys()):
                    with st.expander(f"📁 {cat_name} ({len(tree_data[cat_name])} 類演算法)", expanded=False):
                        for algo_name in sorted(tree_data[cat_name].keys()):
                            cases = tree_data[cat_name][algo_name]
                            st.markdown(f"**└─ 🧩 {algo_name}** ({len(cases)} 案例)")
                            
                            for case in cases:
                                c1, c2, c3 = st.columns([4, 1, 1])
                                with c1:
                                    st.text(f"   📄 {case['desc']}")
                                with c2:
                                    if st.button("Load", key=f"load_tree_{cat_name}_{algo_name}_{case['id']}"):
                                        st.session_state.pipeline = case['pipeline']
                                        if st.session_state.uploaded_image is not None:
                                            h, w = st.session_state.uploaded_image.shape[:2]
                                            engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                                        st.success("Loaded")
                                        st.rerun()
                                with c3:
                                    # Delete Button
                                    if st.button("🗑️", key=f"del_tree_{cat_name}_{algo_name}_{case['id']}", help="刪除此案例"):
                                        if kb.delete_case(case['id']):
                                            st.success(f"已刪除 {case['id']}")
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.error("刪除失敗")

        # Tab 4: Contribute (Needs Image + Pipeline)
        with tab_contribute:
            if st.session_state.uploaded_image is not None and st.session_state.pipeline:
                desc = st.text_input("案例描述", placeholder="例如: 針對強反光的硬幣偵測", key="contrib_desc")
                if st.button("保存至知識庫"):
                    if desc:
                        # Save logic
                        timestamp = int(time.time())
                        save_path = f"uploads/case_{timestamp}.jpg"
                        try:
                            os.makedirs("uploads", exist_ok=True)
                            cv2.imwrite(save_path, st.session_state.uploaded_image)
                            kb.add_case(save_path, st.session_state.pipeline, desc)
                            st.success("已保存")
                        except Exception as e:
                            st.error(f"保存失敗: {e}")
                    else:
                        st.error("請輸入描述")
            else:
                st.info("請先上傳圖片並建立 Pipeline 才能貢獻。")

with col_right:
    st.header("視覺化預覽")
    
    st.subheader("流程圖")
    # Use Refactored Component
    render_pipeline_graph(st.session_state.pipeline)
    
    st.divider()
    
    st.subheader("處理結果")
    
    # [Comparison] Save current result as reference for comparison
    if st.session_state.processed_image is not None:
        col_save, col_clear = st.columns([1, 1])
        with col_save:
            if st.button("📌 保存為對照", help="將目前結果保存，方便調整參數後對比", use_container_width=True):
                st.session_state.reference_image = st.session_state.processed_image.copy()
                st.toast("✅ 已保存為對照基準", icon="📌")
        with col_clear:
            if st.session_state.get('reference_image') is not None:
                if st.button("🗑️ 清除對照", help="清除已保存的對照結果", use_container_width=True):
                    del st.session_state.reference_image
                    st.toast("🗑️ 對照基準已清除", icon="🗑️")
                    st.rerun()
    
    # Video Result Display
    if st.session_state.get('is_video') and st.session_state.get('processed_video_path'):
        st.video(st.session_state.processed_video_path)
        with open(st.session_state.processed_video_path, "rb") as f:
            st.download_button("下載影片", f, "result.mp4", "video/mp4", use_container_width=True)
            
    # Image Result Display
    elif st.session_state.processed_image is not None:
        # Check if reference image exists for comparison
        has_reference = st.session_state.get('reference_image') is not None
        
        if has_reference:
            # Show comparison view
            st.markdown("**👁️ 對照模式**")
            col_current, col_reference = st.columns(2)
            
            with col_current:
                st.caption("🆕 目前結果")
                zoom_current = st.slider("目前縮放", 25, 200, 100, 25, format="%d%%", key="zoom_current")
                result_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                
                if zoom_current != 100:
                    h, w = result_rgb.shape[:2]
                    new_w = int(w * zoom_current / 100)
                    new_h = int(h * zoom_current / 100)
                    result_rgb = cv2.resize(result_rgb, (new_w, new_h))
                
                st.image(result_rgb, use_container_width=True)
            
            with col_reference:
                st.caption("📌 對照基準")
                zoom_ref = st.slider("對照縮放", 25, 200, 100, 25, format="%d%%", key="zoom_ref")
                ref_rgb = cv2.cvtColor(st.session_state.reference_image, cv2.COLOR_BGR2RGB)
                
                if zoom_ref != 100:
                    h, w = ref_rgb.shape[:2]
                    new_w = int(w * zoom_ref / 100)
                    new_h = int(h * zoom_ref / 100)
                    ref_rgb = cv2.resize(ref_rgb, (new_w, new_h))
                
                st.image(ref_rgb, use_container_width=True)
        else:
            # Normal single view
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
