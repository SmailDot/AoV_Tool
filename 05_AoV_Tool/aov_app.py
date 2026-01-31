"""
NKUST AoV Tool - Streamlit Application
Professional UI Redesign with 3-Step Workflow
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

# Render Hero Section
render_hero_section()

# ==================== Custom Styling for Professional Layout ====================
st.markdown("""
<style>
/* Step Card Styling */
.step-card {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 4px solid #3b82f6;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.step-card-step2 {
    border-left-color: #10b981;
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
}

.step-card-step3 {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: #3b82f6;
    color: white;
    border-radius: 50%;
    font-weight: bold;
    font-size: 14px;
    margin-right: 10px;
}

.step-number-2 {
    background: #10b981;
}

.step-number-3 {
    background: #f59e0b;
}

/* Result Card Styling */
.result-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    border: 1px solid #e2e8f0;
}

/* Pipeline Card */
.pipeline-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

/* Section Headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
}

/* Compact spacing */
.compact-section {
    margin-top: 10px;
    margin-bottom: 10px;
}

/* Hide default Streamlit elements */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    border-radius: 6px 6px 0 0;
}
</style>
""", unsafe_allow_html=True)

# ==================== Main Layout ====================
col_left, col_right = st.columns([1, 1.4], gap="large")

# ==================== LEFT COLUMN: 3-Step Workflow ====================
with col_left:
    
    # ========== STEP 1: 影像輸入 ==========
    st.markdown("""
    <div class="step-card">
        <div class="section-header">
            <span class="step-number">1</span>
            <span>影像輸入</span>
        </div>
        <p style="color: #64748b; font-size: 0.9rem; margin: -10px 0 15px 24px;">
            上傳您要處理的影像或影片檔案
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        uploaded_file = st.file_uploader(
            "📁 選擇影像/影片檔案",
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
            st.success(f"✅ 已載入影片: {uploaded_file.name}")
            
            # Use first frame for preview/FPGA calculation
            cap = cv2.VideoCapture(st.session_state.video_path)
            ret, first_frame = cap.read()
            if ret:
                st.session_state.uploaded_image = first_frame
                h, w = first_frame.shape[:2]
                if st.session_state.pipeline:
                    engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
            cap.release()
            
        else:
            # Image Handling
            st.session_state.is_video = False
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                st.error("❌ 無法解碼影像，請確認檔案格式是否正確。")
            else:
                st.session_state.uploaded_image = img_bgr
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Show uploaded image preview
                with st.expander("📸 已上傳影像預覽", expanded=True):
                    st.image(img_rgb, use_container_width=True)
                
                # Auto-Tune Section (Collapsible)
                with st.expander("🎯 目標驅動優化 (Auto-Tune)", expanded=False):
                    st.info("上傳與原圖相同大小的「目標遮罩」(黑白圖片，白色代表目標區域)")
                    
                    mask_file = st.file_uploader("上傳遮罩圖片", type=['png', 'jpg', 'bmp'], key="mask_uploader")
                    
                    if mask_file is not None:
                        mask_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
                        mask_raw = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
                        
                        if mask_raw is not None:
                            _, mask_bin = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)
                            
                            if mask_bin.shape != img_bgr.shape[:2]:
                                st.warning(f"遮罩尺寸不符，將自動縮放至 {img_bgr.shape[1]}x{img_bgr.shape[0]}")
                                mask_bin = cv2.resize(mask_bin, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(img_rgb, caption="原始影像")
                            with col2:
                                st.image(mask_bin, caption="目標遮罩", clamp=True)
                            
                            # Optimization Settings
                            opt_max_iters = st.slider("最大迭代次數", 50, 2000, 500, step=50)
                            opt_time_limit = st.slider("時間限制 (秒)", 30, 600, 180, step=30)
                            opt_target_score = st.slider("目標 IoU", 0.5, 0.99, 0.92, step=0.01)

                            if st.session_state.pipeline:
                                if st.button("🚀 開始自動優化", type="primary", use_container_width=True):
                                    original_pipeline = [n.copy() for n in st.session_state.pipeline]
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    with st.spinner("正在執行演化演算法優化參數..."):
                                        tuner = AutoTuner(method='ga')
                                        status_text.text("初始化優化引擎...")
                                        
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
                                        
                                        st.session_state.pipeline = best_pipeline
                                        
                                        # Force update Streamlit widgets
                                        for node in best_pipeline:
                                            node_id = node.get('id')
                                            for param_name, param_info in node.get('parameters', {}).items():
                                                key = f"param_{node_id}_{param_name}"
                                                if key in st.session_state:
                                                    val = param_info['default']
                                                    if isinstance(val, (list, dict, tuple)):
                                                        try:
                                                            st.session_state[key] = json.dumps(val)
                                                        except:
                                                            st.session_state[key] = str(val)
                                                    elif isinstance(val, (int, float, bool)):
                                                        st.session_state[key] = val
                                                    else:
                                                        st.session_state[key] = str(val)
                                        
                                        # Force Re-execution for Preview
                                        try:
                                            result = processor.execute_pipeline(st.session_state.uploaded_image, best_pipeline)
                                            st.session_state.processed_image = result
                                        except:
                                            pass
                                        
                                        st.success(f"✅ 優化完成！IoU 分數提升至: {best_score:.4f}")
                                        st.balloons()
                                        time.sleep(1)
                                        st.rerun()
                            else:
                                st.warning("請先生成 Pipeline 才能進行優化。")
                        else:
                            st.error("無法讀取遮罩圖片。")
                
                # Dynamic FPGA Estimation
                if st.session_state.pipeline:
                    h, w = img_bgr.shape[:2]
                    engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                    st.toast(f"📊 FPGA 資源已根據解析度 ({w}x{h}) 更新")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== STEP 2: Pipeline 建構 ==========
    st.markdown("""
    <div class="step-card step-card-step2">
        <div class="section-header">
            <span class="step-number step-number-2">2</span>
            <span>Pipeline 建構</span>
        </div>
        <p style="color: #64748b; font-size: 0.9rem; margin: -10px 0 15px 24px;">
            使用 AI 生成或手動建立影像處理流程
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Query Section
    with st.container():
        st.markdown("**🤖 AI 需求描述**")
        user_query = st.text_input(
            "描述您的影像處理需求",
            placeholder="例如：偵測硬幣、找出邊緣、降噪處理...",
            label_visibility="collapsed"
        )
        
        col_gen1, col_gen2 = st.columns([3, 1])
        
        with col_gen1:
            if st.button("✨ Generate Pipeline", type="primary", use_container_width=True):
                if user_query:
                    # Search knowledge base first
                    with st.spinner("🔍 正在知識庫搜尋類似案例..."):
                        kb_matches = kb.find_similar_cases_by_text(user_query, top_k=3)
                    
                    if kb_matches and kb_matches[0][1] > 0.85:
                        best_case, score = kb_matches[0]
                        st.success(f"✅ 從知識庫找到高相似案例 (相似度: {score:.2f})")
                        st.info(f"📚 案例描述: {best_case.get('description', '未命名')}")
                        
                        import copy
                        st.session_state.pipeline = copy.deepcopy(best_case['pipeline'])
                        st.session_state.processed_image = None
                        st.session_state.last_reasoning = f"[知識庫推薦] 找到相似案例: {best_case.get('description', '')}"
                        
                        if st.session_state.uploaded_image is not None:
                            h, w = st.session_state.uploaded_image.shape[:2]
                            engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                        
                        st.rerun()
                    else:
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
                        
                        if result.get("error"):
                            st.error(f"❌ AI 生成失敗: {result['error']}")
                            with st.expander("詳細錯誤資訊"):
                                st.code(result.get('reasoning', 'No details'))
                        else:
                            st.session_state.pipeline = result["pipeline"]
                            st.session_state.last_reasoning = result.get("reasoning", "")
                            
                            if st.session_state.uploaded_image is not None:
                                h, w = st.session_state.uploaded_image.shape[:2]
                                engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                            
                            st.success(f"✅ 已產生 {len(st.session_state.pipeline)} 個節點")
                            st.rerun()
                else:
                    st.warning("請先輸入需求描述")
        
        with col_gen2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.pipeline = []
                st.session_state.processed_image = None
                st.session_state.last_reasoning = ""
                st.rerun()
    
    # AI Reasoning Display
    if st.session_state.get('last_reasoning'):
        with st.expander("🧠 AI Reasoning", expanded=False):
            st.info(st.session_state.last_reasoning)
    
    st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    
    # Quick Templates Section
    with st.container():
        st.markdown("**📋 快速模板**")
        templates = get_default_templates()
        
        col_temp1, col_temp2 = st.columns([3, 1])
        with col_temp1:
            selected_template = st.selectbox("選擇預設場景", list(templates.keys()), label_visibility="collapsed")
        with col_temp2:
            if st.button("📂 Load", use_container_width=True):
                raw_nodes = templates[selected_template]
                hydrated_pipeline = []
                
                for idx, raw_node in enumerate(raw_nodes):
                    func_name = raw_node['function']
                    algo_data = engine.lib_manager.get_algorithm(func_name, 'official')
                    if not algo_data:
                        algo_data = engine.lib_manager.get_algorithm(func_name, 'contributed')
                    
                    node_to_add = raw_node.copy()
                    if algo_data:
                        node_to_add['category'] = algo_data.get('category', 'unknown')
                        node_to_add['description'] = algo_data.get('description', '')
                        node_to_add['fpga_constraints'] = algo_data.get('fpga_constraints', {}).copy()
                    else:
                        node_to_add['fpga_constraints'] = {"estimated_clk": 0, "resource_usage": "Unknown", "latency_type": "Software"}
                    
                    node_to_add['id'] = f"node_{idx}"
                    node_to_add['_enabled'] = True
                    hydrated_pipeline.append(node_to_add)
                
                st.session_state.pipeline = hydrated_pipeline
                st.session_state.processed_image = None
                
                if st.session_state.uploaded_image is not None:
                    h, w = st.session_state.uploaded_image.shape[:2]
                    engine.verilog_guru.recalculate_pipeline_stats(st.session_state.pipeline, w, h)
                
                st.success(f"✅ 已載入: {selected_template}")
                st.rerun()
    
    st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    
    # Pipeline Editor Section
    with st.container():
        st.markdown("**🛠️ Pipeline 編輯器**")
        
        # Add Node Section
        with st.expander("➕ 新增節點", expanded=False):
            all_algos = engine.lib_manager.list_algorithms()
            
            algo_options = {}
            for a in all_algos:
                cn_name = a.get('name_zh', a['name'])
                display_name = f"{a['name']} / {cn_name} ({a['category']})"
                algo_options[display_name] = a
            
            col_algo1, col_algo2 = st.columns([3, 1])
            with col_algo1:
                selected_algo_name = st.selectbox("選擇演算法", list(algo_options.keys()), key="new_algo_select")
            with col_algo2:
                insert_position = st.number_input("插入位置", min_value=0, max_value=len(st.session_state.pipeline), value=len(st.session_state.pipeline), key="insert_pos")
            
            if st.button("➕ Add Node", type="primary", use_container_width=True):
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
                
                st.success(f"✅ 已新增 {algo_data['name']}")
                st.session_state._auto_execute = True
                st.rerun()
        
        # Node List
        if st.session_state.pipeline:
            st.markdown(f"<p style='color: #64748b; font-size: 0.85rem;'>共 {len(st.session_state.pipeline)} 個節點</p>", unsafe_allow_html=True)
            
            for idx, node in enumerate(st.session_state.pipeline):
                node_id = node.get('id', f'node_{idx}')
                node_name = node.get('name', '未知節點')
                fpga = node.get('fpga_constraints', {})
                is_enabled = node.get('_enabled', True)
                
                with st.expander(f"[{idx}] {node_name}" + (" ⛔" if not is_enabled else ""), expanded=False):
                    # Action Buttons
                    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                    
                    with col_btn1:
                        if st.button("⬆️", key=f"up_{idx}", disabled=(idx == 0), help="上移"):
                            st.session_state.pipeline[idx], st.session_state.pipeline[idx-1] = \
                                st.session_state.pipeline[idx-1], st.session_state.pipeline[idx]
                            for i, n in enumerate(st.session_state.pipeline):
                                n['id'] = f"node_{i}"
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("⬇️", key=f"down_{idx}", disabled=(idx == len(st.session_state.pipeline)-1), help="下移"):
                            st.session_state.pipeline[idx], st.session_state.pipeline[idx+1] = \
                                st.session_state.pipeline[idx+1], st.session_state.pipeline[idx]
                            for i, n in enumerate(st.session_state.pipeline):
                                n['id'] = f"node_{i}"
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    with col_btn3:
                        skip_label = "✓" if not is_enabled else "⏸"
                        if st.button(skip_label, key=f"skip_{idx}", help="啟用/停用"):
                            st.session_state.pipeline[idx]['_enabled'] = not is_enabled
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    with col_btn4:
                        if st.button("🗑️", key=f"del_{idx}", help="刪除"):
                            st.session_state.pipeline.pop(idx)
                            for i, n in enumerate(st.session_state.pipeline):
                                n['id'] = f"node_{i}"
                            st.session_state._auto_execute = True
                            st.rerun()
                    
                    with col_btn5:
                        if st.button("↺", key=f"reset_{idx}", help="重置參數"):
                            func_name = node.get('function', node.get('name', ''))
                            algo_data = engine.lib_manager.get_algorithm(func_name, 'official')
                            if not algo_data:
                                algo_data = engine.lib_manager.get_algorithm(func_name, 'contributed')
                            
                            if algo_data and 'parameters' in algo_data:
                                st.session_state.pipeline[idx]['parameters'] = algo_data['parameters'].copy()
                                for param_name in algo_data['parameters'].keys():
                                    key = f"param_{node_id}_{param_name}"
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.session_state._auto_execute = True
                                st.toast(f"✅ 已重置為預設值", icon="🔄")
                                st.rerun()
                            else:
                                st.warning("無法找到原始預設值")
                    
                    # Node Info
                    st.caption(f"類別: {node.get('category', 'N/A')} | 函數: {node.get('function', 'N/A')}")
                    
                    # FPGA Info
                    st.markdown("**FPGA 時脈估計**")
                    current_clk = fpga.get('estimated_clk', 0)
                    st.number_input(
                        "CLK (MHz)",
                        min_value=0,
                        value=current_clk,
                        disabled=True,
                        key=f"clk_{node_id}"
                    )
                    if 'clk_formula' in fpga:
                        st.caption(f"公式: `{fpga['clk_formula']}`")
                    st.caption(f"資源: {fpga.get('resource_usage', 'Unknown')} | 延遲: {fpga.get('latency_type', 'Unknown')}")
                    
                    # Parameter Editor
                    render_parameter_editor(node, idx, node_id)
                    
                    if '_warning' in node:
                        st.warning(node['_warning'])
        else:
            st.info("👆 請先使用 AI 生成或手動新增節點")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== STEP 3: 執行控制 ==========
    st.markdown("""
    <div class="step-card step-card-step3">
        <div class="section-header">
            <span class="step-number step-number-3">3</span>
            <span>執行處理</span>
        </div>
        <p style="color: #64748b; font-size: 0.9rem; margin: -10px 0 15px 24px;">
            執行 Pipeline 並查看處理結果
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        if st.session_state.pipeline and (st.session_state.uploaded_image is not None or st.session_state.get('is_video')):
            if st.button("▶️ Run Pipeline", type="primary", use_container_width=True):
                with st.spinner("🔄 處理中..."):
                    try:
                        active_pipeline = [n for n in st.session_state.pipeline if n.get('_enabled', True)]
                        
                        if st.session_state.get('is_video'):
                            output_path = "temp_output.mp4"
                            stats = processor.process_video(
                                st.session_state.video_path,
                                output_path,
                                active_pipeline
                            )
                            st.session_state.processed_video_path = output_path
                            st.success(f"✅ 影片處理完成: {stats['resolution']} @ {stats['fps']}fps")
                        else:
                            if st.session_state.uploaded_image is not None:
                                result = processor.execute_pipeline(
                                    st.session_state.uploaded_image,
                                    active_pipeline
                                )
                                st.session_state.processed_image = result
                                st.success("✅ 處理完成")
                            else:
                                st.error("❌ 請先上傳影像")
                        
                    except Exception as e:
                        st.error(f"❌ 處理失敗: {e}")
        else:
            if not st.session_state.pipeline:
                st.info("⏳ 請先建立 Pipeline (Step 2)")
            elif st.session_state.uploaded_image is None:
                st.info("⏳ 請先上傳影像 (Step 1)")
    
    # Auto-execution check
    if st.session_state.get('_auto_execute'):
        st.session_state._auto_execute = False
        execute_pipeline_auto()

# ==================== RIGHT COLUMN: Visualization & Results ====================
with col_right:
    
    # Pipeline Visualization Card
    st.markdown("""
    <div class="pipeline-card">
        <div class="section-header">
            <span>🔄 Pipeline 流程圖</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    render_pipeline_graph(st.session_state.pipeline)
    
    # Results Card
    st.markdown("""
    <div class="result-card">
        <div class="section-header">
            <span>🖼️ 處理結果</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison Controls
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
            st.download_button("⬇️ 下載影片", f, "result.mp4", "video/mp4", use_container_width=True)
    
    # Image Result Display
    elif st.session_state.processed_image is not None:
        has_reference = st.session_state.get('reference_image') is not None
        
        if has_reference:
            # Comparison View
            st.markdown("**👁️ 對照模式**")
            col_current, col_reference = st.columns(2)
            
            with col_current:
                st.caption("🆕 目前結果")
                zoom_current = st.slider("縮放", 25, 200, 100, 25, format="%d%%", key="zoom_current")
                result_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                
                if zoom_current != 100:
                    h, w = result_rgb.shape[:2]
                    new_w = int(w * zoom_current / 100)
                    new_h = int(h * zoom_current / 100)
                    result_rgb = cv2.resize(result_rgb, (new_w, new_h))
                
                st.image(result_rgb, use_container_width=True)
            
            with col_reference:
                st.caption("📌 對照基準")
                zoom_ref = st.slider("縮放", 25, 200, 100, 25, format="%d%%", key="zoom_ref")
                ref_rgb = cv2.cvtColor(st.session_state.reference_image, cv2.COLOR_BGR2RGB)
                
                if zoom_ref != 100:
                    h, w = ref_rgb.shape[:2]
                    new_w = int(w * zoom_ref / 100)
                    new_h = int(h * zoom_ref / 100)
                    ref_rgb = cv2.resize(ref_rgb, (new_w, new_h))
                
                st.image(ref_rgb, use_container_width=True)
        else:
            # Single View
            zoom_level = st.slider("縮放", 25, 200, 100, 25, format="%d%%")
            result_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
            
            if zoom_level != 100:
                h, w = result_rgb.shape[:2]
                new_w = int(w * zoom_level / 100)
                new_h = int(h * zoom_level / 100)
                result_rgb = cv2.resize(result_rgb, (new_w, new_h))
            
            st.image(result_rgb, caption=f"縮放: {zoom_level}%")
        
        # Download Button
        is_success, buffer = cv2.imencode(".png", st.session_state.processed_image)
        if is_success:
            st.download_button("⬇️ 下載結果", buffer.tobytes(), "result.png", "image/png", use_container_width=True)
    else:
        st.info("👈 請在左側執行 Pipeline 後查看結果")

# Render Sidebar using Component
render_sidebar(engine)

if __name__ == "__main__":
    pass
