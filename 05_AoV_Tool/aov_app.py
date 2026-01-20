"""
NKUST AoV Tool - Streamlit Application
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import json
# Handle pyvis import safely
try:
    from pyvis.network import Network
except ImportError:
    st.error("Missing dependency: pyvis. Please install it using `pip install pyvis`.")
    Network = None

from logic_engine import LogicEngine
from processor import ImageProcessor
from library_manager import LibraryManager
from project_manager import ProjectManager
from import_parser import ImportParser
from code_generator import CodeGenerator
from templates import get_default_templates

# ==================== UI Components ====================

def render_parameter_editor(node: dict, idx: int, node_id: str):
    """
    渲染參數編輯器 (UI Component)
    """
    params = node.get('parameters', {})
    if not params:
        return

    st.markdown("**操作參數**")
    for param_name, param_info in params.items():
        param_default = param_info.get('default')
        param_desc = param_info.get('description', param_name)
        
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            st.caption(param_name)
        
        with col_p2:
            key = f"param_{node_id}_{param_name}"
            
            # Boolean
            if isinstance(param_default, bool):
                new_value = st.checkbox(param_desc, value=param_default, key=key, label_visibility="collapsed")
            
            # Integer
            elif isinstance(param_default, int):
                new_value = st.number_input(param_desc, value=param_default, step=1, key=key, label_visibility="collapsed")
            
            # Float
            elif isinstance(param_default, float):
                new_value = st.number_input(param_desc, value=param_default, step=0.1, format="%.2f", key=key, label_visibility="collapsed")
            
            # List (as string)
            elif isinstance(param_default, list):
                new_value_str = st.text_input(param_desc, value=str(param_default), key=key, label_visibility="collapsed")
                try:
                    new_value = json.loads(new_value_str)
                except:
                    new_value = param_default
            
            # String/Other
            else:
                new_value = st.text_input(param_desc, value=str(param_default), key=key, label_visibility="collapsed")
            
            # Update State
            if new_value != param_default:
                st.session_state.pipeline[idx]['parameters'][param_name]['default'] = new_value

# ==================== Main App ====================

st.set_page_config(
    page_title="NKUST AoV Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_engine():
    return LogicEngine()

@st.cache_resource
def init_processor():
    return ImageProcessor()

engine = init_engine()
processor = init_processor()

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'llm_api_key' not in st.session_state:
    st.session_state.llm_api_key = ""
if 'llm_base_url' not in st.session_state:
    st.session_state.llm_base_url = "https://api.openai.com/v1"
if 'use_mock_llm' not in st.session_state:
    st.session_state.use_mock_llm = True

st.title("NKUST AoV 演算法視覺化工具")

col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.header("編輯器")
    
    st.subheader("1. 上傳影像")
    uploaded_file = st.file_uploader(
        "選擇影像檔案",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state.uploaded_image = img_bgr
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # width=None caused StreamlitInvalidWidthError in newer versions
        # Changing to None default behavior (letting streamlit handle it) or explicit value
        st.image(img_rgb, caption="原始影像")
    
    st.divider()
    
    st.subheader("2. 描述需求")
    user_query = st.text_input(
        "輸入需求描述",
        placeholder="例如：偵測硬幣、找出邊緣、降噪處理"
    )
    
    col_gen1, col_gen2 = st.columns(2)
    
    with col_gen1:
        if st.button("產生 Pipeline", type="primary", use_container_width=True):
            if user_query:
                # Update API Key if provided
                if st.session_state.llm_api_key:
                    engine.prompt_master.api_key = st.session_state.llm_api_key
                    engine.prompt_master.base_url = st.session_state.llm_base_url
                    engine.prompt_master.llm_available = True
                
                with st.spinner("處理中..."):
                    pipeline = engine.process_user_query(
                        user_query, 
                        use_mock_llm=st.session_state.use_mock_llm
                    )
                    st.session_state.pipeline = pipeline
                    st.success(f"已產生 {len(pipeline)} 個節點")
                    st.rerun()
            else:
                st.warning("請先輸入需求描述")
    
    with col_gen2:
        if st.button("重置", use_container_width=True):
            st.session_state.pipeline = []
            st.session_state.processed_image = None
            st.rerun()
    
    st.divider()

    # ========== 快速模板 (Templates) ==========
    st.subheader("2.5 快速模板")
    templates = get_default_templates()
    selected_template = st.selectbox("選擇預設場景", list(templates.keys()))
    
    if st.button("📥 載入模板", use_container_width=True):
        raw_nodes = templates[selected_template]
        hydrated_pipeline = []
        
        # Hydrate nodes with library data
        for idx, raw_node in enumerate(raw_nodes):
            func_name = raw_node['function']
            # Try to find in library to get constraints and full info
            # We search both official and contributed
            algo_data = engine.lib_manager.get_algorithm(func_name, 'official')
            if not algo_data:
                algo_data = engine.lib_manager.get_algorithm(func_name, 'contributed')
            
            node_to_add = raw_node.copy()
            if algo_data:
                # Merge constraints and description
                node_to_add['category'] = algo_data.get('category', 'unknown')
                node_to_add['description'] = algo_data.get('description', '')
                node_to_add['fpga_constraints'] = algo_data.get('fpga_constraints', {})
                # Keep template parameters as overrides, but ensure structure exists
                # If template has params, use them. If not, use default.
                # Here we trust the template definition.
            else:
                # Fallback constraints
                node_to_add['fpga_constraints'] = {"estimated_clk": 0, "resource_usage": "Unknown", "latency_type": "Software"}
                
            node_to_add['id'] = f"node_{idx}"
            node_to_add['_enabled'] = True
            hydrated_pipeline.append(node_to_add)
            
        st.session_state.pipeline = hydrated_pipeline
        st.session_state.processed_image = None
        st.success(f"已載入: {selected_template}")
        st.rerun()

    st.divider()
    
    if st.session_state.pipeline:
        st.subheader("3. Pipeline 編輯")
        
        # ========== 新增節點功能 ==========
        with st.expander("新增節點", expanded=False):
            all_algos = engine.lib_manager.list_algorithms()
            
            # 建立顯示選項：同時包含中英文名稱
            algo_options = {}
            for a in all_algos:
                # 格式：演算法名稱 / 中文名稱 (類別)
                cn_name = a.get('name_zh', a['name'])
                display_name = f"{a['name']} / {cn_name} ({a['category']})"
                algo_options[display_name] = a
            
            selected_algo_name = st.selectbox("選擇演算法", list(algo_options.keys()), key="new_algo_select")
            insert_position = st.number_input("插入位置", min_value=0, max_value=len(st.session_state.pipeline), value=len(st.session_state.pipeline), key="insert_pos")
            
            if st.button("新增節點", type="primary", use_container_width=True):
                algo_data = algo_options[selected_algo_name]
                algo_id = algo_data['_algo_id']
                
                new_node = {
                    'id': f'node_{insert_position}',
                    'name': algo_data['name'],
                    'function': algo_data.get('opencv_function', algo_id),
                    'category': algo_data['category'],
                    'description': algo_data.get('description', ''),
                    'parameters': algo_data.get('parameters', {}),
                    'fpga_constraints': algo_data['fpga_constraints'],
                    'source': 'manual_add',
                    '_enabled': True
                }
                
                st.session_state.pipeline.insert(insert_position, new_node)
                
                for i, n in enumerate(st.session_state.pipeline):
                    n['id'] = f"node_{i}"
                
                st.success(f"已新增 {algo_data['name']}")
                st.rerun()
        
        for idx, node in enumerate(st.session_state.pipeline):
            node_id = node.get('id', f'node_{idx}')
            node_name = node.get('name', '未知節點')
            fpga = node.get('fpga_constraints', {})
            is_enabled = node.get('_enabled', True)
            
            with st.expander(f"[{idx}] {node_name}" + (" (已停用)" if not is_enabled else ""), expanded=False):
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                
                with col_btn1:
                    if st.button("上移", key=f"up_{idx}", disabled=(idx == 0)):
                        st.session_state.pipeline[idx], st.session_state.pipeline[idx-1] = \
                            st.session_state.pipeline[idx-1], st.session_state.pipeline[idx]
                        for i, n in enumerate(st.session_state.pipeline):
                            n['id'] = f"node_{i}"
                        st.rerun()
                
                with col_btn2:
                    if st.button("下移", key=f"down_{idx}", disabled=(idx == len(st.session_state.pipeline)-1)):
                        st.session_state.pipeline[idx], st.session_state.pipeline[idx+1] = \
                            st.session_state.pipeline[idx+1], st.session_state.pipeline[idx]
                        for i, n in enumerate(st.session_state.pipeline):
                            n['id'] = f"node_{i}"
                        st.rerun()
                
                with col_btn3:
                    move_to = st.number_input("移至", min_value=0, max_value=len(st.session_state.pipeline)-1, value=idx, key=f"moveto_{idx}", label_visibility="collapsed")
                    if move_to != idx and st.button("GO", key=f"move_{idx}"):
                        node_to_move = st.session_state.pipeline.pop(idx)
                        st.session_state.pipeline.insert(move_to, node_to_move)
                        for i, n in enumerate(st.session_state.pipeline):
                            n['id'] = f"node_{i}"
                        st.rerun()
                
                with col_btn4:
                    skip_label = "啟用" if not is_enabled else "跳過"
                    if st.button(skip_label, key=f"skip_{idx}"):
                        st.session_state.pipeline[idx]['_enabled'] = not is_enabled
                        st.rerun()
                
                with col_btn5:
                    if st.button("刪除", key=f"del_{idx}"):
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
                    "CLK",
                    min_value=0,
                    max_value=100000,
                    value=current_clk,
                    step=10,
                    key=f"clk_{node_id}"
                )
                
                if new_clk != current_clk:
                    st.session_state.pipeline[idx]['fpga_constraints']['estimated_clk'] = new_clk
                
                st.caption(f"資源: {fpga.get('resource_usage', 'Unknown')}")
                st.caption(f"延遲: {fpga.get('latency_type', 'Unknown')}")
                
                render_parameter_editor(node, idx, node_id)
                
                if '_warning' in node:
                    st.warning(node['_warning'])
    
    st.divider()
    
    st.subheader("4. 專案管理")
    
    with st.expander("匯出專案 (保存參數)", expanded=False):
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
                    label="下載專案檔 (.json)",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                    use_container_width=True,
                    type="primary",
                    help="點擊後瀏覽器會自動下載JSON檔案"
                )
            
            with col_dl2:
                if st.button("儲存到專案資料夾", use_container_width=True, help="直接儲存到目前資料夾"):
                    try:
                        save_path = f"e:/LAB_DATA/ORB/experiments/05_AoV_Tool/{filename}"
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(json_str)
                        st.success(f"已儲存: {filename}")
                        st.info(f"路徑: {save_path}")
                    except Exception as e:
                        st.error(f"儲存失敗: {e}")
            
            st.caption(f"檔案名稱: {filename} | 大小: {len(json_str)} bytes")
            
            with st.expander("參數驗證 & 即時預覽 (JSON)", expanded=False):
                st.caption("此處顯示即將匯出的 Pipeline 資料，請確認 'default' 值是否正確更新。")
                
                # Show simplified view for validation
                val_view = []
                for idx, node in enumerate(st.session_state.pipeline):
                    node_summary = {
                        "name": node['name'],
                        "parameters": {}
                    }
                    if node.get('parameters'):
                        for pname, pinfo in node['parameters'].items():
                            node_summary["parameters"][pname] = pinfo.get('default')
                    val_view.append(node_summary)
                
                st.json(val_view)
            
            with st.expander("查看完整原始 JSON", expanded=False):
                st.code(json_str, language="json", line_numbers=True)
                st.info("這是實際存檔的內容")
            
            st.markdown("### 程式碼生成 (Code Generation)")
            tab_py, tab_vhdl, tab_verilog = st.tabs(["Python (OpenCV)", "Vivado-VHDL (FPGA)", "Vivado-Verilog (FPGA)"])
            
            with tab_py:
                py_code = CodeGenerator.generate_python_script(st.session_state.pipeline)
                st.code(py_code, language="python")
                st.download_button("下載 Python 腳本", py_code, "pipeline.py", "text/x-python")
                
            with tab_vhdl:
                vhdl_code = CodeGenerator.generate_vhdl(st.session_state.pipeline)
                st.code(vhdl_code, language="vhdl")
                st.download_button("下載 Vivado-VHDL", vhdl_code, "pipeline.vhd", "text/plain")

            with tab_verilog:
                verilog_code = CodeGenerator.generate_verilog(st.session_state.pipeline)
                st.code(verilog_code, language="verilog")
                st.download_button("下載 Vivado-Verilog", verilog_code, "pipeline.v", "text/plain")

            total_clk = sum(n['fpga_constraints'].get('estimated_clk', 0) for n in st.session_state.pipeline)
            st.success(f"總時脈: {total_clk} clk | 節點數: {len(st.session_state.pipeline)}")
        else:
            st.info("請先產生 Pipeline")
    
    with st.expander("匯入專案", expanded=False):
        st.caption("支援: JSON, DOT, TXT")
        
        uploaded_project = st.file_uploader(
            "選擇檔案",
            type=['json', 'dot', 'txt', 'md'],
            key="project_uploader"
        )
        
        if uploaded_project is not None:
            try:
                json_content = uploaded_project.read().decode('utf-8')
                file_extension = uploaded_project.name.split('.')[-1].lower()
                
                parser = ImportParser()
                
                if file_extension == 'json':
                    result = ProjectManager.load_project_from_json(json_content)
                    
                    if result["success"]:
                        meta = result["meta"]
                        hw_summary = result.get("hardware_summary", {})
                        
                        st.info(f"""
專案: {meta.get('project_name', 'N/A')}
作者: {meta.get('author', 'Unknown')}
節點: {hw_summary.get('total_nodes', 0)}
總時脈: {hw_summary.get('total_clk', 0)} clk
                        """)
                        
                        if st.button("載入", type="primary", use_container_width=True, key="load_project_btn"):
                            st.session_state.pipeline = result["pipeline"]
                            st.session_state.processed_image = None
                            st.success("載入成功")
                            st.rerun()
                    else:
                        st.error(f"失敗: {result['error']}")
                
                else:
                    parse_result = parser.parse(json_content)
                    
                    if parse_result['success']:
                        format_name = {'dot': 'DOT', 'txt': 'TXT'}.get(parse_result['format'], parse_result['format'].upper())
                        
                        st.info(f"""
格式: {format_name}
節點: {parse_result['node_count']}
                        """)
                        
                        if st.button("載入", type="primary", use_container_width=True, key="load_pipeline_btn"):
                            st.session_state.pipeline = parse_result['pipeline']
                            st.session_state.processed_image = None
                            st.success("載入成功")
                            st.rerun()
                    else:
                        st.error(f"失敗: {parse_result['error']}")
            
            except Exception as e:
                st.error(f"錯誤: {str(e)}")
    
    st.divider()
    
    if st.session_state.pipeline and st.session_state.uploaded_image is not None:
        if st.button("執行 Pipeline", type="primary", use_container_width=True):
            with st.spinner("處理中..."):
                try:
                    active_pipeline = [n for n in st.session_state.pipeline if n.get('_enabled', True)]
                    
                    result = processor.execute_pipeline(
                        st.session_state.uploaded_image,
                        active_pipeline,
                        debug_mode=False
                    )
                    st.session_state.processed_image = result
                    
                    disabled_count = len(st.session_state.pipeline) - len(active_pipeline)
                    if disabled_count > 0:
                        st.success(f"完成 (跳過 {disabled_count} 個節點)")
                    else:
                        st.success("處理完成")
                except Exception as e:
                    st.error(f"失敗: {e}")

with col_right:
    st.header("視覺化預覽")
    
    st.subheader("流程圖")
    
    if st.session_state.pipeline:
        if Network:
            # Create pyvis network
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            
            # Configure physics
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "stabilization": {"enabled": true, "iterations": 100},
                "barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.3, "springLength": 200}
              },
              "interaction": {"dragNodes": true, "dragView": true, "zoomView": true}
            }
            """)
            
            # Add nodes
            for idx, node in enumerate(st.session_state.pipeline):
                node_id = node.get('id', f'node_{idx}')
                node_name = node.get('name', '未知')
                fpga = node.get('fpga_constraints', {})
                clk = fpga.get('estimated_clk', 0)
                latency_type = fpga.get('latency_type', 'Unknown')
                resource = fpga.get('resource_usage', 'Unknown')
                params = node.get('parameters', {})
                
                param_strs = []
                for i, (k, v) in enumerate(params.items()):
                    if i >= 2:
                        break
                    default_val = v.get('default', '?')
                    if isinstance(default_val, list):
                        default_val = str(default_val)
                    param_strs.append(f"{k}:{default_val}")
                
                param_line = ", ".join(param_strs) if param_strs else "無參數"
                
                label = f"{node_name}\n{param_line}\n{latency_type}\n{clk} clk"
                title = f"<b>{node_name}</b><br>CLK: {clk}<br>資源: {resource}<br>延遲: {latency_type}"
                
                color_map = {
                    'Low': '#C8E6C9',
                    'Medium': '#FFF9C4',
                    'High': '#FFCCBC',
                    'Very High': '#FFCDD2'
                }
                color = color_map.get(resource, '#E0E0E0')
                
                net.add_node(node_id, label=label, title=title, color=color, shape='box',
                            font={'size': 14, 'face': 'Microsoft JhengHei'}, borderWidth=2)
            
            # Add edges
            for i in range(len(st.session_state.pipeline) - 1):
                current_node = st.session_state.pipeline[i]
                next_node = st.session_state.pipeline[i + 1]
                
                current_id = current_node.get('id', f'node_{i}')
                next_id = next_node.get('id', f'node_{i+1}')
                
                clk_label = f"{current_node['fpga_constraints'].get('estimated_clk', 0)} clk"            
                net.add_edge(current_id, next_id, label=clk_label, color='#1976D2', arrows='to',
                            font={'size': 12, 'face': 'Microsoft JhengHei'})
            
            # Generate and display
            html_file = "pipeline_graph.html"
            net.save_graph(html_file)
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            components.html(html_content, height=620, scrolling=False)
            
            total_clk = sum(n['fpga_constraints'].get('estimated_clk', 0) for n in st.session_state.pipeline)
            st.metric("總時脈", f"{total_clk} clk")
        else:
            st.warning("無法載入流程圖模組 (pyvis)")
        
    else:
        st.info("請先產生 Pipeline")
    
    st.divider()
    
    st.subheader("處理結果")
    
    if st.session_state.processed_image is not None:
        zoom_level = st.slider(
            "縮放",
            min_value=25,
            max_value=200,
            value=100,
            step=25,
            format="%d%%"
        )
        
        result_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
        
        if zoom_level != 100:
            h, w = result_rgb.shape[:2]
            new_w = int(w * zoom_level / 100)
            new_h = int(h * zoom_level / 100)
            result_rgb = cv2.resize(result_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        st.image(result_rgb, caption=f"({zoom_level}%)")
        
        is_success, buffer = cv2.imencode(".png", st.session_state.processed_image)
        if is_success:
            st.download_button(
                label="下載",
                data=buffer.tobytes(),
                file_name="result.png",
                mime="image/png",
                use_container_width=True
            )
    else:
        st.info("執行後顯示結果")

with st.sidebar:
    st.header("系統資訊")
    
    st.caption(f"版本: {engine.lib_manager.data.get('schema_version', 'N/A')}")
    st.caption(f"官方: {len(engine.lib_manager.data['libraries']['official'])} 個")
    st.caption(f"貢獻: {len(engine.lib_manager.data['libraries']['contributed'])} 個")
    
    st.divider()
    
    st.header("LLM 設定")
    st.caption("設定 LLM API 以獲得更靈活的建議")
    
    use_mock = st.toggle("使用 Mock 模式 (測試用)", value=st.session_state.use_mock_llm)
    st.session_state.use_mock_llm = use_mock
    
    if not use_mock:
        api_key = st.text_input("API Key (OpenAI format)", type="password", value=st.session_state.llm_api_key)
        
        base_url_help = """
        **Base URL 設定指南：**
        - **OpenAI (官方)**: `https://api.openai.com/v1` (預設)
        - **Groq**: `https://api.groq.com/openai/v1`
        - **DeepSeek**: `https://api.deepseek.com`
        - **Local (LM Studio)**: `http://localhost:1234/v1`
        - **Google Gemini**: `https://generativelanguage.googleapis.com/v1beta/openai/`
        """
        
        base_url = st.text_input(
            "Base URL (Optional)", 
            value=st.session_state.llm_base_url, 
            placeholder="https://api.openai.com/v1",
            help=base_url_help
        )
        
        if api_key:
            st.session_state.llm_api_key = api_key
            st.session_state.llm_base_url = base_url
            st.caption("✅ API Key 已設定")
        else:
            st.warning("請輸入 API Key")
    
    st.divider()
    st.caption("NKUST Vision Lab")

# Add a footer to catch direct execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 0 and "streamlit" not in sys.argv[0]:
        print("\n[INFO] 建議使用以下指令啟動：")
        print("python run_tool.py")
        print("\n或者直接執行：")
        print("python -m streamlit run aov_app.py\n")
