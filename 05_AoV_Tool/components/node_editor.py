
import streamlit as st
import json

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
