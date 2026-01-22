
import streamlit as st

def apply_custom_style():
    """
    注入客製化 CSS 樣式 (Modern Lab Theme)
    """
    st.markdown("""
        <style>
        /* Import Font: Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* 1. Main Container Padding Clean up */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* 2. Custom Hero Header */
        .hero-header {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .hero-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, #60A5FA, #E879F9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
            font-size: 0.9rem;
            color: #94A3B8;
            margin-top: 0.5rem;
        }

        /* 3. Card-like Containers */
        .stExpander {
            background-color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }
        .stExpander:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border-color: #CBD5E1;
        }
        
        /* Remove default expander border to cleaner look */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #334155;
            background-color: transparent;
        }

        /* 4. Buttons */
        div.stButton > button {
            border-radius: 6px;
            font-weight: 500;
            transition: transform 0.1s;
        }
        div.stButton > button:active {
            transform: scale(0.98);
        }
        /* Primary Button Style */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
            border: none;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        }

        /* 5. Metrics & Toasts */
        [data-testid="stMetricValue"] {
            font-family: 'Inter', monospace;
            color: #2563EB;
        }
        
        /* 6. Sidebar Polish */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }
        
        /* 7. Image Captions */
        .stImage > div > div > div {
            font-size: 0.8rem;
            color: #64748B;
        }

        </style>
    """, unsafe_allow_html=True)

def render_hero_section():
    """
    渲染頂部 Hero 區塊
    """
    st.markdown("""
        <div class="hero-header">
            <div>
                <div class="hero-title">NKUST AoV Tool</div>
                <div class="hero-subtitle">FPGA-aware Computer Vision Pipeline Generator</div>
            </div>
            <div style="text-align: right; color: #94A3B8; font-size: 0.8rem;">
                v1.0.1<br>Visual Lab
            </div>
        </div>
    """, unsafe_allow_html=True)
