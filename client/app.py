import streamlit as st
from components.chatUI import render_chat
from components.upload import render_uploader
from components.history_downloader import render_history_download

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global Dark Theme CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
        padding-top: 10px;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }

    /* ── Top Header Banner ── */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d2137 60%, #0a3d62 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 4px 24px rgba(0, 120, 212, 0.15);
    }
    .main-header .icon {
        font-size: 3rem;
        line-height: 1;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.3px;
    }
    .main-header p {
        margin: 4px 0 0;
        font-size: 0.9rem;
        color: #7eb3d8;
    }

    /* ── Sidebar Header ── */
    .sidebar-header {
        background: linear-gradient(135deg, #0d2137, #0a3d62);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid #1e3a5f;
    }
    .sidebar-header h2 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    .sidebar-header p {
        margin: 4px 0 0;
        font-size: 0.75rem;
        color: #7eb3d8;
    }

    /* ── Divider ── */
    .custom-divider {
        border: none;
        border-top: 1px solid #2a2f3e;
        margin: 18px 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0a3d62, #0078d4);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0078d4, #005fa3);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 120, 212, 0.3);
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background-color: #161b27;
        border: 1.5px dashed #2a4a6e;
        border-radius: 12px;
        padding: 12px;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background-color: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background-color: #161b27 !important;
        border: 1px solid #2a4a6e !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        background-color: #1a2235 !important;
        color: #7eb3d8 !important;
        border: 1px solid #2a4a6e !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
    }
    .stDownloadButton > button:hover {
        background-color: #0a3d62 !important;
        color: white !important;
    }

    /* ── Scrollbar ── */ 
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2a4a6e; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #0078d4; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🏥 MediRAG</h2>
        <p>AI Medical Document Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    render_uploader()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    render_history_download()

# ─── Main Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="icon">👨‍⚕️</div>
    <div>
        <h1>AI Medical Assistant</h1>
        <p>Upload your medical reports and ask questions — powered by RAG + LLM</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Chat ────────────────────────────────────────────────────────────────
render_chat()