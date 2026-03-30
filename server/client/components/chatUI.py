import streamlit as st
from utils.api import ask_question
from datetime import datetime


def render_chat():

    # ─── Session State ────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ─── Welcome Screen ───────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <style>
        .welcome-card {
            background: linear-gradient(135deg, #161b27, #1a2235);
            border: 1px solid #2a2f3e;
            border-radius: 16px;
            padding: 32px 36px;
            margin: 20px 0 32px 0;
            text-align: center;
        }
        .welcome-card h2 {
            color: #ffffff;
            font-size: 1.5rem;
            margin-bottom: 8px;
        }
        .welcome-card p {
            color: #7eb3d8;
            font-size: 0.95rem;
            margin-bottom: 24px;
        }
        .example-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 16px;
        }
        .example-chip {
            background-color: #0d2137;
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            padding: 10px 14px;
            color: #58b0e0;
            font-size: 0.85rem;
            text-align: left;
        }
        </style>
        <div class="welcome-card">
            <h2>👋 Welcome to MediRAG Assistant</h2>
            <p>Upload a medical PDF from the sidebar, then ask any question about it.</p>
            <div class="example-grid">
                <div class="example-chip">💊 What medications were prescribed?</div>
                <div class="example-chip">🩺 What is the patient's diagnosis?</div>
                <div class="example-chip">🧪 What were the abnormal lab results?</div>
                <div class="example-chip">📋 What did the doctor recommend?</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Chat Header ──────────────────────────────────────────────────────────
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("""
        <p style="color:#7eb3d8; font-size:0.85rem; margin-bottom:8px;">
            🟢 &nbsp;Assistant is ready — ask anything about your document
        </p>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ─── Render Chat History ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🏥"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("timestamp"):
                st.caption(f"🕐 {msg['timestamp']}")

    # ─── Chat Input ───────────────────────────────────────────────────────────
    user_input = st.chat_input("Type your question about the medical document...")

    if user_input:
        timestamp = datetime.now().strftime("%I:%M %p")

        # Show user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
            st.caption(f"🕐 {timestamp}")

        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        # Get and show assistant response
        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("MediRAG is analyzing your document..."):
                response = ask_question(user_input)

            if response.status_code == 200:
                data   = response.json()
                answer = data["response"]
                sources = data.get("sources", [])

                st.markdown(answer)

                # Show sources if available
                if sources and any(sources):
                    with st.expander("📃 View Sources"):
                        for src in sources:
                            if src:
                                st.markdown(f"- `{src}`")

                bot_time = datetime.now().strftime("%I:%M %p")
                st.caption(f"🕐 {bot_time}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": bot_time
                })

            else:
                st.error(f"⚠️ Error {response.status_code}: {response.text}")