import streamlit as st
from utils.api import ask_question
from datetime import datetime


def render_chat():

    # ─── Session State ────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ─── Welcome Screen ───────────────────────────────────────────
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
        .welcome-card h2 { color: #ffffff; font-size: 1.5rem; margin-bottom: 8px; }
        .welcome-card p  { color: #7eb3d8; font-size: 0.95rem; margin-bottom: 24px; }
        .example-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
        .example-chip {
            background-color: #0d2137; border: 1px solid #1e3a5f;
            border-radius: 10px; padding: 10px 14px;
            color: #58b0e0; font-size: 0.85rem; text-align: left;
        }
        </style>
        <div class="welcome-card">
            <h2>👋 Welcome to MediRAG Assistant</h2>
            <p>Upload medical PDFs from the sidebar, then ask any question about them.</p>
            <div class="example-grid">
                <div class="example-chip">💊 What medications were prescribed?</div>
                <div class="example-chip">🩺 What is the patient's diagnosis?</div>
                <div class="example-chip">🧪 Compare lab results of both patients</div>
                <div class="example-chip">📋 Summarize findings from all reports</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Chat Header ──────────────────────────────────────────────
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("""
        <p style="color:#7eb3d8; font-size:0.85rem; margin-bottom:8px;">
            🟢 &nbsp;Assistant ready — Hybrid Search + Re-ranking + Multi-doc Reasoning
        </p>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ─── Render Chat History ──────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🏥"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Show RAG info for assistant messages
            if msg["role"] == "assistant":
                _render_rag_info(msg)

            if msg.get("timestamp"):
                st.caption(f"🕐 {msg['timestamp']}")

    # ─── Chat Input ───────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about your medical documents...")

    if user_input:
        timestamp = datetime.now().strftime("%I:%M %p")

        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
            st.caption(f"🕐 {timestamp}")

        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("🔍 Searching · Re-ranking · Reasoning..."):
                response = ask_question(user_input)

            if response.status_code == 200:
                data = response.json()

                # Extract all fields
                answer          = data.get("response", "No response received.")
                original_query  = data.get("original_query", user_input)
                rewritten_query = data.get("rewritten_query", user_input)
                reasoning_type  = data.get("reasoning_type", "simple_rag")
                chunks_retrieved= data.get("chunks_retrieved", 0)
                chunks_used     = data.get("chunks_used", 0)
                sources         = data.get("sources_involved", [])
                reranked        = data.get("reranked", [])

                # Show answer
                st.markdown(answer)

                # Show RAG info inline
                msg_data = {
                    "original_query":   original_query,
                    "rewritten_query":  rewritten_query,
                    "reasoning_type":   reasoning_type,
                    "chunks_retrieved": chunks_retrieved,
                    "chunks_used":      chunks_used,
                    "sources_involved": sources,
                    "reranked":         reranked
                }
                _render_rag_info(msg_data)

                bot_time = datetime.now().strftime("%I:%M %p")
                st.caption(f"🕐 {bot_time}")

                # Save to history
                st.session_state.messages.append({
                    "role":             "assistant",
                    "content":          answer,
                    "reranked":         reranked,
                    "timestamp":        bot_time,
                    **msg_data
                }) 

            else:
                error = response.json().get("error", response.text)
                st.error(f"⚠️ Error {response.status_code}: {error}")


# ─────────────────────────────────────────────────────────────────
# HELPER: Render RAG Pipeline Info
# Shows reasoning type, query rewrite, sources, chunks, reranked
# ─────────────────────────────────────────────────────────────────
def _render_rag_info(data: dict):

    reasoning_type  = data.get("reasoning_type", "")
    rewritten_query = data.get("rewritten_query", "")
    original_query  = data.get("original_query", "")
    chunks_retrieved= data.get("chunks_retrieved", 0)
    chunks_used     = data.get("chunks_used", 0)
    sources         = data.get("sources_involved", [])
    reranked        = data.get("reranked", [])

    # ── Reasoning Type Badge ──────────────────────────────────────
    if reasoning_type:
        if reasoning_type == "map_reduce":
            badge_color = "#0a3d0a"
            badge_text  = "🧠 Map-Reduce Multi-doc Reasoning"
            border_color= "#1a5c1a"
        else:
            badge_color = "#0a2e4a"
            badge_text  = "⚡ Simple RAG"
            border_color= "#1e3a5f"

        st.markdown(f"""
        <div style="display:inline-block; background:{badge_color};
                    border:1px solid {border_color}; border-radius:20px;
                    padding:3px 12px; margin:6px 0;">
            <span style="color:#c9d1d9; font-size:0.78rem;">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline Stats ────────────────────────────────────────────
    if chunks_retrieved or chunks_used:
        st.markdown(f"""
        <div style="display:flex; gap:10px; margin:6px 0; flex-wrap:wrap;">
            <span style="background:#1a1f2e; border:1px solid #2a2f3e;
                         border-radius:8px; padding:3px 10px;
                         color:#7eb3d8; font-size:0.75rem;">
                📦 Retrieved: {chunks_retrieved} chunks
            </span>
            <span style="background:#1a1f2e; border:1px solid #2a2f3e;
                         border-radius:8px; padding:3px 10px;
                         color:#7eb3d8; font-size:0.75rem;">
                ✅ Used: {chunks_used} chunks
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Query Rewriting ───────────────────────────────────────────
    if rewritten_query and rewritten_query != original_query:
        with st.expander("🔍 Query Rewriting"):
            st.markdown(f"""
            <div style="background:#0d2137; border:1px solid #1e3a5f;
                        border-radius:8px; padding:10px 14px;">
                <p style="color:#4a5568; font-size:0.78rem; margin:0 0 4px 0;">
                    Original:
                </p>
                <p style="color:#c9d1d9; font-size:0.85rem; margin:0 0 10px 0;">
                    {original_query}
                </p>
                <p style="color:#4a5568; font-size:0.78rem; margin:0 0 4px 0;">
                    Rewritten:
                </p>
                <p style="color:#58b0e0; font-size:0.85rem; margin:0;">
                    {rewritten_query}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Sources Involved ──────────────────────────────────────────
    if sources:
        with st.expander(f"📁 Sources Used ({len(sources)} document(s))"):
            for src in sources:
                # Show just filename not full path
                filename = src.split("\\")[-1].split("/")[-1]
                st.markdown(f"""
                <div style="background:#0d2137; border:1px solid #1e3a5f;
                            border-radius:8px; padding:8px 12px; margin:4px 0;">
                    <span style="color:#58b0e0; font-size:0.85rem;">
                        📄 {filename}
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # ── Reranked Chunks with Scores ───────────────────────────────
    if reranked:
        with st.expander(f"🎯 Re-ranked Chunks ({len(reranked)} selected)"):
            for i, chunk in enumerate(reranked):
                score   = chunk.get("score", 0)
                text    = chunk.get("text", "")[:200]

                # Color score based on value
                if score > 0.5:
                    score_color = "#27ae60"   # green — highly relevant
                elif score > 0.1:
                    score_color = "#e67e22"   # orange — medium
                else:
                    score_color = "#c0392b"   # red — low relevance

                st.markdown(f"""
                <div style="background:#0d2137; border:1px solid #1e3a5f;
                            border-radius:8px; padding:10px 14px; margin:6px 0;">
                    <div style="display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:6px;">
                        <span style="color:#c9d1d9; font-size:0.8rem;
                                     font-weight:600;">
                            Chunk {i+1}
                        </span>
                        <span style="color:{score_color}; font-size:0.78rem;
                                     font-weight:600;">
                            Score: {score:.4f}
                        </span>
                    </div>
                    <p style="color:#7eb3d8; font-size:0.78rem; margin:0;
                              line-height:1.5;">
                        {text}...
                    </p>
                </div>
                """, unsafe_allow_html=True)