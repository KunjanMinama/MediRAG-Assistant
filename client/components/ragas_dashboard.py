import streamlit as st
import requests
import json


API_BASE = "http://127.0.0.1:8000"


# ─────────────────────────────────────────────────────────────────
# SCORE COLORS
# ─────────────────────────────────────────────────────────────────
def _score_color(score: float) -> str:
    if score >= 0.8:   return "#27ae60"   # green  — excellent
    elif score >= 0.6: return "#2e86c1"   # blue   — good
    elif score >= 0.4: return "#e67e22"   # orange — fair 
    else:              return "#c0392b"   # red    — poor

def _score_label(score: float) -> str:
    if score >= 0.8:   return "Excellent"
    elif score >= 0.6: return "Good"
    elif score >= 0.4: return "Fair"
    else:              return "Poor"

def _score_bar(score: float) -> str:
    filled = int(score * 10)
    empty  = 10 - filled
    color  = _score_color(score)
    bar    = "█" * filled + "░" * empty
    return f'<span style="color:{color}; font-size:1.1rem;">{bar}</span>'


# ─────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────
def render_ragas_dashboard():

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1f2e,#0d2137);
                border:1px solid #1e3a5f; border-radius:14px;
                padding:20px 24px; margin-bottom:24px;">
        <h2 style="color:#ffffff; margin:0; font-size:1.4rem;">
            🧪 RAGAS Evaluation Dashboard
        </h2>
        <p style="color:#7eb3d8; margin:6px 0 0; font-size:0.88rem;">
            Measure your RAG pipeline quality with 4 key metrics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Explanations ───────────────────────────────────────
    with st.expander("📚 What do these metrics mean?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Faithfulness**
            Is the answer grounded in the retrieved context?
            Low = LLM is hallucinating. Target: > 0.8

            **💡 Answer Relevancy**
            Does the answer address the question?
            Low = Answer is off-topic. Target: > 0.8
            """)
        with col2:
            st.markdown("""
            **🔍 Context Recall**
            Did retrieval find ALL relevant information?
            Low = Missing chunks → increase TOP_K. Target: > 0.7

            **📌 Context Precision**
            Are retrieved chunks actually relevant?
            Low = Too much noise → improve reranking. Target: > 0.8
            """)

    # ── Evaluation Form ───────────────────────────────────────────
    st.markdown("### 📝 Evaluate a Response")

    # Auto fill from last chat if available
    last_question = ""
    last_answer   = ""
    last_contexts = []

    if st.session_state.get("messages"):
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                last_answer = msg.get("content", "")
                last_contexts = [
                    r["text"]
                    for r in msg.get("reranked", [])
                ]
            if msg["role"] == "user" and last_answer:
                last_question = msg.get("content", "")
                break

    question = st.text_area(
        "Question",
        value=last_question,
        placeholder="Enter the question that was asked...",
        height=80
    )

    answer = st.text_area(
        "Answer (LLM Response)",
        value=last_answer,
        placeholder="Paste the LLM's answer here...",
        height=120
    )

    ground_truth = st.text_area(
        "Ground Truth (Optional but recommended)",
        placeholder="Enter the ideal/correct answer for better evaluation...",
        height=80,
        help="Providing ground truth gives more accurate Context Recall scores"
    )

    # Show auto-detected contexts
    if last_contexts:
        st.markdown(f"**Contexts:** {len(last_contexts)} chunks auto-detected from last response")
        with st.expander("View detected contexts"):
            for i, ctx in enumerate(last_contexts):
                st.markdown(f"**Chunk {i+1}:** {ctx[:200]}...")

    # ── Run Evaluation ────────────────────────────────────────────
    if st.button("🚀 Run RAGAS Evaluation", type="primary", use_container_width=True):

        if not question or not answer:
            st.warning("⚠️ Please provide both question and answer!")
            return

        contexts_to_use = last_contexts if last_contexts else [answer]
        print(f"Contexts to use: {len(contexts_to_use)}")

        with st.spinner("🧪 Running RAGAS evaluation... This may take 30-60 seconds..."):
            try:
                response = requests.post(
                    f"{API_BASE}/evaluate/",
                    data={
                        "question":     question,
                        "answer":       answer,
                        "contexts":     json.dumps(contexts_to_use),
                        "ground_truth": ground_truth if ground_truth else ""
                    },
                    timeout=300
                )

                if response.status_code == 200:
                    data   = response.json()
                    scores = data.get("scores", {})
                    _render_scores(scores)

                    # Save to session for history
                    if "ragas_history" not in st.session_state:
                        st.session_state.ragas_history = []
                    st.session_state.ragas_history.append({
                        "question": question[:60],
                        "scores":   scores
                    })
                else:
                    st.error(f"❌ Evaluation failed: {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Evaluation timed out. Try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ── Evaluation History ────────────────────────────────────────
    if st.session_state.get("ragas_history"):
        st.markdown("---")
        st.markdown("### 📊 Evaluation History")
        _render_history(st.session_state.ragas_history)


# ─────────────────────────────────────────────────────────────────
# RENDER SCORES
# ─────────────────────────────────────────────────────────────────
def _render_scores(scores: dict):

    if scores.get("error"):
        st.error(f"❌ Evaluation error: {scores['error']}")
        return

    st.markdown("### 📊 RAGAS Scores")

    # Overall score banner
    overall = scores.get("overall", 0)
    overall_color = _score_color(overall)
    overall_label = _score_label(overall)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d2137,#0a3d62);
                border:2px solid {overall_color}; border-radius:14px;
                padding:20px; text-align:center; margin-bottom:20px;">
        <p style="color:#7eb3d8; font-size:0.9rem; margin:0 0 6px 0;">
            Overall RAG Score
        </p>
        <h1 style="color:{overall_color}; font-size:3rem; margin:0;">
            {overall:.2f}
        </h1>
        <p style="color:{overall_color}; font-size:1rem; margin:6px 0 0 0;
                  font-weight:600;">
            {overall_label}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Individual metric scores
    metrics = [
        ("🎯 Faithfulness",      "faithfulness",      "Is answer grounded in context?"),
        ("💡 Answer Relevancy",  "answer_relevancy",  "Does answer address the question?"),
        ("🔍 Context Recall",    "context_recall",    "Did retrieval find all relevant info?"),
        ("📌 Context Precision", "context_precision", "Were retrieved chunks relevant?"),
    ]

    col1, col2 = st.columns(2)

    for i, (name, key, desc) in enumerate(metrics):
        score = scores.get(key, 0)
        color = _score_color(score)
        label = _score_label(score)
        bar   = _score_bar(score)

        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div style="background:#161b27; border:1px solid #2a2f3e;
                        border-radius:12px; padding:16px; margin-bottom:12px;">
                <p style="color:#c9d1d9; font-size:0.9rem; font-weight:600;
                           margin:0 0 4px 0;">
                    {name}
                </p>
                <p style="color:#4a5568; font-size:0.75rem; margin:0 0 10px 0;">
                    {desc}
                </p>
                <div style="display:flex; align-items:center;
                            justify-content:space-between;">
                    {bar}
                    <span style="color:{color}; font-size:1.3rem;
                                 font-weight:700; margin-left:10px;">
                        {score:.2f}
                    </span>
                </div>
                <p style="color:{color}; font-size:0.78rem; margin:6px 0 0 0;
                           font-weight:600;">
                    {label}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Improvement suggestions
    st.markdown("### 💡 Improvement Suggestions")
    suggestions = _get_suggestions(scores)
    if suggestions:
        for s in suggestions:
            st.markdown(f"""
            <div style="background:#0d2137; border-left:3px solid #0078d4;
                        border-radius:0 8px 8px 0; padding:10px 14px;
                        margin:6px 0;">
                <p style="color:#c9d1d9; font-size:0.85rem; margin:0;">{s}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 All metrics look great! Your RAG pipeline is performing excellently!")


def _get_suggestions(scores: dict) -> list:
    """Generate improvement suggestions based on scores."""
    suggestions = []

    if scores.get("faithfulness", 1) < 0.8:
        suggestions.append(
            "⚠️ Low Faithfulness — LLM may be hallucinating. "
            "Strengthen prompt to only use provided context."
        )
    if scores.get("answer_relevancy", 1) < 0.8:
        suggestions.append(
            "⚠️ Low Answer Relevancy — Answer is off-topic. "
            "Improve your prompt template to be more focused."
        )
    if scores.get("context_recall", 1) < 0.7:
        suggestions.append(
            "⚠️ Low Context Recall — Missing relevant chunks. "
            "Increase TOP_K from 10 to 15, or increase chunk size."
        )
    if scores.get("context_precision", 1) < 0.8:
        suggestions.append(
            "⚠️ Low Context Precision — Too many irrelevant chunks. "
            "Reduce TOP_N after reranking, or adjust hybrid alpha."
        )

    return suggestions


def _render_history(history: list):
    """Render evaluation history table."""
    st.markdown(f"**{len(history)} evaluation(s) run this session:**")

    for i, item in enumerate(reversed(history)):
        overall = item["scores"].get("overall", 0)
        color   = _score_color(overall)

        st.markdown(f"""
        <div style="background:#161b27; border:1px solid #2a2f3e;
                    border-radius:10px; padding:12px 16px; margin:6px 0;
                    display:flex; justify-content:space-between;
                    align-items:center;">
            <span style="color:#c9d1d9; font-size:0.85rem;">
                {len(history)-i}. {item['question']}...
            </span>
            <span style="color:{color}; font-weight:700; font-size:1rem;">
                {overall:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)