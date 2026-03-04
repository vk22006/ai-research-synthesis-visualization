"""
streamlit_app.py - Streamlit frontend for the AI Research Synthesis & Knowledge Graph Builder.

Run with:
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time
import requests
import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Research Knowledge Graph",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header gradient */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #4A90D9, #F4C430, #5CB85C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p {
        color: rgba(255,255,255,0.65);
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Paper card */
    .paper-card {
        background: linear-gradient(135deg, #1e1e2e, #252535);
        border: 1px solid rgba(74, 144, 217, 0.25);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .paper-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(74, 144, 217, 0.18);
        border-color: rgba(74, 144, 217, 0.5);
    }
    .paper-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #4A90D9;
        margin-bottom: 0.3rem;
    }
    .paper-meta {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        margin-bottom: 0.6rem;
    }
    .paper-summary {
        background: rgba(74, 144, 217, 0.08);
        border-left: 3px solid #4A90D9;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.85);
    }
    .paper-claim {
        background: rgba(244, 196, 48, 0.08);
        border-left: 3px solid #F4C430;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 500;
        margin-right: 6px;
    }
    .badge-blue  { background: rgba(74,144,217,0.2);  color: #4A90D9; }
    .badge-gold  { background: rgba(244,196,48,0.2);  color: #F4C430; }
    .badge-green { background: rgba(92,184,92,0.2);   color: #5CB85C; }

    /* Stat box */
    .stat-box {
        background: linear-gradient(135deg, #1e1e2e, #252535);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 700; color: #4A90D9; }
    .stat-label  { font-size: 0.78rem; color: rgba(255,255,255,0.5); }

    /* Section heading */
    .section-heading {
        font-size: 1.15rem;
        font-weight: 600;
        color: rgba(255,255,255,0.9);
        border-bottom: 2px solid rgba(74,144,217,0.4);
        padding-bottom: 0.4rem;
        margin: 1.4rem 0 1rem 0;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(90deg, #4A90D9, #5CB85C);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.88; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>🔬 AI Research Knowledge Graph</h1>
        <p>Autonomously synthesise research papers · Extract insights · Explore semantic relationships</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Search Settings")
    max_results = st.slider("Papers to fetch", min_value=5, max_value=20, value=12, step=1)
    similarity_threshold = st.slider(
        "Similarity threshold (edges)", min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        help="Minimum cosine similarity for drawing a 'similar_to' edge between papers",
    )
    st.divider()
    st.markdown(
        """
        **Node Legend**
        
        🔶 **Gold star** – Topic hub  
        🔵 **Blue circle** – Research paper  
        🟢 **Green diamond** – Shared concept
        
        **Edge Legend**
        
        ⬜ Gray – *related_to* topic  
        🟠 Orange – *similar_to* peer  
        ⬜ Light – *mentions* concept  
        """
    )
    st.divider()
    st.markdown("*Hover nodes for details · Drag to rearrange*")

# ── Main search bar ───────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Graph Neural Networks, BERT, Diffusion Models…",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("🔍 Search")

# ── Search & pipeline ─────────────────────────────────────────────────────────
if search_clicked and topic.strip():
    progress_placeholder = st.empty()
    progress_placeholder.info("⏳ Fetching papers and running AI analysis… (this may take 1–2 minutes on first run while models load)")

    try:
        resp = requests.get(
            f"{BACKEND_URL}/search",
            params={
                "topic": topic.strip(),
                "max_results": max_results,
                "similarity_threshold": similarity_threshold,
            },
            timeout=600,   # summarisation can take a while on CPU
        )
        progress_placeholder.empty()

        # Always try to get JSON – the backend now always returns JSON even on error
        try:
            data = resp.json()
        except Exception:
            st.error(
                f"❌ Backend returned non-JSON (status {resp.status_code}).\n\n"
                f"Raw response: `{resp.text[:500]}`\n\n"
                "Check the uvicorn terminal for the full traceback."
            )
            st.stop()

        if resp.status_code != 200:
            err_detail = data.get("detail") or data.get("error") or str(data)
            st.error(f"❌ Backend error {resp.status_code}: {err_detail}")
            st.stop()

    except requests.ConnectionError:
        progress_placeholder.empty()
        st.error(
            "❌ Could not connect to the backend. "
            "Make sure FastAPI is running: `uvicorn app:app --reload`"
        )
        st.stop()
    except requests.Timeout:
        progress_placeholder.empty()
        st.error(
            "⏱️ Request timed out. The NLP models may still be loading. "
            "Please wait 30 seconds and try again."
        )
        st.stop()
    except Exception as exc:
        progress_placeholder.empty()
        st.error(f"❌ Unexpected error: {exc}")
        st.stop()

    papers = data.get("papers", [])
    graph_stats = data.get("graph_stats", {})

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-heading">📊 Results for: <em>{topic}</em></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(papers)}</div>'
            f'<div class="stat-label">Papers Fetched</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{graph_stats.get("nodes", 0)}</div>'
            f'<div class="stat-label">Graph Nodes</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{graph_stats.get("edges", 0)}</div>'
            f'<div class="stat-label">Relationships</div></div>',
            unsafe_allow_html=True,
        )

    # ── Knowledge Graph ────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">🕸️ Interactive Knowledge Graph</div>', unsafe_allow_html=True)
    graph_html_path = data.get("graph_path", "")
    if graph_html_path and os.path.exists(graph_html_path):
        with open(graph_html_path, "r", encoding="utf-8") as f:
            graph_html = f.read()
        components.html(graph_html, height=720, scrolling=False)
    else:
        # Fallback: try to load from API
        try:
            gr = requests.get(f"{BACKEND_URL}/graph", timeout=10)
            if gr.status_code == 200:
                components.html(gr.text, height=720, scrolling=False)
            else:
                st.warning("Graph file not found locally. Check backend logs.")
        except Exception:
            st.warning("Could not load the graph visualisation.")

    # ── Paper cards ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">📄 Paper Summaries & Key Claims</div>', unsafe_allow_html=True)

    for i, paper in enumerate(papers, 1):
        authors_short = paper["authors"].split(",")[0] + (" et al." if "," in paper["authors"] else "")
        st.markdown(
            f"""
            <div class="paper-card">
                <div class="paper-title">{i}. {paper['title']}</div>
                <div class="paper-meta">
                    <span class="badge badge-blue">📅 {paper['published']}</span>
                    <span class="badge badge-green">👤 {authors_short}</span>
                    <a href="{paper['url']}" target="_blank" style="color:#4A90D9; font-size:0.78rem;">🔗 arXiv</a>
                </div>
                <div class="paper-summary">
                    <strong>🤖 Summary:</strong> {paper.get('summary', 'N/A')}
                </div>
                <div class="paper-claim">
                    <strong>💡 Key Claim:</strong> {paper.get('claim', 'N/A')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif search_clicked and not topic.strip():
    st.warning("Please enter a research topic first.")

else:
    # Empty state
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 2rem; color: rgba(255,255,255,0.35);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
            <div style="font-size: 1.1rem; font-weight: 500;">Enter a research topic above and click Search</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                Examples: "Transformer Attention", "Protein Folding", "Reinforcement Learning"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )