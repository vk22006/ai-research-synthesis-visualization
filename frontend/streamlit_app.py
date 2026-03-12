from __future__ import annotations

import os
import sys
import time
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AI Research Knowledge Graph",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        background-color: #0d1117;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Modern Flat Header */
    .app-header {
        background: #161b22;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #30363d;
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        color: #f0f6fc;
    }
    .app-header p {
        color: #8b949e;
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }

    /* Sleek Paper Card */
    .paper-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: border-color 0.2s ease;
    }
    .paper-card:hover {
        border-color: #58a6ff;
    }
    .paper-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f0f6fc;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .paper-meta {
        display: flex;
        gap: 12px;
        align-items: center;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        background: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    .paper-summary {
        font-size: 0.9rem;
        color: #8b949e;
        line-height: 1.6;
        margin-bottom: 1rem;
        padding: 1rem;
        background: #0d1117;
        border-radius: 8px;
    }
    .paper-claim {
        font-size: 0.85rem;
        color: #58a6ff;
        font-style: italic;
        padding-left: 1rem;
        border-left: 2px solid #58a6ff;
    }

    /* Minimalist Stat Box */
    .stat-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: left;
    }
    .stat-number { 
        font-size: 2rem; 
        font-weight: 700; 
        color: #58a6ff;
        line-height: 1;
        margin-bottom: 4px;
    }
    .stat-label { 
        font-size: 0.8rem; 
        color: #8b949e; 
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section Heading */
    .section-heading {
        font-size: 1rem;
        font-weight: 600;
        color: #f0f6fc;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 2.5rem 0 1.2rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-heading::after {
        content: "";
        height: 1px;
        flex-grow: 1;
        background: #30363d;
    }

    /* Search Bar & Button */
    div[data-testid="stTextInput"] input {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #f0f6fc !important;
        border-radius: 8px !important;
    }
    div[data-testid="stButton"] > button {
        background-color: #238636 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #2ea043 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>AI Research Knowledge Graph</h1>
        <p>Autonomously synthesise research papers · Extract insights · Explore semantic relationships</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Search Settings")
    max_results = st.slider("Papers to fetch", min_value=5, max_value=20, value=12, step=1)
    similarity_threshold = st.slider(
        "Similarity threshold (edges)", min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        help="Minimum cosine similarity for drawing a 'similar_to' edge between papers",
    )
    st.divider()
    st.markdown(
        """
        **Graph Legend**
        
        🔶 **Topic** | 🔵 **Paper** | 🟢 **Concept**
        🟩 *supports* | 🟥 *contradicts* | ⬜ *related*
        
        **Timeline Legend**
        
        🔵 **Paper** | 🔶 **Concept** | ⬜ **Year**
        *introduces*, *mentions*, *extends*  
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
    search_clicked = st.button("Search")

# ── Search & pipeline ─────────────────────────────────────────────────────────
if search_clicked and topic.strip():
    progress_placeholder = st.empty()
    progress_placeholder.info("⏳ Fetching papers and running AI analysis…")

    try:
        resp = requests.get(
            f"{BACKEND_URL}/search",
            params={
                "topic": topic.strip(),
                "max_results": max_results,
                "similarity_threshold": similarity_threshold,
            },
            timeout=600,
        )
        progress_placeholder.empty()

        try:
            data = resp.json()
        except Exception:
            st.error(f"❌ Backend returned non-JSON (status {resp.status_code}).")
            st.stop()

        if resp.status_code != 200:
            err_detail = data.get("detail") or data.get("error") or str(data)
            st.error(f"❌ Backend error {resp.status_code}: {err_detail}")
            st.stop()

    except Exception as exc:
        progress_placeholder.empty()
        st.error(f"❌ Unexpected error: {exc}")
        st.stop()

    papers = data.get("papers", [])
    graph_stats = data.get("graph_stats", {})
    lit_review = data.get("literature_review", {})

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-heading">Results for: {topic}</div>', unsafe_allow_html=True)
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

    # ── Tabs for Visualizations ────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Visualizations</div>', unsafe_allow_html=True)
    
    tab_graph, tab_timeline = st.tabs(["🕸️ Semantic Knowledge Graph", "⏳ Evolution Timeline"])
    
    with tab_graph:
        graph_html_path = data.get("graph_path", "")
        if graph_html_path and os.path.exists(graph_html_path):
            with open(graph_html_path, "r", encoding="utf-8") as f:
                graph_html = f.read()
            components.html(graph_html, height=720, scrolling=False)
        else:
            try:
                gr = requests.get(f"{BACKEND_URL}/graph", timeout=10)
                if gr.status_code == 200:
                    components.html(gr.text, height=720, scrolling=False)
                else:
                    st.warning("Graph file not found.")
            except Exception:
                st.warning("Could not load the graph visualisation.")
                
    with tab_timeline:
        timeline_html_path = data.get("timeline_path", "")
        if timeline_html_path and os.path.exists(timeline_html_path):
            with open(timeline_html_path, "r", encoding="utf-8") as f:
                timeline_html = f.read()
            components.html(timeline_html, height=720, scrolling=False)
        else:
            try:
                gr = requests.get(f"{BACKEND_URL}/timeline", timeout=10)
                if gr.status_code == 200:
                    components.html(gr.text, height=720, scrolling=False)
                else:
                    st.warning("Timeline file not found. Ensure the server successfully generated it.")
            except Exception:
                st.warning("Could not load the timeline visualisation.")

    # ── AI Generated Literature Review ─────────────────────────────────────────
    if lit_review:
        st.markdown('<div class="section-heading">AI Generated Literature Review</div>', unsafe_allow_html=True)
        
        st.markdown(f"**Overview:**\n{lit_review.get('overview', '')}")
        
        st.markdown("**Key Findings:**")
        for finding in lit_review.get('key_findings', []):
            st.markdown(f"- {finding}")
            
        st.markdown("**Consensus:**")
        for consensus in lit_review.get('consensus', []):
            st.markdown(f"- {consensus}")
            
        st.markdown("**Conflicts:**")
        for conflict in lit_review.get('conflicts', []):
            st.markdown(f"- {conflict}")
            
        st.markdown(f"**Emerging Trends:**\n{lit_review.get('emerging_trends', '')}")

    # ── Paper cards ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Paper Summaries & Key Claims</div>', unsafe_allow_html=True)

    for i, paper in enumerate(papers, 1):
        authors_short = paper["authors"].split(",")[0] + (" et al." if "," in paper["authors"] else "")
        st.markdown(
            f"""
            <div class="paper-card">
                <div class="paper-title">{i}. {paper['title']}</div>
                <div class="paper-meta">
                    <span class="badge">📅 {paper['published']}</span>
                    <span class="badge">👤 {authors_short}</span>
                    <a href="{paper['url']}" target="_blank" style="color:#58a6ff; font-size:0.75rem; text-decoration:none;">🔗 arXiv</a>
                </div>
                <div class="paper-summary">
                    {paper.get('summary', 'N/A')}
                </div>
                <div class="paper-claim">
                    <strong>Key Claim:</strong> {paper.get('claim', 'N/A')}
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
        <div style="text-align:center; padding: 6rem 2rem; color: #8b949e;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
            <div style="font-size: 1.2rem; font-weight: 500; color: #f0f6fc;">Ready to explore?</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem; color: #8b949e;">
                Enter a research topic above to generate an AI-powered knowledge graph.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )