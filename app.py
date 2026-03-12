"""
app.py - FastAPI backend for the AI Research Synthesis & Knowledge Graph Builder.

Endpoints
---------
GET /search?topic=<str>&max_results=<int>&similarity_threshold=<float>
GET /graph    - Returns the latest generated graph HTML.
GET /health   - Simple health-check.
"""
from __future__ import annotations

import os
import sys
import logging
import traceback

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse as StarletteJSONResponse
from concurrent.futures import ThreadPoolExecutor

from backend.fetch_papers import fetch_arxiv
from backend.summarize import summarize_text
from backend.claim_extractor import extract_claim
from backend.embeddings import compute_similarity_matrix
from backend.graph_builder import build_graph
from backend.graph_visualizer import visualize_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research Synthesis API",
    description="Fetch arXiv papers, summarise, build a knowledge graph, visualise.",
    version="1.1.0",
)

# Allow the Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH_OUTPUT_PATH = "data/graph.html"
SEARCH_CACHE = {}

@app.on_event("startup")
async def startup_event():
    logger.info("Warming up SentenceTransformer model...")
    try:
        from backend.embeddings import _get_model
        _get_model()
        logger.info("Model warmup complete.")
    except Exception as exc:
        logger.warning(f"Model warmup failed: {exc}")


# ── Global error handler: always return JSON, never plain text ────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return StarletteJSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return StarletteJSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.1.0"}


@app.get("/search")
def search(
    topic: str = Query(..., description="Research topic to search on arXiv"),
    max_results: int = Query(12, ge=3, le=20),
    similarity_threshold: float = Query(0.40, ge=0.0, le=1.0),
):
    """Fetch papers, enrich with AI summaries/claims, build and save the graph."""

    cache_key = f"{topic.lower()}_{max_results}_{similarity_threshold}"
    if cache_key in SEARCH_CACHE:
        logger.info("Cache hit for: %s", cache_key)
        return JSONResponse(content=SEARCH_CACHE[cache_key])

    logger.info("Search: topic='%s', max_results=%d, threshold=%.2f",
                topic, max_results, similarity_threshold)

    # ── 1. Fetch papers ───────────────────────────────────────────────────────
    try:
        papers = fetch_arxiv(topic, max_results)
    except Exception as exc:
        logger.error("fetch_arxiv failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"arXiv fetch failed: {exc}")

    if not papers:
        raise HTTPException(status_code=404,
                            detail=f"No papers found for topic: '{topic}'")

    logger.info("Fetched %d papers. Enriching…", len(papers))

    # ── 2. Summarise + extract claims ─────────────────────────────────────────
    abstracts: list[str] = []
    
    def process_paper(idx, paper):
        abstract = paper.get("abstract", "")
        logger.info("  [%d/%d] Processing: %s…", idx + 1, len(papers), paper["title"][:60])
        try:
            paper["summary"] = summarize_text(abstract)
        except Exception as exc:
            logger.warning("  summarize_text failed for paper %d: %s", idx, exc)
            paper["summary"] = abstract[:300] + "…"

        try:
            paper["claim"] = extract_claim(abstract)
        except Exception as exc:
            logger.warning("  extract_claim failed for paper %d: %s", idx, exc)
            paper["claim"] = abstract[:200]
        return abstract

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for idx, paper in enumerate(papers):
            futures.append(executor.submit(process_paper, idx, paper))
        
        for future in futures:
            abstracts.append(future.result())

    # ── 3. Similarity matrix ──────────────────────────────────────────────────
    logger.info("Computing similarity matrix…")
    try:
        import numpy as np
        sim_matrix = compute_similarity_matrix(abstracts)
    except Exception as exc:
        logger.warning("Similarity matrix failed: %s. Using zeros.", exc)
        import numpy as np
        n = len(papers)
        sim_matrix = np.zeros((n, n))

    # ── 3.5. Contradiction Detection ──────────────────────────────────────────
    logger.info("Running contradiction detection (%d papers)…", len(papers))
    try:
        from backend.contradiction_detector import detect_contradictions
        relationships = detect_contradictions(papers, sim_matrix, similarity_threshold)
    except Exception as exc:
        logger.warning("Contradiction detection failed: %s", exc)
        relationships = []

    # ── 3.6 Automated Literature Review & Consensus Analyzer ──────────────────
    logger.info("Generating literature review…")
    try:
        from backend.literature_review import cluster_related_papers, detect_consensus, detect_conflicts, generate_literature_review
        clusters = cluster_related_papers(sim_matrix, distance_threshold=0.6)
        consensus_data = detect_consensus(clusters, papers)
        conflicts = detect_conflicts(clusters, relationships)
        lit_review = generate_literature_review(topic, papers, consensus_data, conflicts)
    except Exception as exc:
        logger.warning("Literature review generation failed: %s", exc)
        consensus_data = []
        conflicts = []
        lit_review = {}

    # ── 4. Build knowledge graph ──────────────────────────────────────────────
    logger.info("Building knowledge graph…")
    try:
        G = build_graph(papers, sim_matrix, topic, similarity_threshold, relationships, consensus_data)
    except Exception as exc:
        logger.error("build_graph failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Graph build failed: {exc}")

    # ── 5. Visualise → HTML ───────────────────────────────────────────────────
    logger.info("Visualising graph (%d nodes, %d edges)…",
                G.number_of_nodes(), G.number_of_edges())
    try:
        graph_path = visualize_graph(G, GRAPH_OUTPUT_PATH)
    except Exception as exc:
        logger.error("visualize_graph failed: %s", exc)
        graph_path = ""

    # ── 6. Build Timeline Graph ───────────────────────────────────────────────
    logger.info("Building timeline graph…")
    TIMELINE_OUTPUT_PATH = "data/timeline.html"
    try:
        from backend.timeline_builder import build_timeline_graph, visualize_timeline
        T = build_timeline_graph(papers)
        timeline_path = visualize_timeline(T, TIMELINE_OUTPUT_PATH)
    except Exception as exc:
        logger.error("timeline build failed: %s", exc)
        timeline_path = ""

    # ── 7. Return ─────────────────────────────────────────────────────────────
    response_content = {
        "topic": topic,
        "paper_count": len(papers),
        "graph_path": graph_path,
        "timeline_path": timeline_path,
        "literature_review": lit_review,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "papers": [
            {
                "id": p["id"],
                "title": p["title"],
                "authors": p["authors"],
                "published": p["published"],
                "url": p["url"],
                "abstract": p["abstract"],
                "summary": p.get("summary", ""),
                "claim": p.get("claim", ""),
            }
            for p in papers
        ],
    }
    
    SEARCH_CACHE[cache_key] = response_content
    return JSONResponse(content=response_content)


@app.get("/graph", response_class=HTMLResponse)
def get_graph():
    """Return the latest generated knowledge graph as HTML."""
    abs_path = os.path.abspath(GRAPH_OUTPUT_PATH)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404,
                            detail="Graph not found. Call /search first.")
    with open(abs_path, "r", encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())


@app.get("/timeline", response_class=HTMLResponse)
def get_timeline():
    """Return the latest generated timeline graph as HTML."""
    abs_path = os.path.abspath("data/timeline.html")
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404,
                            detail="Timeline not found. Call /search first.")
    with open(abs_path, "r", encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())