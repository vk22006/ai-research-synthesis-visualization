"""
app.py – FastAPI backend for the AI Research Synthesis & Knowledge Graph Builder.

Endpoints
---------
GET /search?topic=<str>&max_results=<int>&similarity_threshold=<float>
GET /graph    – Returns the latest generated graph HTML.
GET /health   – Simple health-check.
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
    for idx, paper in enumerate(papers):
        abstract = paper.get("abstract", "")
        logger.info("  [%d/%d] Summarising: %s…", idx + 1, len(papers),
                    paper["title"][:60])
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

        abstracts.append(abstract)

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

    # ── 4. Build knowledge graph ──────────────────────────────────────────────
    logger.info("Building knowledge graph…")
    try:
        G = build_graph(papers, sim_matrix, topic, similarity_threshold)
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

    # ── 6. Return ─────────────────────────────────────────────────────────────
    return JSONResponse(content={
        "topic": topic,
        "paper_count": len(papers),
        "graph_path": graph_path,
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
    })


@app.get("/graph", response_class=HTMLResponse)
def get_graph():
    """Return the latest generated knowledge graph as HTML."""
    abs_path = os.path.abspath(GRAPH_OUTPUT_PATH)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404,
                            detail="Graph not found. Call /search first.")
    with open(abs_path, "r", encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())