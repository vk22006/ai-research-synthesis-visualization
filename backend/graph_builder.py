"""
graph_builder.py - Build a NetworkX knowledge graph from enriched paper data.

Node types:
  - "topic"   : The user's search query (central hub, gold)
  - "paper"   : Each individual research paper (blue)
  - "concept" : Keywords / concepts shared between papers (green)

Edge types:
  - "related_to"  : paper → topic hub
  - "similar_to"  : paper ↔ paper (cosine similarity above threshold)
  - "mentions"    : paper → concept keyword
"""
from __future__ import annotations

import re
import networkx as nx
import numpy as np

SIMILARITY_THRESHOLD = 0.40

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "it", "its", "via", "using", "based",
    "towards", "toward", "over", "under", "new", "novel",
}


def _extract_keywords(title: str) -> list[str]:
    """Return meaningful, lowercased words from *title* (length > 3, not stopwords)."""
    words = re.sub(r"[^a-zA-Z\s]", "", title).lower().split()
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS]


def build_graph(
    papers: list[dict],
    sim_matrix: np.ndarray,
    topic: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    relationships: list[dict] = None,
    consensus_data: list[dict] = None
) -> nx.Graph:
    """
    Build and return a NetworkX graph from enriched paper data.

    Parameters
    ----------
    papers : list[dict]
        Each dict must have keys: id, title, authors, abstract, published,
        url, summary, claim.
    sim_matrix : np.ndarray
        N×N cosine similarity matrix from embeddings.py.
    topic : str
        The user's original search query string.
    similarity_threshold : float
        Minimum cosine similarity for drawing a "similar_to" edge.
    """
    G = nx.Graph()

    # ── Topic hub node ────────────────────────────────────────────────────────
    G.add_node(
        topic,
        node_type="topic",
        label=topic,
        title=f"<b>Topic:</b> {topic}",
        color="#F4C430",  # gold
        size=30,
        shape="star",
    )

    # ── Paper nodes ───────────────────────────────────────────────────────────
    for paper in papers:
        pid = paper["id"]
        short_title = (
            paper["title"][:50] + "…" if len(paper["title"]) > 50 else paper["title"]
        )
        tooltip = (
            f"<b>{paper['title']}</b><br>"
            f"<i>{paper['authors']}</i><br>"
            f"📅 {paper['published']}<br><br>"
            f"<b>Summary:</b> {paper.get('summary', '')}<br><br>"
            f"<b>Key Claim:</b> {paper.get('claim', '')}"
        )
        G.add_node(
            pid,
            node_type="paper",
            label=short_title,
            title=tooltip,
            color="#4A90D9",   # steel blue
            size=18,
            shape="dot",
            url=paper.get("url", ""),
            full_title=paper["title"],
            authors=paper["authors"],
            published=paper["published"],
        )
        # Edge: paper → topic hub
        G.add_edge(pid, topic, relation="related_to", width=1, color="#aaaaaa")

    # ── Similarity / Contradiction edges (paper ↔ paper) ──────────────────────
    if relationships is not None:
        for rel in relationships:
            rel_type = rel.get("relation", "similar_to")
            sim_score = rel.get("score") or rel.get("sim_score", 0.0)
            nli_score = rel.get("nli_score", 0.0)
            
            # Map relation to UI colors
            if rel_type == "supports":
                color = "#5CB85C"  # green
                width = 3
                title = f"Supports (Score: {nli_score:.2f})"
            elif rel_type == "contradicts":
                color = "#D9534F"  # red
                width = 3
                title = f"Contradicts (Score: {nli_score:.2f})"
            else: # similar_to or related_to
                color = "#888888"  # gray
                width = max(1, int(sim_score * 5))
                title = f"Related (Sim: {sim_score:.2f})"

            G.add_edge(
                rel["source"],
                rel["target"],
                relation=rel_type,
                label=rel_type.replace("_", " ").title(),
                title=title,
                width=width,
                color=color,
            )
    else:
        # Fallback to similarity matrix
        n = len(papers)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i][j]) if sim_matrix.size else 0.0
                if score >= similarity_threshold:
                    G.add_edge(
                        papers[i]["id"],
                        papers[j]["id"],
                        relation="similar_to",
                        label=f"{score:.2f}",
                        title=f"Similarity: {score:.2f}",
                        width=max(1, int(score * 5)),
                        color="#E07B54",   # warm orange
                    )

    # ── Concept nodes (shared keywords from titles) ───────────────────────────
    keyword_map: dict[str, list[str]] = {}  # keyword → [paper_ids]
    for paper in papers:
        for kw in _extract_keywords(paper["title"]):
            keyword_map.setdefault(kw, []).append(paper["id"])

    # Only add concept nodes shared by at least 2 papers
    for kw, pids in keyword_map.items():
        if len(pids) >= 2:
            concept_id = f"concept::{kw}"
            G.add_node(
                concept_id,
                node_type="concept",
                label=kw,
                title=f"<b>Concept:</b> {kw}<br>Mentioned by {len(pids)} papers",
                color="#5CB85C",   # green
                size=12,
                shape="diamond",
            )
            for pid in pids:
                G.add_edge(
                    pid,
                    concept_id,
                    relation="mentions",
                    width=1,
                    color="#cccccc",
                )

    # ── Consensus nodes ───────────────────────────────────────────────────────
    if consensus_data:
        for cluster in consensus_data:
            c_id = f"consensus::{cluster['cluster_id']}"
            short_topic = cluster.get('topic', 'Theme')[:40]
            G.add_node(
                c_id,
                node_type="consensus",
                label=f"Consensus:\n{short_topic}",
                title=f"<b>Consensus:</b> {cluster['statement']}",
                color="#9B59B6",  # purple
                size=22,
                shape="box",
            )
            for pid in cluster["paper_ids"]:
                G.add_edge(
                    pid,
                    c_id,
                    relation="contributes_to",
                    label="Contributes To",
                    width=2,
                    color="#D2B4DE",
                )

    return G