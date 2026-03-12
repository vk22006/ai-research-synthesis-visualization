from __future__ import annotations

import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

from backend.summarize import summarize_text
from backend.claim_extractor import extract_claim
from backend.embeddings import get_embedding

logger = logging.getLogger(__name__)

def summarize_papers(papers: list[dict]) -> list[dict]:
    for idx, paper in enumerate(papers):
        abstract = paper.get("abstract", "")
        if "summary" not in paper or not paper["summary"]:
            try:
                paper["summary"] = summarize_text(abstract)
            except Exception as exc:
                logger.warning(f"summarize_text failed for paper {idx}: {exc}")
                paper["summary"] = abstract[:300] + "..."
    return papers

def extract_claims(papers: list[dict]) -> list[dict]:
    for idx, paper in enumerate(papers):
        abstract = paper.get("abstract", "")
        if "claim" not in paper or not paper["claim"]:
            try:
                paper["claim"] = extract_claim(abstract)
            except Exception as exc:
                logger.warning(f"extract_claim failed for paper {idx}: {exc}")
                paper["claim"] = abstract[:200]
    return papers

def cluster_related_papers(sim_matrix: np.ndarray, distance_threshold: float = 0.5) -> dict[int, list[int]]:
    if sim_matrix.size == 0:
        return {}
    
    n_samples = sim_matrix.shape[0]
    if n_samples == 1:
        return {0: [0]}
    
    distance_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    np.fill_diagonal(distance_matrix, 0.0)
    
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            metric='precomputed', 
            linkage='average',
            distance_threshold=distance_threshold
        )
        labels = clustering.fit_predict(distance_matrix)
    except Exception as exc:
        logger.warning(f"Clustering failed: {exc}. Falling back to 1 cluster.")
        labels = [0] * n_samples

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
        
    return dict(clusters)

def detect_consensus(clusters: dict[int, list[int]], papers: list[dict]) -> list[dict]:
    
    consensus_data = []
    
    for cluster_id, paper_indices in clusters.items():
        if len(paper_indices) == 0:
            continue
            
        cluster_claims = [papers[i].get("claim", papers[i].get("summary", "")) for i in paper_indices]
        cluster_papers = [papers[i] for i in paper_indices]
        
        if len(paper_indices) == 1:
            consensus_stmt = cluster_claims[0]
        else:
            combined_claims = " ".join(cluster_claims)
            # Use summarize_text to find the consensus
            prompt = f"Synthesise these related research claims into a single concise consensus statement: {combined_claims}"
            try:
                consensus_stmt = summarize_text(prompt)
            except Exception as exc:
                logger.warning(f"Consensus generation failed for cluster {cluster_id}: {exc}")
                consensus_stmt = "Studies report related findings, though exact consensus could not be autonomously summarized."
                
        # Determine theme based on first paper
        first_paper_title = cluster_papers[0].get("title", f"Cluster {cluster_id}")
        
        consensus_data.append({
            "cluster_id": int(cluster_id),
            "paper_indices": paper_indices,
            "paper_ids": [p["id"] for p in cluster_papers],
            "topic": f"Theme related to: {first_paper_title[:50]}...",
            "statement": consensus_stmt
        })
        
    return consensus_data

def detect_conflicts(clusters: dict[int, list[int]], relationships: list[dict]) -> list[dict]:
    """
    Identify scientific disagreements utilizing the relationships list.
    """
    conflicts = []
    
    for rel in relationships:
        if rel.get("relation") == "contradicts":
            conflicts.append({
                "source": rel["source"],
                "target": rel["target"],
                "description": "Scientific disagreement identified."
            })
            
    return conflicts

def generate_literature_review(topic: str, papers: list[dict], consensus_data: list[dict], conflicts: list[dict]) -> dict:
    """
    Generate an automated literature review structured into sections.
    """
    if not papers:
        return {}
        
    overview = f"Recent research focuses on various aspects of {topic}. The analyzed literature spans {len(papers)} key papers."
    
    key_findings = [p.get("claim", p.get("summary", "")) for p in papers[:5]]
    
    consensus_statements = [c["statement"] for c in consensus_data]
    if not consensus_statements:
        consensus_statements = ["No strong consensus found amongst the analyzed papers."]
        
    try:
        sorted_papers = sorted(papers, key=lambda x: x.get("published", ""), reverse=True)
        recent_paper = sorted_papers[0] if sorted_papers else papers[0]
        emerging_trends = f"New developments indicate growing interest in areas such as: {recent_paper.get('title', '')}."
    except Exception:
        emerging_trends = "New architectures and paradigms are continuously being explored."
        
    conflict_summaries = []
    for c in conflicts[:5]:
        conflict_summaries.append(f"Disagreement identified between paper {c['source']} and {c['target']}.")
        
    if not conflict_summaries:
        conflict_summaries.append("No major scientific contradictions detected in the analyzed sample.")

    return {
        "topic": topic,
        "overview": overview,
        "key_findings": key_findings,
        "consensus": consensus_statements,
        "conflicts": conflict_summaries,
        "emerging_trends": emerging_trends
    }

