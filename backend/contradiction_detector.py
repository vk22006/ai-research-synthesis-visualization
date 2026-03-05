"""
contradiction_detector.py - Detects contradiction/support relationships between research papers.

Uses the 'facebook/bart-large-mnli' model to classify the relationship between two claims.
"""
from __future__ import annotations

import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

# Lazily initialized NLI pipeline
_nli_pipeline = None

def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        logger.info("Initializing NLI pipeline (facebook/bart-large-mnli)...")
        _nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")
    return _nli_pipeline

def detect_contradictions(papers: list[dict], sim_matrix, similarity_threshold: float) -> list[dict]:
    """
    Find pairs of papers that are similar and classify their relationship.
    
    Returns a list of dicts with keys: source_id, target_id, relation, score.
    Relations can be: 'supports', 'contradicts', 'related_to'
    """
    relationships = []
    n = len(papers)
    
    if n == 0 or sim_matrix.size == 0:
        return relationships
        
    nli_pipe = _get_nli_pipeline()
    
    for i in range(n):
        for j in range(i + 1, n):
            sim_score = float(sim_matrix[i][j])
            
            # If papers are similar enough, check their relationship
            if sim_score >= similarity_threshold:
                paper_a = papers[i]
                paper_b = papers[j]
                
                claim_a = paper_a.get("claim", paper_a.get("abstract", ""))
                claim_b = paper_b.get("claim", paper_b.get("abstract", ""))
                
                # Make sure we have enough text
                if len(claim_a) < 20 or len(claim_b) < 20:
                    relationships.append({
                        "source": paper_a["id"],
                        "target": paper_b["id"],
                        "relation": "similar_to", 
                        "score": sim_score
                    })
                    continue
                
                # NLI format: Premise string, Hypothesis string
                # We format this for text-classification pipeline that takes a single string.
                # However, for MNLI models via pipeline, it's better to use 'zero-shot-classification' or pass pairs.
                # The text-classification pipeline for MNLI expects a single string sometimes, but let's use the standard dict format or just concatenated with </s> for roberta/bart.
                # The easiest way for HuggingFace text-classification with MNLI is passing:
                # {"text": premise, "text_pair": hypothesis}
                try:
                    result = nli_pipe({"text": claim_a, "text_pair": claim_b})
                    
                    # Result usually looks like: {'label': 'contradiction', 'score': 0.99}
                    label = result["label"].lower()
                    rel_type = "similar_to"
                    
                    if label == "entailment":
                        rel_type = "supports"
                    elif label == "contradiction":
                        rel_type = "contradicts"
                    else: # neutral
                        rel_type = "related_to"
                        
                    relationships.append({
                        "source": paper_a["id"],
                        "target": paper_b["id"],
                        "relation": rel_type,
                        "nli_score": result["score"],
                        "sim_score": sim_score
                    })
                    
                except Exception as e:
                    logger.warning(f"NLI classification failed for {paper_a['id']} - {paper_b['id']}: {e}")
                    relationships.append({
                        "source": paper_a["id"],
                        "target": paper_b["id"],
                        "relation": "similar_to",
                        "score": sim_score
                    })
                    
    return relationships
