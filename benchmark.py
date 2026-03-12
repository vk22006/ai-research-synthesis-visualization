import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.fetch_papers import fetch_arxiv
from backend.summarize import summarize_text
from backend.claim_extractor import extract_claim
from backend.embeddings import compute_similarity_matrix
from backend.graph_builder import build_graph
from backend.graph_visualizer import visualize_graph
from backend.timeline_builder import build_timeline_graph, visualize_timeline
from backend.contradiction_detector import detect_contradictions
from backend.literature_review import cluster_related_papers, detect_consensus, detect_conflicts, generate_literature_review

def main():
    topic = "large language models"
    max_results = 12
    similarity_threshold = 0.40

    t0 = time.time()
    papers = fetch_arxiv(topic, max_results)
    t1 = time.time()
    print(f"fetch_arxiv: {t1 - t0:.2f}s")

    abstracts = []
    t2 = time.time()
    for p in papers:
        abstract = p.get("abstract", "")
        p["summary"] = summarize_text(abstract)
        p["claim"] = extract_claim(abstract)
        abstracts.append(abstract)
    t3 = time.time()
    print(f"summarize & extract claims sequentially: {t3 - t2:.2f}s")

    t4 = time.time()
    sim_matrix = compute_similarity_matrix(abstracts)
    t5 = time.time()
    print(f"compute_similarity_matrix: {t5 - t4:.2f}s")

    t6 = time.time()
    relationships = detect_contradictions(papers, sim_matrix, similarity_threshold)
    t7 = time.time()
    print(f"detect_contradictions: {t7 - t6:.2f}s")

    t8 = time.time()
    clusters = cluster_related_papers(sim_matrix, distance_threshold=0.6)
    consensus_data = detect_consensus(clusters, papers)
    conflicts = detect_conflicts(clusters, relationships)
    lit_review = generate_literature_review(topic, papers, consensus_data, conflicts)
    t9 = time.time()
    print(f"literature review: {t9 - t8:.2f}s")

    print(f"Total time so far: {t9 - t0:.2f}s")

if __name__ == "__main__":
    main()
