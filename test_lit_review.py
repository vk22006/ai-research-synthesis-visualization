import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.fetch_papers import fetch_arxiv
from backend.embeddings import compute_similarity_matrix
from backend.literature_review import cluster_related_papers, detect_consensus, detect_conflicts, generate_literature_review

print("Fetching papers...")
papers = fetch_arxiv("Graph Neural Networks", max_results=5)

abstracts = []
for p in papers:
    p["claim"] = p["abstract"][:100]
    p["summary"] = p["abstract"][:100]
    abstracts.append(p["abstract"])

print("Computing similarities...")
sim_matrix = compute_similarity_matrix(abstracts)

print("Clustering...")
clusters = cluster_related_papers(sim_matrix, distance_threshold=0.6)

print("Detecting consensus...")
consensus_data = detect_consensus(clusters, papers)

print("Formatting lit review...")
lit_review = generate_literature_review("Graph Neural Networks", papers, consensus_data, [])

print("Result:")
print(lit_review)
print("Finished successfully!")
