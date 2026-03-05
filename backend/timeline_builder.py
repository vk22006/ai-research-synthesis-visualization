"""
timeline_builder.py - Build a chronological NetworkX timeline graph and render with PyVis.

Extracts concepts using TF-IDF and groups papers by year to show how ideas evolve.
"""
from __future__ import annotations

import os
import re
import networkx as nx
import numpy as np
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "it", "its", "via", "using", "based",
    "towards", "toward", "over", "under", "new", "novel", "we", "our",
    "model", "approach", "method", "system", "can", "which", "paper", "data",
    "results", "show", "propose", "performance", "task", "tasks", "models", 
    "methods", "stateofart", "also", "two", "one", "learning"
}

def extract_concepts(papers: list[dict], top_k: int = 2) -> dict[str, list[str]]:
    """
    Given a list of papers, extract the top concepts for each using TF-IDF on abstracts.
    Returns: dict mapping paper id to list of concepts.
    """
    if not papers:
        return {}
        
    abstracts = [p.get("abstract", "") for p in papers]
    
    # Simple regex tokenizer that cleans non-letters
    def tokenizer(text):
        words = re.sub(r"[^a-zA-Z\s]", "", text).lower().split()
        return [w for w in words if len(w) > 3 and w not in _STOPWORDS]
        
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=list(_STOPWORDS), max_df=0.85, min_df=1)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(abstracts)
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        # Fallback if vocabulary is empty
        return {p["id"]: [] for p in papers}

    paper_concepts = {}
    for i, paper in enumerate(papers):
        # Get TF-IDF scores for this paper
        row = tfidf_matrix.getrow(i).toarray()[0]
        # Get indices of top k scores
        top_indices = row.argsort()[-top_k:][::-1]
        
        concepts = []
        for idx in top_indices:
            if row[idx] > 0: # Only include if score > 0
                concepts.append(feature_names[idx].title())
                
        paper_concepts[paper["id"]] = concepts
        
    return paper_concepts

def group_papers_by_year(papers: list[dict]) -> dict[int, list[dict]]:
    """Groups papers by their publication year."""
    by_year = {}
    for paper in papers:
        # Expected format: YYYY-MM-DD or similar
        pub_date = paper.get("published", "")
        year_match = re.search(r'\b(19|20)\d{2}\b', pub_date)
        
        if year_match:
            year = int(year_match.group(0))
        else:
            year = 2026 # Default fallback
            
        by_year.setdefault(year, []).append(paper)
        
    return by_year

def build_timeline_graph(papers: list[dict]) -> nx.DiGraph:
    """Build a directed chronological timeline graph."""
    G = nx.DiGraph()
    if not papers:
        return G
        
    # 1. Group by year
    by_year = group_papers_by_year(papers)
    years = sorted(list(by_year.keys()))
    
    # 2. Extract concepts
    paper_concepts = extract_concepts(papers, top_k=2)
    
    # Keep track of concept first appearance to draw evolution
    concept_history = {}
    
    # Create Year nodes and skeleton
    for i, year in enumerate(years):
        year_node = f"year_{year}"
        G.add_node(
            year_node,
            label=str(year),
            title=f"Year: {year}",
            color={"background": "#21262d", "border": "#8b949e"}, # Darker gray with bright border
            size=40,
            shape="box",
            level=year,
            font={"size": 18, "bold": True, "color": "#f0f6fc"} # Larger font
        )
        
        # Link consecutive years (The Timeline Axis)
        if i > 0:
            prev_year_node = f"year_{years[i-1]}"
            G.add_edge(prev_year_node, year_node, color="#58a6ff", width=4, title="Timeline progression")
            
        # Add papers for this year
        for paper in by_year[year]:
            pid = paper["id"]
            short_title = paper["title"][:50] + "…" if len(paper["title"]) > 50 else paper["title"]
            tooltip = (
                f"<b>{paper['title']}</b><br>"
                f"<i>{paper['authors']}</i><br>"
                f"📅 {paper['published']}<br><br>"
                f"<b>Summary:</b> {paper.get('summary', '')}"
            )
            
            G.add_node(
                pid,
                label=short_title,
                title=tooltip,
                color={"background": "#4A90D9", "border": "#316091"}, # Blue
                size=25,
                shape="dot",
                level=year
            )
            
            # Link paper to its year
            G.add_edge(year_node, pid, color="#8b949e", width=2, style="dashed")
            
            # Map Concepts
            concepts = paper_concepts.get(pid, [])
            for concept in concepts:
                concept_id = f"concept_{concept}"
                
                # If concept doesn't exist, create it
                if not G.has_node(concept_id):
                    G.add_node(
                        concept_id,
                        label=concept,
                        title=f"<b>Concept:</b> {concept}",
                        color="#E07B54", # Orange
                        size=25,
                        shape="diamond",
                        level=year # Assign to the year it first appears
                    )
                    G.add_edge(pid, concept_id, label="introduces", color="#E07B54", width=2)
                    concept_history[concept] = year
                else:
                    # Concept already exists, link this paper to the existing concept
                    first_year = concept_history[concept]
                    if year > first_year:
                        edge_label = "extends"
                    else:
                        edge_label = "mentions"
                        
                    G.add_edge(pid, concept_id, label=edge_label, color="#F0A880", width=1.5)
                    
    return G

def visualize_timeline(G: nx.DiGraph, output_path: str = "data/timeline.html") -> str:
    """Render PyVis graph with hierarchical Left-to-Right layout."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0F1117", 
        font_color="#FFFFFF",
        notebook=False,
        directed=True,
        layout=True # Enable layout engine
    )
    
    # We use a hierarchical layout based on the 'level' attribute (year)
    # We turn on physics specifically for hierarchical repulsion so nodes don't overlap vertically
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "nodeSpacing": 150,
          "treeSpacing": 250,
          "levelSeparation": 350
        }
      },
      "physics": {
        "enabled": true,
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 150,
          "springConstant": 0.01,
          "nodeDistance": 200,
          "damping": 0.09
        },
        "solver": "hierarchicalRepulsion"
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": true,
        "zoomView": true
      },
      "edges": {
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal",
          "roundness": 0.5
        }
      },
      "nodes": {
        "font": {
          "size": 14,
          "face": "Inter, Arial, sans-serif",
          "color": "#FFFFFF"
        },
        "borderWidth": 2,
        "borderWidthSelected": 4
      }
    }
    """)
    
    # Translate NetworkX graph to PyVis
    for node, attrs in G.nodes(data=True):
        net.add_node(
            str(node),
            label=attrs.get("label", str(node)),
            title=attrs.get("title", str(node)),
            color=attrs.get("color"),
            size=attrs.get("size", 15),
            shape=attrs.get("shape", "dot"),
            level=attrs.get("level", 0) # Essential for hierarchical layout
        )
        
    for u, v, attrs in G.edges(data=True):
        # Dashed style support for PyVis (hacky way via JS mapping but pyvis has 'dashes' prop)
        dashes = True if attrs.get("style") == "dashed" else False
        net.add_edge(
            str(u),
            str(v),
            title=attrs.get("title", ""),
            label=attrs.get("label", ""),
            width=attrs.get("width", 1),
            color=attrs.get("color"),
            dashes=dashes
        )
        
    abs_path = os.path.abspath(output_path)
    net.save_graph(abs_path)
    return abs_path
