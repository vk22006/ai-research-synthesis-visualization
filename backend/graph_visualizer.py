"""
graph_visualizer.py - Convert a NetworkX graph to an interactive PyVis HTML file.
"""
from __future__ import annotations

import os
import networkx as nx
from pyvis.network import Network


def visualize_graph(G: nx.Graph, output_path: str = "data/graph.html") -> str:
    """
    Render *G* as an interactive HTML file using PyVis.

    Parameters
    ----------
    G : nx.Graph
        The NetworkX graph produced by graph_builder.build_graph().
    output_path : str
        Relative or absolute path where the HTML file should be saved.

    Returns
    -------
    str
        Absolute path to the saved HTML file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0F1117",       # dark background to match Streamlit dark theme
        font_color="#FFFFFF",
        notebook=False,
        directed=False,
    )

    # Physics for a nice organic layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        },
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "smooth": {
          "type": "continuous"
        }
      },
      "nodes": {
        "font": {
          "size": 13,
          "face": "Inter, Arial, sans-serif"
        }
      }
    }
    """)

    # ── Add nodes from NetworkX ───────────────────────────────────────────────
    for node, attrs in G.nodes(data=True):
        net.add_node(
            str(node),
            label=attrs.get("label", str(node)),
            title=attrs.get("title", str(node)),
            color=attrs.get("color", "#888888"),
            size=attrs.get("size", 15),
            shape=attrs.get("shape", "dot"),
        )

    # ── Add edges from NetworkX ───────────────────────────────────────────────
    for u, v, attrs in G.edges(data=True):
        net.add_edge(
            str(u),
            str(v),
            title=attrs.get("title", ""),
            label=attrs.get("label", ""),
            width=attrs.get("width", 1),
            color=attrs.get("color", "#888888"),
        )

    # Save
    abs_path = os.path.abspath(output_path)
    net.save_graph(abs_path)
    return abs_path
