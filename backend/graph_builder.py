import networkx as nx

def build_graph(papers):
    G = nx.Graph()

    for paper in papers:
        G.add_node(paper["title"])

    return G