"""
Build graphs from datasets.
"""
import gzip

import networkx as nx


def build_as_733_graphs() -> list[nx.Graph]:
    """
    Build the Autonomous systems AS-733 graphs.
    link: https://snap.stanford.edu/data/as-733.html
    """
    graphs = []
    graph = None
    with gzip.open('data/as-733.tar.gz','rt') as stream:
        for line in stream:
            if "Autonomous systems" in line and graph:
                    graphs.append(graph)
                    graph = nx.Graph()
                    continue
            if line[0].isdigit():
                edge = line.split("\t")
                graph.add_edge(*edge)
        graphs.append(graph)
        graphs.reverse() # Input is in reverse chronological order
        return graphs