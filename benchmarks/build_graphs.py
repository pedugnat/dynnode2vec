"""
Build graphs from datasets.
"""
from __future__ import annotations

import gzip

import networkx as nx


def build_as_733_graphs() -> list[nx.Graph]:
    """
    Build the Autonomous systems AS-733 graphs.
    link: https://snap.stanford.edu/data/as-733.html
    """
    graphs = []
    graph = nx.Graph()
    with gzip.open("benchmarks/data/as-733.tar.gz", "rt") as stream:
        for line in stream:
            if "Autonomous systems" in line:
                if graph.nodes:
                    graphs.append(graph)
                graph.clear()
                continue
            if line[0].isdigit():
                edge = map(int, line.strip().split("\t"))
                graph.add_edge(*edge)
        graphs.append(graph)
        graphs.reverse()  # Input is in reverse chronological order
        return graphs
