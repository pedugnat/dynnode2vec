"""
Utility file to define miscellaneous functions.
"""
from __future__ import annotations

import random

import networkx as nx


def sample_nodes(graph: nx.Graph, k: int) -> list[int]:
    """
    Samples nodes randomly from a graph.
    """
    return random.sample(graph.nodes, k=k)


def create_dynamic_graph(
    n_base_nodes: int = 100, n_steps: int = 10, base_density: float = 0.01
) -> list[nx.Graph]:
    """
    Creates a list of graphs representing the evolution of a dynamic graph,
    i.e. graphs that each depend on the previous graph.
    """
    # Create a random graph
    graph = nx.fast_gnp_random_graph(n=n_base_nodes, p=base_density)

    # add one to each node to avoid the perfect case where true_ids match int_ids
    graph = nx.relabel_nodes(graph, mapping={n: str(n) for n in graph.nodes()})

    # initialize graphs list with first graph
    graphs = [graph.copy()]

    # modify the graph randomly at each time step
    change_size = 1 + n_base_nodes // 10
    for _ in range(n_steps - 1):
        # remove some nodes
        for node in sample_nodes(graph, k=change_size):
            graph.remove_node(node)

        # add some more nodes
        node_idx = max(map(int, graph.nodes)) + 1
        for i in range(2 * change_size):
            graph.add_node(str(node_idx + i))

        # add some edges for the new nodes
        for edge in zip(
            sample_nodes(graph, k=5 * change_size),
            sample_nodes(graph, k=5 * change_size),
        ):
            graph.add_edge(*edge)

        graphs.append(graph.copy())

    return graphs
