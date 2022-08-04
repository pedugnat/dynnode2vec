from typing import List

import random

import networkx as nx


def sample_nodes(graph: nx.Graph, k: int) -> List[int]:
    """
    Sample nodes randomly from a graph.
    """
    return random.sample(list(graph.nodes), k=k)


def generate_dynamic_graphs(
    n_base_nodes: int = 100, n_steps: int = 10, base_density: float = 0.01
) -> List[nx.Graph]:
    # Create a random graph
    graph = nx.fast_gnp_random_graph(n=n_base_nodes, p=base_density)

    # initialize graphs list with first graph
    graphs = [graph.copy()]

    # modify the graph randomly at each time step
    change_size = 1 + n_base_nodes // 10
    for _ in range(n_steps - 1):
        # remove some nodes
        for n in sample_nodes(graph, k=change_size):
            graph.remove_node(n)

        # add some more nodes
        node_idx = max(graph.nodes) + 1
        for i in range(2 * change_size):
            graph.add_node(node_idx + i)

        # add some edges for the new nodes
        for edge in zip(
            sample_nodes(graph, k=5 * change_size),
            sample_nodes(graph, k=5 * change_size),
        ):
            graph.add_edge(*edge)

        graphs.append(graph.copy())

    return graphs
