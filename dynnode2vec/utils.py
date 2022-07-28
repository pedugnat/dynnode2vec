import random

import networkx as nx


def generate_dynamic_graphs(n_base_nodes, n_steps, base_density):
    # Create a random graph
    graph = nx.fast_gnp_random_graph(n=n_base_nodes, p=base_density)

    # initialize graphs list with first graph
    graphs = [graph.copy()]

    # modify the graph randomly at each time step
    for _ in range(n_steps):
        # remove some nodes
        nodes_to_remove = random.sample(list(graph.nodes()), k=2)
        [graph.remove_node(n) for n in nodes_to_remove]

        # add some more nodes
        [graph.add_node(max(graph.nodes) + 1) for _ in range(5)]

        # add edges for the new nodes
        edges_to_add = zip(
            random.sample(list(graph.nodes()), k=10),
            random.sample(list(graph.nodes()), k=10),
        )
        [graph.add_edge(e1, e2) for e1, e2 in edges_to_add]

        graphs.append(graph.copy())

    return graphs
