"""
Build training and test sets from dynamic graphs.
"""
import networkx as nx

from dynnode2vec.dynnode2vec import DynNode2Vec, Embedding


def get_node2vec_embeddings(
    graphs: list[nx.Graph], parameters: dict
) -> list[Embedding]:
    """
    Build plain node2vec embeddings at each time step.
    """
    dynnode2vec_obj = DynNode2Vec(**parameters)
    embeddings = []
    for graph in graphs:
        _, embedding = dynnode2vec_obj.get_node2vec_embeddings(graph)
        embeddings.extend(embedding)
    return embeddings
