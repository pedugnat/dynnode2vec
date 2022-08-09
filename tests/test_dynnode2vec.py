"""
Test the DynNode2Vec class
"""
# pylint: disable=missing-function-docstring

import gensim
import networkx as nx
import pytest

import dynnode2vec


@pytest.fixture(name="graphs")
def fixture_graphs():
    return dynnode2vec.utils.create_dynamic_graph(
        n_base_nodes=30, n_steps=5, base_density=0.02
    )


@pytest.fixture(name="dynnode2vec_object")
def dynnode2vec_fixture():
    return dynnode2vec.DynNode2Vec(
        n_walks_per_node=5, walk_length=5, parallel_processes=1
    )


@pytest.fixture(name="parallel_dynnode2vec_object")
def dynnode2vec_parallel_fixture():
    return dynnode2vec.DynNode2Vec(
        n_walks_per_node=5, walk_length=5, parallel_processes=2
    )


@pytest.fixture(name="node2vec_object")
def plain_node2vec_parallel_fixture():
    return dynnode2vec.DynNode2Vec(
        n_walks_per_node=5, walk_length=5, plain_node2vec=True
    )


def test_initialize_embeddings(graphs, dynnode2vec_object):
    # pylint: disable=protected-access
    init_model, init_embeddings = dynnode2vec_object._initialize_embeddings(graphs)

    assert isinstance(init_model, gensim.models.Word2Vec)
    assert isinstance(init_embeddings[0], dynnode2vec.Embedding)


def test_get_delta_nodes(graphs, dynnode2vec_object):
    current, previous = graphs[1], graphs[0]

    delta_nodes = dynnode2vec_object.get_delta_nodes(current, previous)

    assert isinstance(delta_nodes, set)
    assert delta_nodes.issubset(current.nodes)


def test_get_delta_nodes2(dynnode2vec_object):
    previous = nx.complete_graph(n=4)

    current = previous.copy()
    current.add_node(5)
    current.add_edge(0, 5)
    current.remove_edge(1, 3)

    delta_nodes = dynnode2vec_object.get_delta_nodes(current, previous)

    assert delta_nodes == {0, 1, 3, 5}


def test_generate_updated_walks(graphs, dynnode2vec_object):
    current, previous = graphs[1], graphs[0]

    updated_walks = dynnode2vec_object.generate_updated_walks(current, previous)

    assert isinstance(updated_walks, list)
    assert all(node in current.nodes for walk in updated_walks for node in walk)


def test_node2vec_generate_updated_walks(graphs, node2vec_object):
    current, previous = graphs[1], graphs[0]

    updated_walks = node2vec_object.generate_updated_walks(current, previous)

    assert isinstance(updated_walks, list)
    assert all(node in current.nodes for walk in updated_walks for node in walk)


def test_compute_embeddings(graphs, dynnode2vec_object):
    embeddings = dynnode2vec_object.compute_embeddings(graphs)

    assert isinstance(embeddings, list)
    assert all(isinstance(emb, dynnode2vec.Embedding) for emb in embeddings)


def test_parallel_compute_embeddings(graphs, parallel_dynnode2vec_object):
    embeddings = parallel_dynnode2vec_object.compute_embeddings(graphs)

    assert isinstance(embeddings, list)
    assert all(isinstance(emb, dynnode2vec.Embedding) for emb in embeddings)


def test_node2vec_compute_embeddings(graphs, node2vec_object):
    embeddings = node2vec_object.compute_embeddings(graphs)

    assert isinstance(embeddings, list)
    assert all(isinstance(emb, dynnode2vec.Embedding) for emb in embeddings)
