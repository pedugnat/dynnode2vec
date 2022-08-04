from collections import namedtuple

import gensim
import networkx as nx
import pytest

import dynnode2vec


@pytest.fixture
def graphs():
    return dynnode2vec.utils.generate_dynamic_graphs(
        n_base_nodes=30, n_steps=5, base_density=0.02
    )


@pytest.fixture
def dynnode2vec_fixture():
    return dynnode2vec.DynNode2Vec(
        n_walks_per_node=5, walk_length=5, parallel_processes=1
    )


@pytest.fixture
def dynnode2vec_parallel_fixture():
    return dynnode2vec.DynNode2Vec(
        n_walks_per_node=5, walk_length=5, parallel_processes=2
    )


@pytest.fixture
def plain_node2vec_parallel_fixture():
    return dynnode2vec.DynNode2Vec(
        n_walks_per_node=5, walk_length=5, plain_node2vec=True
    )


def test_initialize_embeddings(graphs, dynnode2vec_fixture):
    init_model, init_embeddings = dynnode2vec_fixture._initialize_embeddings(graphs)

    assert isinstance(init_model, gensim.models.Word2Vec)
    assert isinstance(init_embeddings[0], dynnode2vec.Embedding)


def test_find_evolving_samples(graphs, dynnode2vec_fixture):
    current, previous = graphs[1], graphs[0]

    delta_nodes = dynnode2vec_fixture.get_delta_nodes(current, previous)

    assert isinstance(delta_nodes, set)
    assert delta_nodes <= current.nodes


def test_find_evolving_samples2(dynnode2vec_fixture):
    previous = nx.complete_graph(n=4)

    current = previous.copy()
    current.add_node(5)
    current.add_edge(0, 5)
    current.remove_edge(1, 3)

    delta_nodes = dynnode2vec_fixture.get_delta_nodes(current, previous)

    assert delta_nodes == {0, 1, 3, 5}


def test_generate_updated_walks(graphs, dynnode2vec_fixture):
    current, previous = graphs[1], graphs[0]

    updated_walks = dynnode2vec_fixture.generate_updated_walks(current, previous)

    assert isinstance(updated_walks, list)
    assert all(node in current.nodes() for walk in updated_walks for node in walk)


def test_node2vec_generate_updated_walks(graphs, plain_node2vec_parallel_fixture):
    current, previous = graphs[1], graphs[0]

    updated_walks = plain_node2vec_parallel_fixture.generate_updated_walks(
        current, previous
    )

    assert isinstance(updated_walks, list)
    assert all(node in current.nodes() for walk in updated_walks for node in walk)


def test_compute_embeddings(graphs, dynnode2vec_fixture):
    embeddings = dynnode2vec_fixture.compute_embeddings(graphs)

    assert isinstance(embeddings, list)
    assert all(isinstance(emb, dynnode2vec.Embedding) for emb in embeddings)


def test_parallel_compute_embeddings(graphs, dynnode2vec_parallel_fixture):
    embeddings = dynnode2vec_parallel_fixture.compute_embeddings(graphs)

    assert isinstance(embeddings, list)
    assert all(isinstance(emb, dynnode2vec.Embedding) for emb in embeddings)


def test_node2vec_compute_embeddings(graphs, plain_node2vec_parallel_fixture):
    embeddings = plain_node2vec_parallel_fixture.compute_embeddings(graphs)

    assert isinstance(embeddings, list)
    assert all(isinstance(emb, dynnode2vec.Embedding) for emb in embeddings)
