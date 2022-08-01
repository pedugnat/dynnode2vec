from collections import namedtuple

import gensim
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


def test_initialize_embeddings(graphs, dynnode2vec_fixture):
    init_model, init_embeddings = dynnode2vec_fixture._initialize_embeddings(graphs)

    assert isinstance(init_model, gensim.models.Word2Vec)
    assert isinstance(init_embeddings[0], dynnode2vec.Embedding)


def test_find_evolving_samples(graphs, dynnode2vec_fixture):
    current, previous = graphs[1], graphs[0]

    delta_nodes = dynnode2vec_fixture.find_evolving_nodes(current, previous)

    assert isinstance(delta_nodes, set)
    assert all(n in current.nodes() for n in delta_nodes)


def test_generate_updated_walks(graphs, dynnode2vec_fixture):
    current, previous = graphs[1], graphs[0]

    updated_walks = dynnode2vec_fixture.generate_updated_walks(current, previous)

    assert isinstance(updated_walks, list)
    assert all(node in current.nodes() for walk in updated_walks for node in walk)


def test_compute_embeddings(graphs, dynnode2vec_fixture):
    embeddings = dynnode2vec_fixture.compute_embeddings(graphs)

    assert isinstance(embeddings, list)
    assert all(isinstance(emb, dynnode2vec.Embedding) for emb in embeddings)
