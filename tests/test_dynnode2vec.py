import gensim
import pandas as pd
import pytest

from dynnode2vec import DynNode2Vec, utils


@pytest.fixture
def graphs():
    return utils.generate_dynamic_graphs(n_base_nodes=30, n_steps=5, base_density=0.02)


@pytest.fixture
def dynnode2vec():
    return DynNode2Vec(n_walks_per_node=5, walk_length=5, parallel_processes=1)


def test_initialize_embeddings(graphs, dynnode2vec):
    init_model, init_embeddings = dynnode2vec._initialize_embeddings(graphs)

    assert isinstance(init_embeddings[0], pd.DataFrame)
    assert isinstance(init_model, gensim.models.Word2Vec)


def test_find_evolving_samples(graphs, dynnode2vec):
    current, previous = graphs[1], graphs[0]

    delta_nodes = dynnode2vec.find_evolving_samples(current, previous)

    assert isinstance(delta_nodes, set)


def test_generate_updated_walks(graphs, dynnode2vec):
    current, previous = graphs[1], graphs[0]

    updated_walks = dynnode2vec.generate_updated_walks(current, previous)

    assert isinstance(updated_walks, list)


def test_compute_embeddings(graphs, dynnode2vec):
    embeddings = dynnode2vec.compute_embeddings(graphs)

    assert isinstance(embeddings, list)
