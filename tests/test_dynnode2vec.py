import pandas as pd
import pytest

from dynnode2vec import DynNode2Vec, utils


@pytest.fixture
def graphs():
    return generate_dynamic_graphs(n_base_nodes=100, n_steps=10, base_density=0.01)


@pytest.fixture
def dynnode2vec():
    return DynNode2Vec()


def test_initialize_embeddings(dynnode2vec, graphs):
    init_model, init_embeddings = dynnode2vec._initialize_embeddings(graphs)

    assert isinstance(init_embeddings, pd.DataFrame)


def test_find_evolving_samples(graphs):
    current, previous = graphs[1], graphs[0]

    delta_nodes = dynnode2vec.find_evolving_samples(current, previous)

    assert isinstance(delta_nodes, set)
