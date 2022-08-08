import random

import numpy as np
import pytest

import dynnode2vec
from dynnode2vec.biased_random_walk import RandomWalks


@pytest.fixture
def graphs():
    return dynnode2vec.utils.generate_dynamic_graphs(
        n_base_nodes=30, n_steps=5, base_density=0.02
    )


def test_init(graphs):
    BRW = dynnode2vec.biased_random_walk.BiasedRandomWalk(graphs[0])

    # make sure nodes ids were converted to integers
    assert list(BRW.graph.nodes()) == list(range(BRW.graph.number_of_nodes()))


def test_weighted_choice(graphs):
    BRW = dynnode2vec.biased_random_walk.BiasedRandomWalk(graphs[0])
    rn = random.Random(0)
    eps = 0.02
    n_try = 1_000

    choices = [BRW.weighted_choice(rn, np.array([1, 4, 4])) for _ in range(n_try)]

    assert all(choice in [0, 1, 2] for choice in choices)

    assert choices.count(0) / n_try == pytest.approx(1 / 9, abs=eps)
    assert choices.count(1) / n_try == pytest.approx(4 / 9, abs=eps)
    assert choices.count(2) / n_try == pytest.approx(4 / 9, abs=eps)


@pytest.mark.parametrize("ip", [0.5, 1.0])
@pytest.mark.parametrize("iq", [1.0, 2.0])
@pytest.mark.parametrize("weighted", [True, False])
def test_generate_walk(graphs, ip, iq, weighted):
    # make sure that tested node has at least one neighbor
    G = graphs[0]
    G.add_edge(0, 1, weight=0.5)

    # add random weights to the graph for the weighted case
    if weighted:
        for _, _, w in G.edges(data=True):
            w["weight"] = random.random()

    BRW = dynnode2vec.biased_random_walk.BiasedRandomWalk(G)
    rn = random.Random(0)

    walk = BRW._generate_walk(
        node=0, walk_length=10, ip=ip, iq=iq, weighted=weighted, rn=rn
    )

    assert isinstance(walk, list)
    assert all(n in BRW.graph.nodes() for n in walk)


@pytest.mark.parametrize("p", [0.5, 1.0])
@pytest.mark.parametrize("q", [1.0, 2.0])
@pytest.mark.parametrize("weighted", [True, False])
def test_run(graphs, p, q, weighted):
    G = graphs[0]

    # add random weights to the graph for the weighted case
    if weighted:
        for _, _, w in G.edges(data=True):
            w["weight"] = random.random()

    BRW = dynnode2vec.biased_random_walk.BiasedRandomWalk(G)

    random_walks = BRW.run(p=p, q=q, weighted=weighted)

    assert all(isinstance(walk, list) for walk in random_walks)
    assert all(n in BRW.graph.nodes() for walk in random_walks for n in walk)
