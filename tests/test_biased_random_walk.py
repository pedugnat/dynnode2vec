"""
Test the BiasedRandomWalk class
"""
# pylint: disable=missing-function-docstring
import random

import numpy as np
import pytest

import dynnode2vec


@pytest.fixture(name="graphs")
def fixture_graphs():
    return dynnode2vec.utils.create_dynamic_graph(
        n_base_nodes=30, n_steps=5, base_density=0.02
    )


def add_random_weights(graph):
    for *_, data in graph.edges(data=True):
        data["weight"] = random.random()


def test_init(graphs):
    brw = dynnode2vec.biased_random_walk.BiasedRandomWalk(graphs[0])

    # make sure nodes ids were converted to integers
    assert list(brw.graph.nodes()) == list(range(brw.graph.number_of_nodes()))


def test_weighted_choice(graphs):
    brw = dynnode2vec.biased_random_walk.BiasedRandomWalk(graphs[0])
    rng = random.Random(0)
    eps = 0.02
    n_try = 1_000

    choices = [brw.weighted_choice(rng, np.array([1, 4, 4])) for _ in range(n_try)]

    assert all(choice in [0, 1, 2] for choice in choices)

    assert choices.count(0) / n_try == pytest.approx(1 / 9, abs=eps)
    assert choices.count(1) / n_try == pytest.approx(4 / 9, abs=eps)
    assert choices.count(2) / n_try == pytest.approx(4 / 9, abs=eps)


@pytest.mark.parametrize("ip", [0.5, 1.0])
@pytest.mark.parametrize("iq", [1.0, 2.0])
@pytest.mark.parametrize("weighted", [True, False])
def test_generate_walk(graphs, ip, iq, weighted):
    # pylint: disable=invalid-name
    # make sure that tested node has at least one neighbor
    graph = graphs[0]
    graph.add_edge("0", "1")

    # add random weights to the graph for the weighted case
    if weighted:
        add_random_weights(graph)

    brw = dynnode2vec.biased_random_walk.BiasedRandomWalk(graph)
    rng = random.Random(0)

    # pylint: disable=protected-access
    walk = brw._generate_walk(
        node=0, walk_length=10, ip=ip, iq=iq, weighted=weighted, rn=rng
    )

    assert isinstance(walk, list)
    assert all(n in brw.graph.nodes() for n in walk)


@pytest.mark.parametrize("p", [0.5, 1.0])
@pytest.mark.parametrize("q", [1.0, 2.0])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("n_processes", [1, 2])
def test_run(graphs, p, q, weighted, n_processes):
    # pylint: disable=invalid-name
    graph = graphs[0]

    # add random weights to the graph for the weighted case
    if weighted:
        add_random_weights(graph)

    brw = dynnode2vec.biased_random_walk.BiasedRandomWalk(graph)

    random_walks = brw.run(
        graph.nodes(), p=p, q=q, weighted=weighted, n_processes=n_processes
    )
    assert all(isinstance(walk, list) for walk in random_walks)
    assert all(n in graph.nodes() for walk in random_walks for n in walk)
