from typing import Any, List, Union

import bisect
import random
from functools import partial

import networkx as nx
import numpy as np

RandomWalks = List[List[Any]]


class BiasedRandomWalk:
    """
    Performs biased second order random walks (like those used in Node2Vec algorithm)
    controlled by the values of two parameters p (return parameter) and q (in-out parameter).
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Instantiate a BiasedRandomWalk object.

        :param graph: graph to run walk on
        """
        self.graph = nx.convert_node_labels_to_integers(
            graph, ordering="sorted", label_attribute="true_label"
        )

    def map_int_ids_to_true_ids(self, walks: RandomWalks) -> None:
        # map back integers id to true node id
        mapping = nx.get_node_attributes(self.graph, "true_label")

        # inplace replace walks of integer ids by true ids
        for i in range(len(walks)):
            walks[i] = list(map(mapping.get, walks[i]))

    @staticmethod
    def weighted_choice(rn: random.Random, weights: Any) -> int:
        """
        Choose a random index in an array, based on weights.

        This method is fastest than built-in numpy functions like `numpy.random.choice`
        or `numpy.random.multinomial`.
        See https://stackoverflow.com/questions/24140114/fast-way-to-obtain-a-random-index-from-an-array-of-weights-in-python

        Example: for array [1, 4, 4], index 0 will be chosen with probabilty 1/9,
        index 1 and index 2 will be chosen with probability 4/9.
        """
        probs = np.cumsum(weights)
        total = probs[-1]

        return bisect.bisect(probs, rn.random() * total)

    def _generate_walk(
        self,
        node: int,
        walk_length: int,
        ip: float,
        iq: float,
        weighted: bool,
        rn: random.Random,
    ) -> List[int]:
        """
        Generate a number of random walks starting from a given node.
        """
        # the walk starts at the root
        walk = [node]

        previous_node = None
        previous_node_neighbours: Any = []

        current_node = node

        if self.graph.degree[node] == 0:
            # the starting node has no neighbor, so we return
            return walk

        for _ in range(walk_length - 1):
            # select one of the neighbours using the
            # appropriate transition probabilities
            if weighted:
                edges_data = self.graph.edges(current_node, data="weight")

                # edges_data is a list of triplets (current_node, out_node, weight)
                neighbours = np.array([e[1] for e in edges_data])
                weights = np.array([e[2] for e in edges_data])
            else:
                neighbours = np.array(list(self.graph.neighbors(current_node)))
                weights = np.ones(neighbours.shape)

            if (ip != 1.0) or (iq != 1.0):
                # we update the weights according to return (p) and in-out (q)
                # parameters
                mask = neighbours == previous_node
                weights[mask] *= ip
                mask |= np.isin(neighbours, previous_node_neighbours)
                weights[~mask] *= iq

            choice = self.weighted_choice(rn, weights)

            previous_node = current_node
            previous_node_neighbours = neighbours
            current_node = neighbours[choice]

            walk.append(current_node)

        return walk

    def run(
        self,
        n_walks: int = 10,
        walk_length: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        weighted: bool = False,
        seed: Union[int, None] = None,
    ) -> RandomWalks:
        """
        Perform a number of random walks for all the nodes of the graph. The
        behavior of the random walk is mainly conditioned by two parameters p and q.
        """
        rn = random.Random(seed)

        # weights are multiplied by inverse p and q
        ip, iq = 1.0 / p, 1.0 / q

        generate_walk = partial(
            self._generate_walk,
            walk_length=walk_length,
            ip=ip,
            iq=iq,
            weighted=weighted,
            rn=rn,
        )

        walks = [
            generate_walk(node) for node in self.graph.nodes() for _ in range(n_walks)
        ]

        # we map back the integer ids (used for speed) to the original node ids
        self.map_int_ids_to_true_ids(walks)

        return walks
