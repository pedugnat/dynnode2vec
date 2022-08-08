"""
Define a DynNode2Vec class to run dynnode2vec algorithm over dynamic graphs.
"""
# pylint: disable=invalid-name
from typing import Any, Iterable, List, Optional, Set, Tuple

from collections import namedtuple
from itertools import chain, starmap
from multiprocessing import Pool

import networkx as nx
from gensim.models import Word2Vec

from dynnode2vec.biased_random_walk import BiasedRandomWalk, RandomWalks

Embedding = namedtuple("Embedding", ["vectors", "mapping"])


class DynNode2Vec:
    """
    dynnode2vec is an algorithm to embed dynamic graphs.

    It is heavily inspired from node2vec but uses previous
    states' embeddings with updates in order to get stable
    embeddings over time and reduce computation load.
    Source paper: http://www.cs.yorku.ca/~aan/research/paper/dynnode2vec.pdf
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        *,
        p: float = 1.0,
        q: float = 1.0,
        walk_length: int = 50,
        n_walks_per_node: int = 10,
        embedding_size: int = 128,
        window: int = 10,
        seed: Optional[int] = 0,
        parallel_processes: int = 4,
        plain_node2vec: bool = False,
    ):
        """Instantiate a DynNode2Vec object.

        :param p: Return hyper parameter (default: 1.0)
        :param q: Inout parameter (default: 1.0)
        :param walk_length: Number of nodes in each walk (default: 30)
        :param n_walks_per_node: Number of walks per node (default: 10)
        :param embedding_size: Embedding dimensions (default: 128)
        :param window: Size of the Word2Vec window around each node (default: 10)
        :param seed: Seed for the random number generators (default: 0)
        :param parallel_processes: Number of workers for parallel execution (default: 4)
        :param plain_node2vec: Whether to apply simple sequential node2vec (default: False)
        """
        # argument validation
        assert isinstance(p, float) and p > 0, "p should be a strictly positive float"
        assert isinstance(q, float) and q > 0, "q should be a strictly positive float"
        assert (
            isinstance(walk_length, int) and walk_length > 0
        ), "walk_length should be a strictly positive integer"
        assert (
            isinstance(n_walks_per_node, int) and n_walks_per_node > 0
        ), "n_walks_per_node should be a strictly positive integer"
        assert (
            isinstance(embedding_size, int) and embedding_size > 0
        ), "embedding_size should be a strictly positive integer"
        assert (
            isinstance(window, int) and embedding_size > 0
        ), "window should be a strictly positive integer"
        assert (
            seed is None or isinstance(seed, int)
        ) and embedding_size > 0, "seed should be either None or int"
        assert (
            isinstance(parallel_processes, int) and 0 < parallel_processes < 128
        ), "parallel_processes should be a strictly positive integer"
        assert isinstance(plain_node2vec, bool), "plain_node2vec should be a boolean"

        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node
        self.embedding_size = embedding_size
        self.window = window
        self.seed = seed
        self.parallel_processes = parallel_processes
        self.plain_node2vec = plain_node2vec

        # see https://stackoverflow.com/questions/53417258/what-is-workers-parameter-in-word2vec-in-nlp  # pylint: disable=line-too-long
        self.gensim_workers = max(self.parallel_processes - 1, 12)

    def _initialize_embeddings(
        self, graphs: List[nx.Graph]
    ) -> Tuple[Word2Vec, List[Embedding]]:
        """
        Compute normal node2vec embedding at timestep 0.
        """
        first_graph = graphs[0]

        first_walks = BiasedRandomWalk(first_graph).run(
            nodes=first_graph.nodes(),
            walk_length=self.walk_length,
            n_walks=self.n_walks_per_node,
            p=self.p,
            q=self.q,
        )

        model = Word2Vec(
            sentences=first_walks,
            vector_size=self.embedding_size,
            window=self.window,
            min_count=0,
            sg=1,
            seed=self.seed,
            workers=self.gensim_workers,
        )

        embedding = Embedding(model.wv.vectors.copy(), model.wv.index_to_key.copy())

        return model, [embedding]

    @staticmethod
    def get_delta_nodes(current_graph: nx.Graph, previous_graph: nx.Graph) -> Set[Any]:
        """
        Find nodes in the current graph which have been modified, i.e. they have been added,
        or at least one of their edge have been updated.
        This is the subset of nodes for which we will generate new random walks.

        We compute the output of equation (1) of the paper, i.e.
        ∆V_t = V_add ∪ {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add ∪ E_del)}

        We make the assumption about V_add that we only care about nodes that are connected
        to at least one other node, i.e. nodes that have at least one edge.
        This assumption yields that V_add ⊆ {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add ∪ E_del)},
        which we can use to avoid computing V_add.
        """
        # find edges that were either added or removed between current and previous
        delta_edges = current_graph.edges ^ previous_graph.edges

        # find nodes in the current graph which edges have been updated
        # = {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add ∪ E_del)}
        nodes_with_modified_edges = set(chain(*delta_edges))

        # Delta nodes are new nodes (V_add) and current nodes which edges have changed.
        # Since we only care about nodes that have at least one edge, we can
        # assume that V_add ⊆ {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add ∪ E_del)}
        delta_nodes: Set[Any] = current_graph.nodes & nodes_with_modified_edges

        return delta_nodes

    def generate_updated_walks(
        self, current_graph: nx.Graph, previous_graph: nx.Graph
    ) -> RandomWalks:
        """
        Compute delta nodes and generate new walks for them.
        """
        if self.plain_node2vec:
            # if we stick to node2vec implementation, we sample walks
            # for all nodes at each time step
            delta_nodes = current_graph.nodes()
        else:
            # if we use dynnode2vec, we sample walks only for nodes
            # that changed compared to the previous time step
            delta_nodes = self.get_delta_nodes(current_graph, previous_graph)

        brw = BiasedRandomWalk(current_graph)
        delta_nodes = brw.convert_true_ids_to_int_ids(delta_nodes)

        # run walks for updated nodes only
        updated_walks = brw.run(
            nodes=delta_nodes,
            walk_length=self.walk_length,
            n_walks=self.n_walks_per_node,
            p=self.p,
            q=self.q,
        )

        return updated_walks

    def _simulate_walks(self, graphs: List[nx.Graph]) -> Iterable[RandomWalks]:
        """
        Parallelize the generation of walks on the time steps graphs.
        """
        if self.parallel_processes > 1:
            with Pool(self.parallel_processes) as p:
                return p.starmap(self.generate_updated_walks, zip(graphs[1:], graphs))

        return starmap(self.generate_updated_walks, zip(graphs[1:], graphs))

    def _update_embeddings(
        self,
        embeddings: List[Embedding],
        time_walks: Iterable[RandomWalks],
        model: Word2Vec,
    ) -> None:
        """
        Update sequentially the embeddings based on the first iteration.

        For each time step, we get the embeddings from the previous time step
        and update the Word2Vec model with new vocabulary and new walks.
        """
        for walks in time_walks:
            # this is the only sequential step that cannot be parallelized
            # since Z_t depends on Z_t-1...Z_0

            if self.plain_node2vec:
                # if we stick to plain node2vec, we reinitialize word2vec
                # model at each time step
                model = Word2Vec(
                    sentences=walks,
                    vector_size=self.embedding_size,
                    window=self.window,
                    min_count=0,
                    sg=1,
                    seed=self.seed,
                    workers=self.gensim_workers,
                )

            else:
                # update word2vec model with new nodes (i.e. new vocabulary)
                model.build_vocab(walks, update=True)

                # update embedding by retraining the model with additional walks
                model.train(
                    walks, total_examples=model.corpus_count, epochs=model.epochs
                )

            embedding = Embedding(model.wv.vectors.copy(), model.wv.index_to_key.copy())

            embeddings.append(embedding)

    def compute_embeddings(self, graphs: List[nx.Graph]) -> List[Embedding]:
        """
        Compute dynamic embeddings on a list of graphs.
        """
        # TO DO : check graph weights valid
        model, embeddings = self._initialize_embeddings(graphs)
        time_walks = self._simulate_walks(graphs)
        self._update_embeddings(embeddings, time_walks, model)

        return embeddings
