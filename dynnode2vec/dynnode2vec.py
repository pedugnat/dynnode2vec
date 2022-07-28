"""Blabla. """
# pylint: disable=invalid-name

from typing import List, Optional

from multiprocessing import Pool

import networkx as nx
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk


class DynNode2Vec:
    """
    dynnode2vec is an algorithm that embeds dynamic graphs.

    It is heavily inspired from node2vec but uses previous
    states' embeddings with updates in order to get stable
    embeddings over time and reduce computation load.
    Source paper: http://www.cs.yorku.ca/~aan/research/paper/dynnode2vec.pdf
    """

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
        use_delta_nodes: bool = True,
    ):
        # pylint: disable=too-many-instance-attributes
        """Instantiate the DynNode2Vec object.

        :param p: Return hyper parameter (default: 1.0)
        :param q: Inout parameter (default: 1.0)
        :param walk_length: Number of nodes in each walk (default: 30)
        :param n_walks_per_node: Number of walks per node (default: 10)
        :param embedding_size: Embedding dimensions (default: 128)
        :param window: Size of the Word2Vec window around each node (default: 10)
        :param seed: Seed for the random number generators
        :param parallel_processes: Number of workers for parallel execution (default: 4)
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
        assert isinstance(use_delta_nodes, bool), "use_delta_nodes should be a boolean"

        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node
        self.embedding_size = embedding_size
        self.window = window
        self.seed = seed
        self.parallel_processes = parallel_processes
        self.use_delta_nodes = use_delta_nodes  # if False, equivalent to plain node2vec

    def _initialize_embeddings(self, graphs):
        """
        Compute normal node2vec embedding at timestep 0.
        """
        first_graph = StellarGraph.from_networkx(graphs[0])

        first_walks = BiasedRandomWalk(first_graph).run(
            nodes=first_graph.nodes(),
            length=self.walk_length,
            n=self.n_walks_per_node,
            p=self.p,
            q=self.q,
            seed=self.seed,
        )

        model = Word2Vec(
            sentences=first_walks,
            vector_size=self.embedding_size,
            window=self.window,
            min_count=0,
            sg=1,
            seed=self.seed,
            workers=self.parallel_processes,
        )

        df_embeddings = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)

        embeddings = [df_embeddings]

        return model, embeddings

    @staticmethod
    def find_evolving_samples(current_graph, previous_graph):
        """
        Find for which nodes we will have to run new walks.

        We compute the output of equation (1) of the paper, i.e.
        ∆V_t = V_add ∪ {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add ∪ E_del)}
        """

        # find edges that were either added or removed between current and previous
        added_edges = {
            n for n in current_graph.edges() if n not in previous_graph.edges()
        }
        removed_edges = {
            n for n in previous_graph.edges() if n not in current_graph.edges()
        }
        delta_edges = added_edges | removed_edges

        # find V_add ie nodes that were added
        added_nodes = {
            n for n in current_graph.nodes() if n not in previous_graph.nodes()
        }

        # find nodes which edges were modified
        modified_nodes = {
            n
            for n in current_graph.nodes()
            if any(edge in delta_edges for edge in current_graph.edges(n))
        }

        # delta nodes are either new nodes or nodes which edges changed
        delta_nodes = added_nodes | modified_nodes

        return delta_nodes

    def generate_updated_walks(self, current_graph, previous_graph):
        """
        Compute delta nodes and generate new walks for them.
        """
        if self.use_delta_nodes:
            delta_nodes = self.find_evolving_samples(current_graph, previous_graph)
        else:
            delta_nodes = current_graph.nodes()

        G = StellarGraph.from_networkx(current_graph)

        # run walks for updated nodes only
        updated_walks = BiasedRandomWalk(G).run(
            nodes=delta_nodes,
            length=self.walk_length,
            n=self.n_walks_per_node,
            p=self.p,
            q=self.q,
            seed=self.seed,
        )

        return updated_walks

    def _simulate_walks(self, graphs):
        """
        Parallelize the generation of walks on the time steps graphs.
        """
        if self.parallel_processes > 1:
            with Pool(self.parallel_processes) as p:
                return p.starmap(self.generate_updated_walks, zip(graphs[1:], graphs))

        return map(self.generate_updated_walks, zip(graphs[1:], graphs))

    @staticmethod
    def _update_embeddings(time_walks, model, embeddings):
        """
        Update sequentially the embeddings based on the first iteration.

        For each time step, we get the embeddings from the previous time step
        and update the Word2Vec model with new vocabulary and new walks.
        """
        for walks in time_walks:
            # this is the only sequential step that can not be parallelized
            # since Z_t depends on Z_t-1...Z_0

            # update word2vec model with new nodes (ie new vocabulary)
            model.build_vocab(walks, update=True)

            # update embedding by retraining the models with additional walks
            model.train(walks, total_examples=model.corpus_count, epochs=model.epochs)

            df_embeddings = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)

            embeddings.append(df_embeddings)

        return embeddings

    def compute_embeddings(self, graphs: List[nx.Graph]):
        """
        Compute dynamic embeddings on a list of graphs.
        """
        model, embeddings = self._initialize_embeddings(graphs)
        time_walks = self._simulate_walks(graphs)
        embeddings = self._update_embeddings(time_walks, model, embeddings)

        return embeddings
