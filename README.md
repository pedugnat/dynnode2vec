# dynnode2vec

<div align="center">

[![Python Version](https://img.shields.io/pypi/pyversions/dynnode2vec.svg)](https://pypi.org/project/dynnode2vec/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/pedugnat/dynnode2vec/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
![Coverage Report](assets/images/coverage.svg)

</div>

<h4>

`dynnode2vec` is a package to embed dynamic graphs. 

It is the python implementation of [S. Mahdavi, S. Khoshraftar, A. An: dynnode2vec: Scalable Dynamic Network Embedding. IEEE BigData 2018](http://www.cs.yorku.ca/~aan/research/paper/dynnode2vec.pdf)

</h4>

## Installation

```bash
pip install -U dynnode2vec
```

## Usage

```python
import pickle

from dynnode2vec.dynnode2vec import DynNode2Vec
from dynnode2vec.utils import generate_dynamic_graphs

# Create random graphs
graphs = generate_dynamic_graphs(
  n_base_nodes=100, n_steps=50, base_density=0.05
)

# Instantiate dynnode2vec object
dynnode2vec = DynNode2Vec(
    p=1., 
    q=1., 
    walk_length=10, 
    n_walks_per_node=20, 
    embedding_size=64
)

# Embed the dynamic graphs
embeddings = dynnode2vec.compute_embeddings(graphs)

# Save embeddings to disk
with open('example_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
```

## Parameters
- `DynNode2Vec` class:
  - `p`: Return hyper parameter (default: 1)
  - `q`: Inout parameter (default: 1)
  - `walk_length`: Number of nodes in each walk (default: 80)
  - `n_walks_per_node`: Number of walks per node (default: 10)
  - `embedding_size`: Embedding dimensions (default: 128)
  - `seed`: Number of workers for parallel execution (default: 1)
  - `parallel_processes`: Number of workers for parallel execution (default: 1)
  - `use_delta_nodes`: Number of workers for parallel execution (default: 1)

- `DynNode2Vec.fit` method:
  - `graphs`: list of nx.Graph (ordered by time)

## TO DO 
- [x] remove pandas use in embeddings formatting
- [ ] code examples of synthetic and real-life uses
- [ ] get rid of Stellar Graph dependency


## Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/pedugnat/dynnode2vec/releases) page.

## License

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/pedugnat/dynnode2vec/blob/master/LICENSE) for more details.

## Citation

```bibtex
@misc{dynnode2vec,
  author = {Paul-Emile Dugnat},
  title = {dynnode2vec, a package to embed dynamic graphs},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/pedugnat/dynnode2vec}}
}
```

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template)
