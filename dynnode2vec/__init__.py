"""
dynnode2vec is a package to embed dynamic graphs.
"""

import sys
from importlib import metadata as importlib_metadata

from .biased_random_walk import BiasedRandomWalk, RandomWalks
from .dynnode2vec import DynNode2Vec, Embedding
from .utils import generate_dynamic_graphs


def get_version() -> str:
    # pylint: disable=missing-function-docstring
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
