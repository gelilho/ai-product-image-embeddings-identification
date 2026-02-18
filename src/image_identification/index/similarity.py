"""Similarity search interface and cosine similarity utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SimilarityMatch:
    """A single similarity search result."""

    index: int  # index into the catalog
    item_code: str
    score: float  # cosine similarity [0, 1]


class SimilarityIndex(ABC):
    """Abstract base class for similarity search indices."""

    @abstractmethod
    def build(
        self,
        embeddings: NDArray[np.float32],
        item_codes: list[str],
    ) -> None:
        """Build the index from a matrix of embeddings.

        Parameters
        ----------
        embeddings:
            2-D array of shape (N, dim).
        item_codes:
            List of item codes corresponding to each row.
        """

    @abstractmethod
    def search(
        self,
        query: NDArray[np.float32],
        top_k: int = 5,
    ) -> list[SimilarityMatch]:
        """Search for the top-K most similar items.

        Parameters
        ----------
        query:
            1-D query embedding of shape (dim,).
        top_k:
            Number of results to return.
        """

    @abstractmethod
    def size(self) -> int:
        """Return the number of indexed items."""


def cosine_similarity(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute cosine similarity between a query vector and a matrix.

    Parameters
    ----------
    a:
        Query vector of shape (dim,).
    b:
        Matrix of shape (N, dim).

    Returns
    -------
    NDArray[np.float32]
        Similarity scores of shape (N,), values in [-1, 1].
    """
    # Normalize
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return (b_norm @ a_norm).astype(np.float32)
