"""In-memory similarity index using NumPy.

Simple, zero-dependency (beyond numpy) cosine similarity search.
Suitable for datasets up to ~100K items. For larger datasets,
consider FAISS or Annoy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from product_image_id.index.similarity import SimilarityIndex, SimilarityMatch, cosine_similarity
from product_image_id.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class InMemoryIndex(SimilarityIndex):
    """NumPy-based in-memory cosine similarity index."""

    def __init__(self) -> None:
        self._embeddings: NDArray[np.float32] | None = None
        self._item_codes: list[str] = []

    def build(
        self,
        embeddings: NDArray[np.float32],
        item_codes: list[str],
    ) -> None:
        """Build the index from embeddings and their corresponding item codes."""
        if embeddings.shape[0] != len(item_codes):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs {len(item_codes)} item codes"
            )

        # Pre-normalize for faster search
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        self._embeddings = (embeddings / norms).astype(np.float32)
        self._item_codes = list(item_codes)
        logger.info(
            "Built in-memory index with {} items (dim={})", len(item_codes), embeddings.shape[1]
        )

    def search(
        self,
        query: NDArray[np.float32],
        top_k: int = 5,
    ) -> list[SimilarityMatch]:
        """Find the top-K most similar items to the query vector."""
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build() first.")

        scores = cosine_similarity(query, self._embeddings)

        # Get top-K indices (descending by score)
        k = min(top_k, len(self._item_codes))
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            SimilarityMatch(
                index=int(idx),
                item_code=self._item_codes[idx],
                score=float(scores[idx]),
            )
            for idx in top_indices
        ]

    def size(self) -> int:
        return len(self._item_codes)
