"""Embedding storage — save and load embedding vectors as .npy files.

Each embedding-generation run produces:
- embeddings.npy  — (N, dim) float32 matrix
- item_codes.json — ordered list of item codes matching rows
- metadata.json   — model info, dimensions, timestamps
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from product_image_id.embeddings.metadata import EmbeddingMetadata
from product_image_id.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


class EmbeddingStore:
    """Manages persisted embedding artifacts on disk."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embeddings_path(self) -> Path:
        return self._base_dir / "embeddings.npy"

    @property
    def item_codes_path(self) -> Path:
        return self._base_dir / "item_codes.json"

    @property
    def metadata_path(self) -> Path:
        return self._base_dir / "metadata.json"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        embeddings: NDArray[np.float32],
        item_codes: list[str],
        metadata: EmbeddingMetadata,
    ) -> None:
        """Persist embeddings, item codes, and metadata to disk."""
        np.save(self.embeddings_path, embeddings)
        self.item_codes_path.write_text(json.dumps(item_codes, indent=2))
        metadata.save(self.metadata_path)

        logger.info(
            "Saved {} embeddings (dim={}) to {}",
            len(item_codes),
            embeddings.shape[1],
            self._base_dir,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_embeddings(self) -> NDArray[np.float32]:
        """Load the embeddings matrix."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {self.embeddings_path}")
        data: NDArray[np.float32] = np.load(self.embeddings_path)
        return data

    def load_item_codes(self) -> list[str]:
        """Load the ordered list of item codes."""
        if not self.item_codes_path.exists():
            raise FileNotFoundError(f"Item codes not found: {self.item_codes_path}")
        return json.loads(self.item_codes_path.read_text())  # type: ignore[no-any-return]

    def load_metadata(self) -> EmbeddingMetadata:
        """Load the embedding metadata."""
        return EmbeddingMetadata.load(self.metadata_path)

    def exists(self) -> bool:
        """Check if all artifact files exist."""
        return (
            self.embeddings_path.exists()
            and self.item_codes_path.exists()
            and self.metadata_path.exists()
        )
