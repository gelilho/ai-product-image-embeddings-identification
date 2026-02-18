"""Pluggable embedding backend interface (Strategy pattern).

To add a new backend:
1. Create a new module (e.g. siglip_backend.py)
2. Implement the EmbeddingBackend ABC
3. Register it in the BACKEND_REGISTRY

Usage:
    backend = get_backend("marqo", model_name="...", device="cpu")
    vector = backend.embed_image(pil_image)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image


class EmbeddingBackend(ABC):
    """Abstract base class for all embedding backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Full model identifier (e.g. HuggingFace model name)."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model version string for metadata tracking."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory.

        Called once before first use. May download model files.
        """

    @abstractmethod
    def embed_image(self, image: Image.Image) -> NDArray[np.float32]:
        """Generate an embedding vector for a single PIL image.

        Parameters
        ----------
        image:
            A PIL Image (RGB mode expected).

        Returns
        -------
        NDArray[np.float32]
            1-D embedding vector of shape (embedding_dim,).
        """

    def embed_images(self, images: list[Image.Image]) -> NDArray[np.float32]:
        """Generate embeddings for a batch of images.

        Default implementation processes sequentially. Override for
        batch-optimized backends.

        Returns
        -------
        NDArray[np.float32]
            2-D array of shape (N, embedding_dim).
        """
        vectors = [self.embed_image(img) for img in images]
        return np.stack(vectors)

    def embed_text(self, texts: str | list[str]) -> NDArray[np.float32]:
        """Generate embedding(s) for text prompt(s).

        Used for zero-shot classification (e.g. comparing a query image
        against text prompts like "a shoe", "apparel clothing").

        Parameters
        ----------
        texts:
            A single text string or a list of text strings.

        Returns
        -------
        NDArray[np.float32]
            2-D array of shape (N, embedding_dim) where N is the number
            of texts, or 1-D of shape (embedding_dim,) for a single text.

        Raises
        ------
        NotImplementedError
            If the backend does not support text embedding.
        """
        raise NotImplementedError(f"{self.name} backend does not support text embedding")

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready."""


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type[EmbeddingBackend]] = {}


def register_backend(key: str, cls: type[EmbeddingBackend]) -> None:
    """Register an embedding backend class by key."""
    _BACKEND_REGISTRY[key] = cls


def get_backend(
    key: str,
    *,
    model_name: str | None = None,
    device: str = "",
) -> EmbeddingBackend:
    """Instantiate a registered backend by key.

    Raises KeyError if the backend is not registered.
    """
    if key not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise KeyError(f"Unknown backend '{key}'. Available: {available}")

    cls = _BACKEND_REGISTRY[key]
    # Pass kwargs that the constructor accepts
    kwargs: dict[str, str] = {}
    if model_name:
        kwargs["model_name"] = model_name
    if device:
        kwargs["device"] = device
    return cls(**kwargs)  # type: ignore[call-arg]


def list_backends() -> list[str]:
    """Return all registered backend keys."""
    return sorted(_BACKEND_REGISTRY.keys())
