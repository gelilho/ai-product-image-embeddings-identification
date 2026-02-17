"""Marqo-Ecommerce-L embedding backend.

Uses the open_clip library to load the Marqo e-commerce model.
This model is purpose-built for product image retrieval and achieves
SOTA results on e-commerce benchmarks.

Model: Marqo/marqo-ecommerce-embeddings-L
Dims:  1024
License: Apache 2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from product_image_id.embeddings.interface import EmbeddingBackend, register_backend
from product_image_id.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image

_DEFAULT_MODEL = "Marqo/marqo-ecommerce-embeddings-L"
_DEFAULT_PRETRAINED = "marqo-ecommerce-embeddings-L"
_EMBEDDING_DIM = 1024


class MarqoEcommerceBackend(EmbeddingBackend):
    """Marqo E-Commerce L embedding backend via open_clip."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "",
    ) -> None:
        self._model_name = model_name
        self._requested_device = device
        self._device: torch.device | None = None
        self._model: object | None = None  # open_clip model
        self._preprocess: object | None = None
        self._tokenizer: object | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "marqo"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return _DEFAULT_PRETRAINED

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the Marqo model weights. Downloads on first run."""
        import open_clip

        if self._loaded:
            return

        self._device = self._resolve_device()
        logger.info(
            "Loading Marqo embedding model '{}' on {}",
            self._model_name,
            self._device,
        )

        # Load via HuggingFace Hub using the hf-hub: prefix
        hf_model_id = f"hf-hub:{self._model_name}"
        model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            hf_model_id,
        )
        model = model.to(self._device)  # type: ignore[union-attr]
        model.eval()  # type: ignore[union-attr]

        self._model = model
        self._preprocess = preprocess_val
        self._tokenizer = open_clip.get_tokenizer(hf_model_id)
        self._loaded = True

        logger.info("Marqo backend loaded (dim={})", self.embedding_dim)

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_image(self, image: Image.Image) -> NDArray[np.float32]:
        """Generate embedding for a single image."""
        if not self._loaded:
            self.load()

        image_rgb = image.convert("RGB")
        tensor = self._preprocess(image_rgb).unsqueeze(0).to(self._device)  # type: ignore[operator, union-attr]

        with torch.no_grad():
            features = self._model.encode_image(tensor)  # type: ignore[union-attr]
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten().astype(np.float32)

    def embed_images(self, images: list[Image.Image]) -> NDArray[np.float32]:
        """Batch embedding — more efficient than sequential."""
        if not self._loaded:
            self.load()

        tensors = []
        for img in images:
            img_rgb = img.convert("RGB")
            tensor = self._preprocess(img_rgb)  # type: ignore[operator]
            tensors.append(tensor)

        batch = torch.stack(tensors).to(self._device)  # type: ignore[arg-type]

        with torch.no_grad():
            features = self._model.encode_image(batch)  # type: ignore[union-attr]
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)

    def embed_text(self, texts: str | list[str]) -> NDArray[np.float32]:
        """Generate embedding(s) for text prompt(s).

        Uses the CLIP text encoder to embed text into the same 1024-d
        space as images. This enables zero-shot classification by
        comparing image embeddings against text prompt embeddings.
        """
        if not self._loaded:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        tokens = self._tokenizer(texts).to(self._device)  # type: ignore[operator]

        with torch.no_grad():
            features = self._model.encode_text(tokens)  # type: ignore[union-attr]
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self._requested_device and self._requested_device not in ("", "auto"):
            return torch.device(self._requested_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Auto-register
# ---------------------------------------------------------------------------

register_backend("marqo", MarqoEcommerceBackend)
