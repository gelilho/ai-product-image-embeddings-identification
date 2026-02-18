"""Tests for the pluggable embedding backend interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from image_identification.embeddings.interface import (
    EmbeddingBackend,
    get_backend,
    list_backends,
    register_backend,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image


class FakeBackend(EmbeddingBackend):
    """Minimal test backend."""

    def __init__(self, model_name: str = "fake", device: str = "") -> None:
        self._model_name = model_name
        self._device = device
        self._loaded = False

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return "1.0.0"

    @property
    def embedding_dim(self) -> int:
        return 128

    def load(self) -> None:
        self._loaded = True

    def embed_image(self, image: Image.Image) -> NDArray[np.float32]:
        return np.random.randn(128).astype(np.float32)

    def is_loaded(self) -> bool:
        return self._loaded


class TestEmbeddingInterface:
    """Test backend registration and retrieval."""

    def test_register_and_get(self) -> None:
        register_backend("fake_test", FakeBackend)
        backend = get_backend("fake_test")
        assert backend.name == "fake"
        assert backend.embedding_dim == 128

    def test_get_with_kwargs(self) -> None:
        register_backend("fake_test2", FakeBackend)
        backend = get_backend("fake_test2", model_name="custom-model", device="cpu")
        assert backend.model_name == "custom-model"

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown backend"):
            get_backend("nonexistent_backend_xyz")

    def test_list_backends(self) -> None:
        register_backend("fake_list_test", FakeBackend)
        backends = list_backends()
        assert "fake_list_test" in backends

    def test_load_and_check(self) -> None:
        register_backend("fake_load_test", FakeBackend)
        backend = get_backend("fake_load_test")
        assert not backend.is_loaded()
        backend.load()
        assert backend.is_loaded()

    def test_embed_images_default_batch(self) -> None:
        """Default batch implementation processes sequentially."""
        from PIL import Image

        register_backend("fake_batch_test", FakeBackend)
        backend = get_backend("fake_batch_test")
        backend.load()

        # Create small test images
        images = [Image.new("RGB", (10, 10)) for _ in range(3)]
        result = backend.embed_images(images)
        assert result.shape == (3, 128)

    def test_backend_properties(self) -> None:
        backend = FakeBackend()
        assert backend.name == "fake"
        assert backend.model_version == "1.0.0"
        assert backend.embedding_dim == 128

    def test_embed_text_not_implemented_by_default(self) -> None:
        """Base class embed_text raises NotImplementedError."""
        backend = FakeBackend()
        with pytest.raises(NotImplementedError, match="fake backend does not support text"):
            backend.embed_text("hello")
