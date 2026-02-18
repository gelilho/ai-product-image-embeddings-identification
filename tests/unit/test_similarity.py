"""Tests for cosine similarity and the in-memory index."""

import numpy as np
import pytest

from image_identification.index.in_memory import InMemoryIndex
from image_identification.index.similarity import cosine_similarity


class TestCosineSimilarity:
    """Test the cosine similarity function."""

    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(a, b)
        assert scores.shape == (1,)
        assert pytest.approx(scores[0], abs=1e-5) == 1.0

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(a, b)
        assert pytest.approx(scores[0], abs=1e-5) == 0.0

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(a, b)
        assert pytest.approx(scores[0], abs=1e-5) == -1.0

    def test_batch_similarity(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        scores = cosine_similarity(a, b)
        assert scores.shape == (3,)
        assert scores[0] > scores[2] > scores[1]


class TestInMemoryIndex:
    """Test the in-memory index build and search."""

    @pytest.fixture
    def populated_index(self) -> InMemoryIndex:
        index = InMemoryIndex()
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.7, 0.7, 0.0],
            ],
            dtype=np.float32,
        )
        codes = ["item_a", "item_b", "item_c", "item_d"]
        index.build(embeddings, codes)
        return index

    def test_build_size(self, populated_index: InMemoryIndex) -> None:
        assert populated_index.size() == 4

    def test_exact_match_top_1(self, populated_index: InMemoryIndex) -> None:
        """Querying with an exact vector should return itself as top-1."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matches = populated_index.search(query, top_k=1)
        assert len(matches) == 1
        assert matches[0].item_code == "item_a"
        assert pytest.approx(matches[0].score, abs=1e-4) == 1.0

    def test_top_k_ordering(self, populated_index: InMemoryIndex) -> None:
        """Results should be ordered by descending score."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matches = populated_index.search(query, top_k=4)
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, populated_index: InMemoryIndex) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matches = populated_index.search(query, top_k=2)
        assert len(matches) == 2

    def test_mismatch_raises(self) -> None:
        index = InMemoryIndex()
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="Mismatch"):
            index.build(embeddings, ["a", "b"])

    def test_search_before_build_raises(self) -> None:
        index = InMemoryIndex()
        with pytest.raises(RuntimeError, match="not built"):
            index.search(np.array([1.0], dtype=np.float32))
