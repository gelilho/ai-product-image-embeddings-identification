"""Tests for zero-shot category classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from product_image_id.domain.models import Category
from product_image_id.features.zero_shot_classifier import (
    CATEGORY_PROMPTS,
    classify_image_zero_shot,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FakeTextBackend:
    """Fake backend that returns controllable text embeddings.

    For testing, we pre-set the text embeddings so we can control
    what category wins the zero-shot classification.
    """

    def __init__(self, *, shoe_score: float = 0.5, apparel_score: float = 0.3, accessory_score: float = 0.2) -> None:
        self._shoe_score = shoe_score
        self._apparel_score = apparel_score
        self._accessory_score = accessory_score

    @property
    def name(self) -> str:
        return "fake_text"

    def embed_text(self, texts: str | list[str]) -> NDArray[np.float32]:
        """Return embeddings that produce the configured similarity scores.

        We create 2-D vectors where:
        - shoe prompts → [shoe_score, 0]
        - apparel prompts → [apparel_score, 0]
        - accessory prompts → [accessory_score, 0]

        When the query image is [1, 0], the dot product equals the score.
        """
        if isinstance(texts, str):
            texts = [texts]

        shoe_prompts = set(CATEGORY_PROMPTS[Category.SHOE])
        apparel_prompts = set(CATEGORY_PROMPTS[Category.APPAREL])

        vectors = []
        for text in texts:
            if text in shoe_prompts:
                vectors.append([self._shoe_score, 0.0])
            elif text in apparel_prompts:
                vectors.append([self._apparel_score, 0.0])
            else:
                vectors.append([self._accessory_score, 0.0])

        return np.array(vectors, dtype=np.float32)


class TestClassifyImageZeroShot:
    """Test zero-shot classification logic."""

    def test_shoe_wins(self) -> None:
        backend = FakeTextBackend(shoe_score=0.85, apparel_score=0.30, accessory_score=0.20)
        image_embedding = np.array([1.0, 0.0], dtype=np.float32)

        predicted, scores = classify_image_zero_shot(image_embedding, backend)  # type: ignore[arg-type]

        assert predicted == Category.SHOE
        assert scores[Category.SHOE] > scores[Category.APPAREL]
        assert scores[Category.SHOE] > scores[Category.ACCESSORY]

    def test_apparel_wins(self) -> None:
        backend = FakeTextBackend(shoe_score=0.20, apparel_score=0.90, accessory_score=0.15)
        image_embedding = np.array([1.0, 0.0], dtype=np.float32)

        predicted, scores = classify_image_zero_shot(image_embedding, backend)  # type: ignore[arg-type]

        assert predicted == Category.APPAREL
        assert scores[Category.APPAREL] > scores[Category.SHOE]

    def test_accessory_wins(self) -> None:
        backend = FakeTextBackend(shoe_score=0.10, apparel_score=0.15, accessory_score=0.95)
        image_embedding = np.array([1.0, 0.0], dtype=np.float32)

        predicted, scores = classify_image_zero_shot(image_embedding, backend)  # type: ignore[arg-type]

        assert predicted == Category.ACCESSORY
        assert scores[Category.ACCESSORY] > scores[Category.SHOE]
        assert scores[Category.ACCESSORY] > scores[Category.APPAREL]

    def test_returns_all_three_scores(self) -> None:
        backend = FakeTextBackend()
        image_embedding = np.array([1.0, 0.0], dtype=np.float32)

        _, scores = classify_image_zero_shot(image_embedding, backend)  # type: ignore[arg-type]

        assert Category.SHOE in scores
        assert Category.APPAREL in scores
        assert Category.ACCESSORY in scores
        assert len(scores) == 3

    def test_2d_image_embedding(self) -> None:
        """Should handle both 1-D and 2-D image embeddings."""
        backend = FakeTextBackend(shoe_score=0.80, apparel_score=0.30, accessory_score=0.20)
        # Pass as 2-D (1, dim) — should still work
        image_embedding = np.array([[1.0, 0.0]], dtype=np.float32)

        predicted, _scores = classify_image_zero_shot(image_embedding, backend)  # type: ignore[arg-type]

        assert predicted == Category.SHOE


class TestCategoryPrompts:
    """Test that category prompts are well-formed."""

    def test_all_categories_have_prompts(self) -> None:
        assert Category.SHOE in CATEGORY_PROMPTS
        assert Category.APPAREL in CATEGORY_PROMPTS
        assert Category.ACCESSORY in CATEGORY_PROMPTS

    def test_each_category_has_multiple_prompts(self) -> None:
        for cat, prompts in CATEGORY_PROMPTS.items():
            assert len(prompts) >= 2, f"{cat} should have at least 2 prompts"

    def test_prompts_are_non_empty_strings(self) -> None:
        for cat, prompts in CATEGORY_PROMPTS.items():
            for prompt in prompts:
                assert isinstance(prompt, str)
                assert len(prompt.strip()) > 0, f"Empty prompt found in {cat}"
