"""Zero-shot category classification using CLIP text embeddings.

Instead of relying on rule-based keyword matching or the top-1 match
from the index, this module classifies a query image by comparing its
embedding against descriptive text prompts for each category.

The CLIP model encodes both images and text into the same 1024-d space,
so cosine similarity between an image embedding and a text embedding
directly measures how well the text describes the image.

Example:
    Image of a shoe → highest similarity to "a photo of a running shoe"
    Image of a jacket → highest similarity to "a photo of a jacket"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from image_identification.domain.models import Category
from image_identification.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from image_identification.embeddings.interface import EmbeddingBackend

# ---------------------------------------------------------------------------
# Text prompts per category — multiple prompts per category for robustness.
# The final score for a category is the MAX across its prompts.
# ---------------------------------------------------------------------------

CATEGORY_PROMPTS: dict[Category, list[str]] = {
    Category.SHOE: [
        "a photo of a running shoe",
        "a photo of a sneaker",
        "a photo of athletic footwear",
        "a photo of a sports shoe",
        "a photo of a trainer shoe",
    ],
    Category.APPAREL: [
        "a photo of clothing apparel",
        "a photo of a jacket",
        "a photo of a shirt or t-shirt",
        "a photo of pants or shorts",
        "a photo of a hoodie or sweater",
    ],
    Category.ACCESSORY: [
        "a photo of a hat or beanie",
        "a photo of socks",
        "a photo of a bag or backpack",
        "a photo of a sports accessory",
        "a photo of a cap or visor",
    ],
}

# All categories we classify (ordered for consistent results)
_CATEGORIES = [Category.SHOE, Category.APPAREL, Category.ACCESSORY]


def classify_image_zero_shot(
    image_embedding: NDArray[np.float32],
    backend: EmbeddingBackend,
) -> tuple[Category, dict[Category, float]]:
    """Classify a query image into a product category using zero-shot CLIP.

    Compares the image embedding against descriptive text prompts for each
    category. The category with the highest similarity score wins.

    Parameters
    ----------
    image_embedding:
        1-D embedding vector of the query image (shape: embedding_dim).
    backend:
        An embedding backend that supports ``embed_text()``.

    Returns
    -------
    tuple[Category, dict[Category, float]]
        The predicted category and a dict mapping each category to its
        best similarity score (for logging / debugging).
    """
    # Build all text prompts and track which category each belongs to
    all_prompts: list[str] = []
    prompt_to_category: list[Category] = []
    for cat in _CATEGORIES:
        prompts = CATEGORY_PROMPTS[cat]
        all_prompts.extend(prompts)
        prompt_to_category.extend([cat] * len(prompts))

    # Embed all text prompts in one batch (efficient)
    text_embeddings = backend.embed_text(all_prompts)  # (N_prompts, dim)

    # Cosine similarity: image_embedding is already L2-normalized (from embed_image),
    # and text_embeddings are L2-normalized (from embed_text).
    # Dot product = cosine similarity for normalized vectors.
    if image_embedding.ndim == 1:
        image_embedding = image_embedding.reshape(1, -1)
    similarities = (image_embedding @ text_embeddings.T).flatten()  # (N_prompts,)

    # Aggregate: take the MAX score per category
    category_scores: dict[Category, float] = {}
    for cat in _CATEGORIES:
        cat_indices = [i for i, c in enumerate(prompt_to_category) if c == cat]
        cat_scores = similarities[cat_indices]
        category_scores[cat] = float(np.max(cat_scores))

    # Winner = highest score
    predicted = max(category_scores, key=lambda c: category_scores[c])

    logger.info(
        "Zero-shot classification: {} (scores: shoe={:.4f}, apparel={:.4f}, accessory={:.4f})",
        predicted.value,
        category_scores[Category.SHOE],
        category_scores[Category.APPAREL],
        category_scores[Category.ACCESSORY],
    )

    return predicted, category_scores
