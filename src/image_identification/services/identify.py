"""Service layer for product identification.

Extracts all business logic from the ``identify`` CLI handler into a single
public function ``run_identification`` plus private helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from image_identification.domain.errors import EmbeddingsNotFoundError, NoCategoryItemsError
from image_identification.logging import logger

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from image_identification.config import AppConfig
    from image_identification.index.in_memory import SimilarityMatch


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class IdentifyResult:
    """Value object returned by ``run_identification``."""

    matches: list[SimilarityMatch] = field(default_factory=list)
    detected_category: str = ""
    filtered_count: int = 0
    total_count: int = 0
    duration_seconds: float = 0.0
    name_lookup: dict[str, str] = field(default_factory=dict)
    url_lookup: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_identification(
    config: AppConfig,
    *,
    image_path: str,
    top_k: int = 5,
    embedding_dir: str = "",
    category: str = "",
) -> IdentifyResult:
    """Identify a product image against the catalog.

    Raises:
        EmbeddingsNotFoundError: No embeddings on disk.
        NoCategoryItemsError: No items of the detected category in the index.
    """
    start = time.time()

    # STEP 1/6 — Load query image ----------------------------------------
    image, _file_size_kb = _load_query_image(image_path)

    # STEP 2/6 — Load embedding model ------------------------------------
    emb_dir = Path(embedding_dir) if embedding_dir else config.storage.embedding_dir
    backend, _metadata = _load_backend(config, emb_dir)

    # STEP 3/6 — Generate query embedding --------------------------------
    query_vector = _embed_query(image, backend)

    # STEP 4/6 — Detect category -----------------------------------------
    detected_category = _detect_category(query_vector, backend, category)

    # STEP 5/6 — Load catalog embeddings from disk -----------------------
    (
        _enriched_df,
        name_lookup,
        url_lookup,
        category_lookup,
        embeddings,
        item_codes,
    ) = _load_catalog_data(config, emb_dir)

    # STEP 6/6 — Filter & search -----------------------------------------
    matches, filtered_count = _search_and_rank(
        query_vector, embeddings, item_codes, category_lookup,
        detected_category, top_k,
    )

    total_time = time.time() - start
    logger.info("=" * 60)
    logger.info("DONE \u2014 Total identify time: {:.2f}s", total_time)
    logger.info("=" * 60)

    return IdentifyResult(
        matches=matches,
        detected_category=detected_category,
        filtered_count=filtered_count,
        total_count=len(item_codes),
        duration_seconds=total_time,
        name_lookup=name_lookup,
        url_lookup=url_lookup,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_query_image(image_path: str) -> tuple[object, float]:
    """Load and validate the query image."""
    from PIL import Image

    logger.info("=" * 60)
    logger.info("STEP 1/6 \u2014 Loading query image")
    logger.info("=" * 60)

    image = Image.open(image_path).convert("RGB")
    file_size_kb = Path(image_path).stat().st_size / 1024
    logger.info(
        "Query image loaded: {} ({}x{}, {:.0f} KB)",
        image_path, image.width, image.height, file_size_kb,
    )
    return image, file_size_kb


def _load_backend(
    config: AppConfig,
    emb_dir: Path,
) -> tuple[object, object]:
    """Load the embedding backend using metadata from the stored index."""
    import image_identification.embeddings.marqo_backend  # noqa: F401
    from image_identification.embeddings.interface import get_backend
    from image_identification.storage.embedding_store import EmbeddingStore

    logger.info("=" * 60)
    logger.info("STEP 2/6 \u2014 Loading embedding model")
    logger.info("=" * 60)

    store = EmbeddingStore(emb_dir)
    if not store.exists():
        raise EmbeddingsNotFoundError(str(emb_dir))

    metadata = store.load_metadata()
    logger.info(
        "Embedding metadata: model='{}', dim={}, backend='{}'",
        metadata.model_name, metadata.embedding_dim, metadata.backend_key,
    )

    backend = get_backend(
        metadata.backend_key or config.embedding.backend,
        model_name=metadata.model_name,
        device=config.embedding.device,
    )
    model_load_start = time.time()
    backend.load()
    model_load_time = time.time() - model_load_start
    logger.info("Backend '{}' loaded in {:.2f}s", backend.name, model_load_time)

    return backend, metadata


def _embed_query(image: object, backend: object) -> NDArray[np.float32]:
    """Generate the embedding vector for the query image."""
    logger.info("=" * 60)
    logger.info("STEP 3/6 \u2014 Generating query image embedding")
    logger.info("=" * 60)

    embed_start = time.time()
    query_vector = backend.embed_image(image)  # type: ignore[union-attr]
    embed_time = time.time() - embed_start
    vec_preview = str(query_vector[:4].tolist())[:40]
    logger.info(
        "Query embedding generated in {:.2f}s (dim={}, preview: {}\u2026)",
        embed_time, len(query_vector), vec_preview,
    )
    return query_vector


def _detect_category(
    query_vector: NDArray[np.float32],
    backend: object,
    manual_category: str,
) -> str:
    """Detect the product category via zero-shot or manual override."""
    from image_identification.domain.models import Category
    from image_identification.features.zero_shot_classifier import classify_image_zero_shot

    logger.info("=" * 60)
    logger.info("STEP 4/6 \u2014 Detecting query category")
    logger.info("=" * 60)

    if manual_category:
        detected = manual_category.strip().lower()
        logger.info("Category manually set by user: '{}'", detected)
        return detected

    logger.info("Running zero-shot CLIP classification (image vs text prompts)\u2026")
    zs_start = time.time()
    predicted_cat, scores = classify_image_zero_shot(query_vector, backend)  # type: ignore[arg-type]
    zs_time = time.time() - zs_start
    detected = predicted_cat.value
    logger.info(
        "Zero-shot result in {:.2f}s: category='{}' "
        "(shoe={:.4f}, apparel={:.4f}, accessory={:.4f})",
        zs_time, detected,
        scores[Category.SHOE], scores[Category.APPAREL], scores[Category.ACCESSORY],
    )
    return detected


def _load_catalog_data(
    config: AppConfig,
    emb_dir: Path,
) -> tuple[object, dict[str, str], dict[str, str], dict[str, str], NDArray[np.float32], list[str]]:
    """Load enriched CSV and embedding vectors from disk."""
    from image_identification.storage.csv_store import load_enriched_csv
    from image_identification.storage.embedding_store import EmbeddingStore

    logger.info("=" * 60)
    logger.info("STEP 5/6 \u2014 Loading catalog embeddings and enriched data")
    logger.info("=" * 60)

    output_csv = config.storage.output_dir / "enriched_catalog.csv"
    if not output_csv.exists():
        raise EmbeddingsNotFoundError(str(output_csv))

    enriched_df = load_enriched_csv(output_csv)
    codes_series = enriched_df["item_code"].astype(str)
    name_lookup = dict(zip(codes_series, enriched_df["item_name"].astype(str), strict=False))
    url_lookup = dict(zip(codes_series, enriched_df["image_url"].astype(str), strict=False))
    category_lookup = dict(zip(codes_series, enriched_df["category"].astype(str), strict=False))
    logger.info(
        "Enriched catalog loaded: {} items from {}",
        len(enriched_df), output_csv,
    )

    cat_counts: dict[str, int] = {}
    for cat_val in category_lookup.values():
        cat_counts[cat_val] = cat_counts.get(cat_val, 0) + 1
    for cat_name, count in sorted(cat_counts.items()):
        logger.info("  Category '{}': {} items", cat_name, count)

    store = EmbeddingStore(emb_dir)
    embeddings = store.load_embeddings()
    item_codes = store.load_item_codes()
    logger.info(
        "Embedding vectors loaded: {} items, dim={}",
        len(item_codes), embeddings.shape[1],
    )

    return enriched_df, name_lookup, url_lookup, category_lookup, embeddings, item_codes


def _search_and_rank(
    query_vector: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    item_codes: list[str],
    category_lookup: dict[str, str],
    detected_category: str,
    top_k: int,
) -> tuple[list[SimilarityMatch], int]:
    """Filter by category, build index, and run cosine-similarity search."""
    import numpy as np

    from image_identification.index.in_memory import InMemoryIndex

    logger.info("=" * 60)
    logger.info("STEP 6/6 \u2014 Searching within category '{}'", detected_category)
    logger.info("=" * 60)

    filtered_indices = [
        i
        for i, code in enumerate(item_codes)
        if category_lookup.get(str(code), "unknown") == detected_category
    ]

    if not filtered_indices:
        raise NoCategoryItemsError(detected_category)

    filtered_embeddings = np.stack([embeddings[i] for i in filtered_indices])
    filtered_codes = [item_codes[i] for i in filtered_indices]
    logger.info(
        "Filtered index: {} items of category '{}' (from {} total)",
        len(filtered_codes), detected_category, len(item_codes),
    )

    filtered_index = InMemoryIndex()
    filtered_index.build(filtered_embeddings, filtered_codes)
    logger.info("Index built, running cosine similarity search (top_k={})...", top_k)

    search_start = time.time()
    matches = filtered_index.search(query_vector, top_k=top_k)
    search_time = time.time() - search_start
    logger.info("Search completed in {:.4f}s \u2014 {} matches found", search_time, len(matches))

    for i, match in enumerate(matches, 1):
        code = str(match.item_code)
        logger.info("  #{} score={:.4f} \u2192 {}", i, match.score, code)

    return matches, len(filtered_codes)
