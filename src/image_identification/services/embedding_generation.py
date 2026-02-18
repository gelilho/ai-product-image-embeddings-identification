"""Service layer for embedding generation.

Extracts all business logic from the CLI handler into a single public
function ``run_pipeline`` plus private helpers.  The CLI stays a thin
shell that wires config, calls this module, and formats output.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from image_identification.logging import logger

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from image_identification.config import AppConfig
    from image_identification.domain.models import CatalogItem, EnrichedItem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_BATCH_SIZE = 20
_BATCH_COOLDOWN_SECONDS = 5.0
_EMBEDDING_PREVIEW_LEN = 20  # chars of embedding vector shown in logs


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Value object returned by ``run_pipeline``."""

    total_items: int
    successful: int
    failed: int
    skipped: int
    newly_generated: int
    errors: list[dict[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0
    output_path: Path = field(default_factory=lambda: Path("."))
    embedding_dir: Path = field(default_factory=lambda: Path("."))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_pipeline(
    config: AppConfig,
    *,
    limit: int = 0,
    output: str = "",
    batch_size: int = _DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> PipelineResult:
    """Execute the full embedding-generation pipeline.

    Returns a :class:`PipelineResult` with summary statistics.  Raises no
    Typer-specific exceptions — the CLI layer is responsible for converting
    errors into user-facing messages.
    """
    from image_identification.images.url_builder import UrlBuilder
    from image_identification.storage.csv_store import save_enriched_csv
    from image_identification.storage.embedding_store import EmbeddingStore

    logger.info("Starting embedding generation (limit={})", limit or "all")
    logger.info("Config: {}", config.log_summary())

    start = time.time()

    # 0. Load existing embeddings (per-item skip logic) ------------------
    store = EmbeddingStore(config.storage.embedding_dir)
    _existing_codes_set, existing_embeddings_map = _load_existing_embeddings(
        store, force=force
    )

    # 1. Load catalog ----------------------------------------------------
    items = _load_catalog(config, limit=limit)

    # 2. Determine which items need generation ---------------------------
    url_builder = UrlBuilder(config.fetcher)
    items_to_fetch, skipped_count = _filter_items_needing_generation(
        items, existing_embeddings_map
    )

    # 3. Fetch images (batched) ------------------------------------------
    all_fetch_results = _fetch_images_batched(
        items_to_fetch, config, url_builder, batch_size
    )

    # 4. Load embedding backend (only if needed) -------------------------
    backend = _load_backend_if_needed(items_to_fetch, config)

    # 5. Process each item — reuse existing or generate new --------------
    enriched, errors, all_embeddings, all_item_codes, model_name, model_version, embedding_dim = (
        _process_items(
            items,
            existing_embeddings_map,
            all_fetch_results,
            backend,
            store,
            url_builder,
        )
    )

    # 6. Save enriched CSV -----------------------------------------------
    output_path = Path(output) if output else config.storage.output_dir / "enriched_catalog.csv"
    save_enriched_csv(enriched, output_path)

    # 7. Save embeddings -------------------------------------------------
    _save_results(
        all_embeddings, all_item_codes, model_name, model_version,
        embedding_dim, backend, store,
    )

    duration = time.time() - start
    newly_generated = len(items_to_fetch) - len(errors)

    logger.info("Embedding generation complete in {:.1f}s", duration)
    logger.info(
        "Results: {}/{} total ({} reused, {} newly generated, {} failed)",
        len(enriched),
        len(items),
        skipped_count,
        newly_generated,
        len(errors),
    )
    if errors:
        logger.warning("Errors:")
        for err in errors:
            logger.warning("  {} \u2192 {}", err["item_code"], err["error"])

    return PipelineResult(
        total_items=len(items),
        successful=len(enriched),
        failed=len(errors),
        skipped=skipped_count,
        newly_generated=newly_generated,
        errors=errors,
        duration_seconds=duration,
        output_path=output_path,
        embedding_dir=config.storage.embedding_dir,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_existing_embeddings(
    store: object,
    *,
    force: bool,
) -> tuple[set[str], dict[str, NDArray[np.float32]]]:
    """Return (set of codes, map code->vector) from disk, or empties if force."""
    existing_codes_set: set[str] = set()
    existing_embeddings_map: dict[str, NDArray[np.float32]] = {}

    # store is an EmbeddingStore — access via duck-typing to avoid top-level import
    if store.exists() and not force:  # type: ignore[union-attr]
        existing_codes = store.load_item_codes()  # type: ignore[union-attr]
        existing_matrix = store.load_embeddings()  # type: ignore[union-attr]
        existing_codes_set = set(existing_codes)
        for i, code in enumerate(existing_codes):
            existing_embeddings_map[code] = existing_matrix[i]
        logger.info("Found {} existing embeddings on disk", len(existing_codes_set))

    return existing_codes_set, existing_embeddings_map


def _load_catalog(config: AppConfig, *, limit: int) -> list[CatalogItem]:
    from image_identification.catalog.loader import load_catalog

    items = load_catalog(
        config.storage.golden_dataset,
        limit=limit if limit > 0 else None,
    )
    logger.info("Loaded {} catalog items", len(items))

    items_with_dataset_category = sum(1 for i in items if i.category is not None)
    if items_with_dataset_category:
        logger.info(
            "Category source: {} from golden dataset, {} will use rule-based classifier",
            items_with_dataset_category,
            len(items) - items_with_dataset_category,
        )
    else:
        logger.info("No categories in golden dataset \u2014 all items will use rule-based classifier")

    return items


def _filter_items_needing_generation(
    items: list[CatalogItem],
    existing_embeddings_map: dict[str, NDArray[np.float32]],
) -> tuple[list[str], int]:
    """Return (item_codes_to_fetch, skipped_count)."""
    items_to_fetch: list[str] = []
    skipped_count = 0

    for idx, item in enumerate(items, 1):
        if item.item_code in existing_embeddings_map:
            vec = existing_embeddings_map[item.item_code]
            vec_preview = str(vec[:4].tolist())[:_EMBEDDING_PREVIEW_LEN]
            logger.info(
                "[{}/{}] \u23ed  {} embedding already generated (embedding: {}\u2026)",
                idx, len(items), item.item_code, vec_preview,
            )
            skipped_count += 1
        else:
            logger.info(
                "[{}/{}] \ud83c\udd95 {} embedding not found, will generate",
                idx, len(items), item.item_code,
            )
            items_to_fetch.append(item.item_code)

    logger.info(
        "Summary: {} already generated, {} to generate",
        skipped_count, len(items_to_fetch),
    )
    return items_to_fetch, skipped_count


def _fetch_images_batched(
    items_to_fetch: list[str],
    config: AppConfig,
    url_builder: object,
    batch_size: int,
) -> list[object]:
    """Fetch product images with batched pagination (anti-DoS)."""
    from image_identification.images.fetcher import ImageFetcher

    all_fetch_results: list[object] = []

    if not items_to_fetch:
        return all_fetch_results

    fetcher = ImageFetcher(
        config=config.fetcher,
        url_builder=url_builder,  # type: ignore[arg-type]
        cache_dir=config.storage.cache_dir,
    )

    if len(items_to_fetch) <= 5:
        logger.info("Small batch ({}), fetching all at once", len(items_to_fetch))
        all_fetch_results = asyncio.run(fetcher.fetch_batch(items_to_fetch))
    else:
        total_batches = (len(items_to_fetch) + batch_size - 1) // batch_size
        logger.info(
            "Fetching {} images in {} batches of {} (cooldown={}s between batches)",
            len(items_to_fetch), total_batches, batch_size, _BATCH_COOLDOWN_SECONDS,
        )

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(items_to_fetch))
            batch_codes = items_to_fetch[batch_start:batch_end]

            logger.info(
                "Batch {}/{}: fetching items {}-{} ({} items)",
                batch_idx + 1, total_batches,
                batch_start + 1, batch_end, len(batch_codes),
            )

            batch_results = asyncio.run(fetcher.fetch_batch(batch_codes))
            all_fetch_results.extend(batch_results)

            if batch_idx < total_batches - 1:
                logger.info(
                    "Batch {}/{} done. Cooling down {:.0f}s before next batch\u2026",
                    batch_idx + 1, total_batches, _BATCH_COOLDOWN_SECONDS,
                )
                time.sleep(_BATCH_COOLDOWN_SECONDS)

    return all_fetch_results


def _load_backend_if_needed(
    items_to_fetch: list[str],
    config: AppConfig,
) -> object | None:
    """Load the embedding backend only when there are items to generate."""
    if not items_to_fetch:
        return None

    import image_identification.embeddings.marqo_backend  # noqa: F401
    from image_identification.embeddings.interface import get_backend

    backend = get_backend(
        config.embedding.backend,
        model_name=config.embedding.model_name,
        device=config.embedding.device,
    )
    backend.load()
    return backend


def _process_items(
    items: list[CatalogItem],
    existing_embeddings_map: dict[str, NDArray[np.float32]],
    all_fetch_results: list[object],
    backend: object | None,
    store: object,
    url_builder: object,
) -> tuple[
    list[EnrichedItem],
    list[dict[str, str]],
    list[NDArray[np.float32]],
    list[str],
    str,
    str,
    int,
]:
    """Process all items: reuse existing embeddings or generate new ones.

    Returns (enriched, errors, all_embeddings, all_item_codes,
             model_name, model_version, embedding_dim).
    """
    from image_identification.domain.models import EnrichedItem
    from image_identification.features.category_classifier import classify_category
    from image_identification.features.color_extractor import extract_color
    from image_identification.images.validators import validate_image_bytes

    enriched: list[EnrichedItem] = []
    errors: list[dict[str, str]] = []
    all_embeddings: list[NDArray[np.float32]] = []
    all_item_codes: list[str] = []

    fetch_map = {r.item_code: r for r in all_fetch_results}  # type: ignore[union-attr]

    # Resolve model metadata
    model_name = ""
    model_version = ""
    embedding_dim = 0
    if backend is not None:
        model_name = backend.model_name  # type: ignore[union-attr]
        model_version = backend.model_version  # type: ignore[union-attr]
        embedding_dim = backend.embedding_dim  # type: ignore[union-attr]
    elif store.exists():  # type: ignore[union-attr]
        meta = store.load_metadata()  # type: ignore[union-attr]
        model_name = meta.model_name
        model_version = meta.model_version
        embedding_dim = meta.embedding_dim

    for idx, item in enumerate(items, 1):
        # --- Case A: Existing embedding (skip) ---
        if item.item_code in existing_embeddings_map:
            vector = existing_embeddings_map[item.item_code]
            all_embeddings.append(vector)
            all_item_codes.append(item.item_code)

            color = extract_color(item.item_name)
            category = (
                item.category
                if item.category is not None
                else classify_category(item.item_name, item.family, item.vertical)
            )
            url = url_builder.build(item.item_code)  # type: ignore[union-attr]

            enriched.append(
                EnrichedItem(
                    item=item,
                    image_url=url,
                    color=color,
                    category=category,
                    embedding_ref=f"{item.item_code}.npy",
                    embedding_model=model_name,
                    embedding_dim=embedding_dim or len(vector),
                )
            )
            continue

        # --- Case B: New embedding needed ---
        fetch_result = fetch_map.get(item.item_code)
        if not fetch_result or not fetch_result.success or not fetch_result.image_bytes:
            errors.append(
                {
                    "item_code": item.item_code,
                    "error": fetch_result.error if fetch_result else "Not fetched",
                }
            )
            logger.warning(
                "[{}/{}] \u2717 {} \u2014 fetch failed: {}",
                idx, len(items), item.item_code,
                fetch_result.error if fetch_result else "Not fetched",
            )
            continue

        if backend is None:
            errors.append({"item_code": item.item_code, "error": "No backend loaded"})
            continue

        try:
            pil_image = validate_image_bytes(
                fetch_result.image_bytes, item.item_code, fetch_result.url,
            )
            logger.info(
                "[{}/{}] \ud83d\udd04 Generating embedding for {} ({}x{}, {:.0f} KB) \u2026",
                idx, len(items), item.item_code,
                pil_image.width, pil_image.height,
                len(fetch_result.image_bytes) / 1024,
            )

            vector = backend.embed_image(pil_image)  # type: ignore[union-attr]
            vec_preview = str(vector[:4].tolist())[:_EMBEDDING_PREVIEW_LEN]
            all_embeddings.append(vector)
            all_item_codes.append(item.item_code)

            color = extract_color(item.item_name)
            category = (
                item.category
                if item.category is not None
                else classify_category(item.item_name, item.family, item.vertical)
            )
            url = url_builder.build(item.item_code)  # type: ignore[union-attr]

            enriched.append(
                EnrichedItem(
                    item=item,
                    image_url=url,
                    color=color,
                    category=category,
                    embedding_ref=f"{item.item_code}.npy",
                    embedding_model=backend.model_name,  # type: ignore[union-attr]
                    embedding_dim=backend.embedding_dim,  # type: ignore[union-attr]
                )
            )

            cat_source = "dataset" if item.category is not None else "classifier"
            logger.info(
                "[{}/{}] \u2713 {} \u2192 generated (embedding: {}\u2026), color='{}', category='{}' ({})",
                idx, len(items), item.item_code,
                vec_preview, color, category.value, cat_source,
            )

        except Exception as exc:
            errors.append({"item_code": item.item_code, "error": str(exc)})
            logger.warning("[{}/{}] \u2717 {} \u2014 error: {}", idx, len(items), item.item_code, exc)

    return enriched, errors, all_embeddings, all_item_codes, model_name, model_version, embedding_dim


def _save_results(
    all_embeddings: list[NDArray[np.float32]],
    all_item_codes: list[str],
    model_name: str,
    model_version: str,
    embedding_dim: int,
    backend: object | None,
    store: object,
) -> None:
    """Save embedding matrix, item codes, and metadata to disk."""
    if not all_embeddings:
        return

    import numpy as np

    from image_identification.embeddings.metadata import EmbeddingMetadata

    embedding_matrix = np.stack(all_embeddings)
    metadata = EmbeddingMetadata(
        model_name=model_name,
        model_version=model_version,
        embedding_dim=embedding_dim or embedding_matrix.shape[1],
        total_items=len(all_item_codes),
        backend_key=backend.name if backend else "marqo",  # type: ignore[union-attr]
    )
    store.save(embedding_matrix, all_item_codes, metadata)  # type: ignore[union-attr]
