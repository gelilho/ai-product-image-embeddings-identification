"""CLI entrypoint — exposes embedding-generation, identify, and evaluate commands.

Usage:
    uv run embedding-generation --limit 30
    uv run identify path/to/image.png
    uv run evaluate --top-k 5
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer

from product_image_id.config import AppConfig
from product_image_id.logging import logger, setup_logging

app = typer.Typer(
    name="ai-product-image-embeddings-identification",
    help="Product Image Identification — embedding-based product matching.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Default batch size for paginated fetching (anti-DoS)
# ---------------------------------------------------------------------------
_DEFAULT_BATCH_SIZE = 20
_BATCH_COOLDOWN_SECONDS = 5.0
_EMBEDDING_PREVIEW_LEN = 20  # how many chars of the embedding vector to show in logs


@app.command("embedding-generation")
def embedding_generation(
    limit: int = typer.Option(0, help="Max items to process (0 = all)"),
    output: str = typer.Option("", help="Output CSV path (default: auto)"),
    batch_size: int = typer.Option(
        _DEFAULT_BATCH_SIZE,
        help="Items per batch for image fetching (anti-DoS). Batches pause between rounds.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-generation even if embeddings exist"
    ),
) -> None:
    """Build the embedding catalog from the golden dataset.

    Steps: load catalog → fetch images (batched) → embed → extract attrs → save.

    Skips items that already have embeddings unless --force is passed.
    For datasets larger than 5 items, images are fetched in batches with
    a cooldown pause between each batch to avoid overwhelming the server.
    """
    config = AppConfig()
    setup_logging(config.logging)
    logger.info("Starting embedding generation (limit={})", limit or "all")
    logger.info("Config: {}", config.log_summary())

    start = time.time()

    # Lazy imports to avoid heavy model loading at CLI parse time
    import numpy as np

    from product_image_id.catalog.loader import load_catalog
    from product_image_id.domain.models import EnrichedItem, PipelineReport
    from product_image_id.embeddings.interface import get_backend
    from product_image_id.embeddings.metadata import EmbeddingMetadata
    from product_image_id.features.category_classifier import classify_category
    from product_image_id.features.color_extractor import extract_color
    from product_image_id.images.fetcher import ImageFetcher
    from product_image_id.images.url_builder import UrlBuilder
    from product_image_id.images.validators import validate_image_bytes
    from product_image_id.storage.csv_store import save_enriched_csv
    from product_image_id.storage.embedding_store import EmbeddingStore

    # ------------------------------------------------------------------
    # 0. Load existing embeddings (for per-item skip logic)
    # ------------------------------------------------------------------
    store = EmbeddingStore(config.storage.embedding_dir)
    existing_codes_set: set[str] = set()
    existing_embeddings_map: dict[str, np.ndarray] = {}

    if store.exists() and not force:
        existing_codes = store.load_item_codes()
        existing_matrix = store.load_embeddings()
        existing_codes_set = set(existing_codes)
        for i, code in enumerate(existing_codes):
            existing_embeddings_map[code] = existing_matrix[i]
        logger.info("Found {} existing embeddings on disk", len(existing_codes_set))

    # ------------------------------------------------------------------
    # 1. Load catalog
    # ------------------------------------------------------------------
    items = load_catalog(
        config.storage.golden_dataset,
        limit=limit if limit > 0 else None,
    )
    logger.info("Loaded {} catalog items", len(items))

    # Log category source breakdown
    items_with_dataset_category = sum(1 for i in items if i.category is not None)
    if items_with_dataset_category:
        logger.info(
            "Category source: {} from golden dataset, {} will use rule-based classifier",
            items_with_dataset_category,
            len(items) - items_with_dataset_category,
        )
    else:
        logger.info("No categories in golden dataset — all items will use rule-based classifier")

    # ------------------------------------------------------------------
    # 2. Determine which items need embedding generation
    # ------------------------------------------------------------------
    url_builder = UrlBuilder(config.fetcher)
    items_to_fetch: list[str] = []
    skipped_count = 0

    for idx, item in enumerate(items, 1):
        if item.item_code in existing_embeddings_map:
            vec = existing_embeddings_map[item.item_code]
            vec_preview = str(vec[:4].tolist())[:_EMBEDDING_PREVIEW_LEN]
            logger.info(
                "[{}/{}] ⏭  {} embedding already generated (embedding: {}…)",
                idx,
                len(items),
                item.item_code,
                vec_preview,
            )
            skipped_count += 1
        else:
            logger.info(
                "[{}/{}] 🆕 {} embedding not found, will generate",
                idx,
                len(items),
                item.item_code,
            )
            items_to_fetch.append(item.item_code)

    logger.info(
        "Summary: {} already generated, {} to generate",
        skipped_count,
        len(items_to_fetch),
    )

    # ------------------------------------------------------------------
    # 3. Fetch images — only for items that need embeddings (batched)
    # ------------------------------------------------------------------
    all_fetch_results = []

    if items_to_fetch:
        fetcher = ImageFetcher(
            config=config.fetcher,
            url_builder=url_builder,
            cache_dir=config.storage.cache_dir,
        )

        if len(items_to_fetch) <= 5:
            logger.info("Small batch ({}), fetching all at once", len(items_to_fetch))
            all_fetch_results = asyncio.run(fetcher.fetch_batch(items_to_fetch))
        else:
            total_batches = (len(items_to_fetch) + batch_size - 1) // batch_size
            logger.info(
                "Fetching {} images in {} batches of {} (cooldown={}s between batches)",
                len(items_to_fetch),
                total_batches,
                batch_size,
                _BATCH_COOLDOWN_SECONDS,
            )

            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(items_to_fetch))
                batch_codes = items_to_fetch[batch_start:batch_end]

                logger.info(
                    "Batch {}/{}: fetching items {}-{} ({} items)",
                    batch_idx + 1,
                    total_batches,
                    batch_start + 1,
                    batch_end,
                    len(batch_codes),
                )

                batch_results = asyncio.run(fetcher.fetch_batch(batch_codes))
                all_fetch_results.extend(batch_results)

                if batch_idx < total_batches - 1:
                    logger.info(
                        "Batch {}/{} done. Cooling down {:.0f}s before next batch…",
                        batch_idx + 1,
                        total_batches,
                        _BATCH_COOLDOWN_SECONDS,
                    )
                    time.sleep(_BATCH_COOLDOWN_SECONDS)

    # ------------------------------------------------------------------
    # 4. Load embedding backend (only if there are items to generate)
    # ------------------------------------------------------------------
    backend = None
    if items_to_fetch:
        import product_image_id.embeddings.marqo_backend  # noqa: F401

        backend = get_backend(
            config.embedding.backend,
            model_name=config.embedding.model_name,
            device=config.embedding.device,
        )
        backend.load()

    # ------------------------------------------------------------------
    # 5. Process each item — reuse existing or generate new
    # ------------------------------------------------------------------
    enriched: list[EnrichedItem] = []
    errors: list[dict[str, str]] = []

    all_embeddings: list[np.ndarray] = []
    all_item_codes: list[str] = []

    fetch_map = {r.item_code: r for r in all_fetch_results}

    # We need a model name for enriched items — use existing metadata or backend
    model_name = ""
    model_version = ""
    embedding_dim = 0
    if backend is not None:
        model_name = backend.model_name
        model_version = backend.model_version
        embedding_dim = backend.embedding_dim
    elif store.exists():
        meta = store.load_metadata()
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
            # Prefer golden dataset category; fall back to rule-based classifier
            category = (
                item.category
                if item.category is not None
                else classify_category(item.item_name, item.family, item.vertical)
            )
            url = url_builder.build(item.item_code)

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
                "[{}/{}] ✗ {} — fetch failed: {}",
                idx,
                len(items),
                item.item_code,
                fetch_result.error if fetch_result else "Not fetched",
            )
            continue

        if backend is None:
            errors.append({"item_code": item.item_code, "error": "No backend loaded"})
            continue

        try:
            pil_image = validate_image_bytes(
                fetch_result.image_bytes, item.item_code, fetch_result.url
            )
            logger.info(
                "[{}/{}] 🔄 Generating embedding for {} ({}x{}, {:.0f} KB) …",
                idx,
                len(items),
                item.item_code,
                pil_image.width,
                pil_image.height,
                len(fetch_result.image_bytes) / 1024,
            )

            vector = backend.embed_image(pil_image)
            vec_preview = str(vector[:4].tolist())[:_EMBEDDING_PREVIEW_LEN]
            all_embeddings.append(vector)
            all_item_codes.append(item.item_code)

            color = extract_color(item.item_name)
            # Prefer golden dataset category; fall back to rule-based classifier
            category = (
                item.category
                if item.category is not None
                else classify_category(item.item_name, item.family, item.vertical)
            )
            url = url_builder.build(item.item_code)

            enriched.append(
                EnrichedItem(
                    item=item,
                    image_url=url,
                    color=color,
                    category=category,
                    embedding_ref=f"{item.item_code}.npy",
                    embedding_model=backend.model_name,
                    embedding_dim=backend.embedding_dim,
                )
            )

            cat_source = "dataset" if item.category is not None else "classifier"
            logger.info(
                "[{}/{}] ✓ {} → generated (embedding: {}…), color='{}', category='{}' ({})",
                idx,
                len(items),
                item.item_code,
                vec_preview,
                color,
                category.value,
                cat_source,
            )

        except Exception as exc:
            errors.append({"item_code": item.item_code, "error": str(exc)})
            logger.warning("[{}/{}] ✗ {} — error: {}", idx, len(items), item.item_code, exc)

    # ------------------------------------------------------------------
    # 5. Save enriched CSV
    # ------------------------------------------------------------------
    output_path = Path(output) if output else config.storage.output_dir / "enriched_catalog.csv"
    save_enriched_csv(enriched, output_path)

    # ------------------------------------------------------------------
    # 6. Save embeddings
    # ------------------------------------------------------------------
    if all_embeddings:
        embedding_matrix = np.stack(all_embeddings)
        metadata = EmbeddingMetadata(
            model_name=model_name,
            model_version=model_version,
            embedding_dim=embedding_dim or embedding_matrix.shape[1],
            total_items=len(all_item_codes),
            backend_key=backend.name if backend else "marqo",
        )
        store.save(embedding_matrix, all_item_codes, metadata)

    duration = time.time() - start

    # ------------------------------------------------------------------
    # 7. Report
    # ------------------------------------------------------------------
    newly_generated = len(items_to_fetch) - len(errors)
    report = PipelineReport(
        total_items=len(items),
        successful=len(enriched),
        failed=len(errors),
        skipped=skipped_count,
        errors=errors,
        duration_seconds=duration,
    )

    logger.info("Embedding generation complete in {:.1f}s", duration)
    logger.info(
        "Results: {}/{} total ({} reused, {} newly generated, {} failed)",
        report.successful,
        report.total_items,
        skipped_count,
        newly_generated,
        report.failed,
    )
    if errors:
        logger.warning("Errors:")
        for err in errors:
            logger.warning("  {} → {}", err["item_code"], err["error"])

    typer.echo(f"\n✓ Embedding generation complete: {report.successful}/{report.total_items} items")
    typer.echo(f"  Reused:     {skipped_count}")
    typer.echo(f"  Generated:  {newly_generated}")
    typer.echo(f"  Failed:     {report.failed}")
    typer.echo(f"  Output:     {output_path}")
    typer.echo(f"  Embeddings: {config.storage.embedding_dir}")
    typer.echo(f"  Duration:   {duration:.1f}s")


@app.command()
def identify(
    image_path: str = typer.Argument(..., help="Path to the query image"),
    top_k: int = typer.Option(5, help="Number of results to return"),
    embedding_dir: str = typer.Option("", help="Embedding directory (default: from config)"),
    category: str = typer.Option(
        "",
        "--category",
        "-c",
        help="Force category filter (shoe, apparel, accessory). "
        "If omitted, the category is detected automatically via zero-shot CLIP classification.",
    ),
) -> None:
    """Identify a product image by finding the closest match in the catalog.

    The query image category is detected automatically using zero-shot CLIP
    classification: the image embedding is compared against descriptive text
    prompts for each category (shoe, apparel, accessory). Results are then
    filtered to only show items of the detected category.

    Use --category to override the automatic detection (e.g. when the model
    is uncertain or the image is ambiguous).
    """
    config = AppConfig()
    setup_logging(config.logging)
    start = time.time()

    import numpy as np
    from PIL import Image

    # Import backend to trigger registration
    import product_image_id.embeddings.marqo_backend  # noqa: F401
    from product_image_id.domain.models import Category
    from product_image_id.embeddings.interface import get_backend
    from product_image_id.features.zero_shot_classifier import classify_image_zero_shot
    from product_image_id.index.in_memory import InMemoryIndex
    from product_image_id.storage.csv_store import load_enriched_csv
    from product_image_id.storage.embedding_store import EmbeddingStore

    # ==================================================================
    # STEP 1/6 — Load query image
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 1/6 — Loading query image")
    logger.info("=" * 60)
    image = Image.open(image_path).convert("RGB")
    file_size_kb = Path(image_path).stat().st_size / 1024
    logger.info(
        "Query image loaded: {} ({}x{}, {:.0f} KB)",
        image_path,
        image.width,
        image.height,
        file_size_kb,
    )

    # ==================================================================
    # STEP 2/6 — Load embedding model
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 2/6 — Loading embedding model")
    logger.info("=" * 60)

    # We need metadata to know which backend/model was used for the index.
    # Load it from the embedding store (lightweight — just reads metadata.json).
    emb_dir = Path(embedding_dir) if embedding_dir else config.storage.embedding_dir
    store = EmbeddingStore(emb_dir)
    if not store.exists():
        typer.echo("✗ No embeddings found. Run 'embedding-generation' first.", err=True)
        raise typer.Exit(code=1)

    metadata = store.load_metadata()
    logger.info(
        "Embedding metadata: model='{}', dim={}, backend='{}'",
        metadata.model_name,
        metadata.embedding_dim,
        metadata.backend_key,
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

    # ==================================================================
    # STEP 3/6 — Generate query image embedding
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 3/6 — Generating query image embedding")
    logger.info("=" * 60)

    embed_start = time.time()
    query_vector = backend.embed_image(image)
    embed_time = time.time() - embed_start
    vec_preview = str(query_vector[:4].tolist())[:40]
    logger.info(
        "Query embedding generated in {:.2f}s (dim={}, preview: {}…)",
        embed_time,
        len(query_vector),
        vec_preview,
    )

    # ==================================================================
    # STEP 4/6 — Detect query category
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 4/6 — Detecting query category")
    logger.info("=" * 60)

    if category:
        # Manual override: user knows what they're uploading
        detected_category = category.strip().lower()
        logger.info("Category manually set by user: '{}'", detected_category)
    else:
        # Zero-shot CLIP classification: compare image against text prompts
        logger.info("Running zero-shot CLIP classification (image vs text prompts)…")
        zs_start = time.time()
        predicted_cat, scores = classify_image_zero_shot(query_vector, backend)
        zs_time = time.time() - zs_start
        detected_category = predicted_cat.value
        logger.info(
            "Zero-shot result in {:.2f}s: category='{}' "
            "(shoe={:.4f}, apparel={:.4f}, accessory={:.4f})",
            zs_time,
            detected_category,
            scores[Category.SHOE],
            scores[Category.APPAREL],
            scores[Category.ACCESSORY],
        )

    # ==================================================================
    # STEP 5/6 — Load catalog embeddings from disk
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 5/6 — Loading catalog embeddings and enriched data")
    logger.info("=" * 60)

    # Load enriched CSV — we need categories and item names for filtering & display
    output_csv = config.storage.output_dir / "enriched_catalog.csv"
    if not output_csv.exists():
        typer.echo(
            "✗ No enriched catalog found. Run 'embedding-generation' first.",
            err=True,
        )
        raise typer.Exit(code=1)

    enriched_df = load_enriched_csv(output_csv)
    codes_series = enriched_df["item_code"].astype(str)
    name_lookup = dict(zip(codes_series, enriched_df["item_name"].astype(str), strict=False))
    url_lookup = dict(zip(codes_series, enriched_df["image_url"].astype(str), strict=False))
    category_lookup = dict(zip(codes_series, enriched_df["category"].astype(str), strict=False))
    logger.info(
        "Enriched catalog loaded: {} items from {}",
        len(enriched_df),
        output_csv,
    )

    # Count categories in the catalog
    cat_counts: dict[str, int] = {}
    for cat_val in category_lookup.values():
        cat_counts[cat_val] = cat_counts.get(cat_val, 0) + 1
    for cat_name, count in sorted(cat_counts.items()):
        logger.info("  Category '{}': {} items", cat_name, count)

    # Load stored embedding vectors
    embeddings = store.load_embeddings()
    item_codes = store.load_item_codes()
    logger.info(
        "Embedding vectors loaded: {} items, dim={}",
        len(item_codes),
        embeddings.shape[1],
    )

    # ==================================================================
    # STEP 6/6 — Filter by category and search
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 6/6 — Searching within category '{}'", detected_category)
    logger.info("=" * 60)

    filtered_indices = [
        i
        for i, code in enumerate(item_codes)
        if category_lookup.get(str(code), "unknown") == detected_category
    ]

    if not filtered_indices:
        logger.warning("No items of category '{}' found in the index!", detected_category)
        typer.echo(f"✗ No items of category '{detected_category}' in the index.", err=True)
        raise typer.Exit(code=1)

    filtered_embeddings = np.stack([embeddings[i] for i in filtered_indices])
    filtered_codes = [item_codes[i] for i in filtered_indices]
    logger.info(
        "Filtered index: {} items of category '{}' (from {} total)",
        len(filtered_codes),
        detected_category,
        len(item_codes),
    )

    filtered_index = InMemoryIndex()
    filtered_index.build(filtered_embeddings, filtered_codes)
    logger.info("Index built, running cosine similarity search (top_k={})…", top_k)

    search_start = time.time()
    matches = filtered_index.search(query_vector, top_k=top_k)
    search_time = time.time() - search_start
    logger.info("Search completed in {:.4f}s — {} matches found", search_time, len(matches))

    # Log each match
    for i, match in enumerate(matches, 1):
        code = str(match.item_code)
        item_name = name_lookup.get(code, "—")
        logger.info(
            "  #{} score={:.4f} → {} ({})",
            i,
            match.score,
            code,
            item_name,
        )

    total_time = time.time() - start
    logger.info("=" * 60)
    logger.info("DONE — Total identify time: {:.2f}s", total_time)
    logger.info("=" * 60)

    # Display results
    typer.echo(f"\n🔍 Top-{top_k} matches for: {image_path}")
    typer.echo(f"   Category: {detected_category} ({len(filtered_codes)} items in index)")
    typer.echo(f"   Duration: {total_time:.2f}s\n")
    typer.echo(f"{'Rank':<6}{'Score':<10}{'Item Code':<20}{'Item Name':<45}{'Image URL'}")
    typer.echo("-" * 160)
    for i, match in enumerate(matches, 1):
        code = str(match.item_code)
        item_name = name_lookup.get(code, "—")
        image_url = url_lookup.get(code, "—")
        typer.echo(f"{i:<6}{match.score:<10.4f}{code:<20}{item_name:<45}{image_url}")


@app.command()
def evaluate(
    top_k: int = typer.Option(5, help="Top-K for evaluation"),
    limit: int = typer.Option(0, help="Limit golden entries (0 = all)"),
) -> None:
    """Evaluate identification accuracy against the golden dataset."""
    config = AppConfig()
    setup_logging(config.logging)

    from product_image_id.eval.evaluator import evaluate_from_files

    metrics = evaluate_from_files(
        golden_path=config.storage.golden_dataset,
        embedding_dir=config.storage.embedding_dir,
        limit=limit if limit > 0 else None,
        top_k=top_k,
    )

    summary = metrics.summary()

    typer.echo("\n📊 Evaluation Results\n")
    typer.echo(f"  Total queries:    {int(summary['total_queries'])}")
    typer.echo(f"  Top-1 accuracy:   {summary['top_1_accuracy']:.1%}")
    typer.echo(f"  Top-3 accuracy:   {summary['top_3_accuracy']:.1%}")
    typer.echo(f"  Top-5 accuracy:   {summary['top_5_accuracy']:.1%}")
    typer.echo(f"  MRR:              {summary['mrr']:.4f}")


def run_embedding_generation() -> None:
    """Standalone entrypoint for `uv run embedding-generation`."""
    app(["embedding-generation", *_get_sys_args()])


def run_identify() -> None:
    """Standalone entrypoint for `uv run identify`."""
    app(["identify", *_get_sys_args()])


def run_evaluate() -> None:
    """Standalone entrypoint for `uv run evaluate`."""
    app(["evaluate", *_get_sys_args()])


def _get_sys_args() -> list[str]:
    import sys

    return sys.argv[1:]


if __name__ == "__main__":
    app()
