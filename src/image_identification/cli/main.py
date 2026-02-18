"""CLI entrypoint — exposes embedding-generation, identify, and evaluate commands.

Each command is a thin handler that wires configuration, delegates to a
service module, and formats terminal output.

Usage:
    uv run embedding-generation --limit 30
    uv run identify path/to/image.png
    uv run evaluate --top-k 5
"""

from __future__ import annotations

import typer

from image_identification.config import AppConfig
from image_identification.logging import setup_logging

app = typer.Typer(
    name="ai-product-image-embeddings-identification",
    help="Product Image Identification — embedding-based product matching.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Default batch size for paginated fetching (anti-DoS)
# ---------------------------------------------------------------------------
_DEFAULT_BATCH_SIZE = 20


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

    Steps: load catalog -> fetch images (batched) -> embed -> extract attrs -> save.

    Skips items that already have embeddings unless --force is passed.
    For datasets larger than 5 items, images are fetched in batches with
    a cooldown pause between each batch to avoid overwhelming the server.
    """
    config = AppConfig()
    setup_logging(config.logging)

    from image_identification.services.embedding_generation import run_pipeline

    result = run_pipeline(
        config, limit=limit, output=output, batch_size=batch_size, force=force,
    )

    typer.echo(f"\n\u2713 Embedding generation complete: {result.successful}/{result.total_items} items")
    typer.echo(f"  Reused:     {result.skipped}")
    typer.echo(f"  Generated:  {result.newly_generated}")
    typer.echo(f"  Failed:     {result.failed}")
    typer.echo(f"  Output:     {result.output_path}")
    typer.echo(f"  Embeddings: {result.embedding_dir}")
    typer.echo(f"  Duration:   {result.duration_seconds:.1f}s")


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

    from image_identification.domain.errors import (
        EmbeddingsNotFoundError,
        NoCategoryItemsError,
    )
    from image_identification.services.identify import run_identification

    try:
        result = run_identification(
            config,
            image_path=image_path,
            top_k=top_k,
            embedding_dir=embedding_dir,
            category=category,
        )
    except EmbeddingsNotFoundError as exc:
        typer.echo(f"\u2717 {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except NoCategoryItemsError as exc:
        typer.echo(f"\u2717 {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # Display results table
    typer.echo(f"\n\ud83d\udd0d Top-{top_k} matches for: {image_path}")
    typer.echo(f"   Category: {result.detected_category} ({result.filtered_count} items in index)")
    typer.echo(f"   Duration: {result.duration_seconds:.2f}s\n")
    typer.echo(f"{'Rank':<6}{'Score':<10}{'Item Code':<20}{'Item Name':<45}{'Image URL'}")
    typer.echo("-" * 160)
    for i, match in enumerate(result.matches, 1):
        code = str(match.item_code)
        item_name = result.name_lookup.get(code, "\u2014")
        image_url = result.url_lookup.get(code, "\u2014")
        typer.echo(f"{i:<6}{match.score:<10.4f}{code:<20}{item_name:<45}{image_url}")


@app.command()
def evaluate(
    top_k: int = typer.Option(5, help="Top-K for evaluation"),
    limit: int = typer.Option(0, help="Limit golden entries (0 = all)"),
) -> None:
    """Evaluate identification accuracy against the golden dataset."""
    config = AppConfig()
    setup_logging(config.logging)

    from image_identification.eval.evaluator import evaluate_from_files

    metrics = evaluate_from_files(
        golden_path=config.storage.golden_dataset,
        embedding_dir=config.storage.embedding_dir,
        limit=limit if limit > 0 else None,
        top_k=top_k,
    )

    summary = metrics.summary()

    typer.echo("\n\ud83d\udcca Evaluation Results\n")
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
