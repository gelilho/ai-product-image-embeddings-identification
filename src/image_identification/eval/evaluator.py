"""Evaluation harness — run identification against the golden dataset.

Orchestrates: load golden → for each entry, identify → record metrics → report.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from image_identification.eval.golden_loader import load_golden_dataset
from image_identification.eval.metrics import EvalMetrics
from image_identification.logging import logger

if TYPE_CHECKING:
    import numpy as np


def evaluate_from_files(
    golden_path: Path | str,
    embedding_dir: Path | str,
    *,
    limit: int | None = None,
    top_k: int = 5,
) -> EvalMetrics:
    """Convenience function: load artifacts from disk and evaluate.

    Parameters
    ----------
    golden_path:
        Path to the golden dataset.
    embedding_dir:
        Directory containing embeddings.npy, item_codes.json, metadata.json.
    limit:
        Optional limit on golden entries.
    top_k:
        Top-K to evaluate.

    Returns
    -------
    EvalMetrics
        Evaluation results.
    """

    from image_identification.index.in_memory import InMemoryIndex
    from image_identification.storage.embedding_store import EmbeddingStore

    # Load golden dataset
    golden_entries = load_golden_dataset(golden_path, limit=limit)

    # Load stored embeddings
    store = EmbeddingStore(Path(embedding_dir))
    embeddings = store.load_embeddings()
    item_codes = store.load_item_codes()

    # Build index
    index = InMemoryIndex()
    index.build(embeddings, item_codes)

    # Create embeddings lookup
    embeddings_by_code: dict[str, np.ndarray] = {
        code: embeddings[i] for i, code in enumerate(item_codes)
    }

    # Self-identification eval: each item's embedding queries the index
    metrics = EvalMetrics()
    for entry in golden_entries:
        embedding = embeddings_by_code.get(entry.expected_code)
        if embedding is None:
            logger.warning("No embedding for {}, skipping", entry.expected_code)
            continue

        matches = index.search(embedding, top_k=top_k)
        result_codes = [m.item_code for m in matches]
        metrics.record(entry.expected_code, result_codes)

    logger.info("Evaluation complete: {} queries", metrics.total_queries)
    return metrics
