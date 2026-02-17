"""CSV storage adapter for enriched catalog data.

Reads and writes the enriched CSV that includes derived attributes
(color, category) and embedding references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from product_image_id.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    from product_image_id.domain.models import EnrichedItem

# Output column order
OUTPUT_COLUMNS: list[str] = [
    "item_code",
    "item_name",
    "vertical",
    "family",
    "gender",
    "image_url",
    "color",
    "category",
    "embedding_ref",
    "embedding_model",
    "embedding_dim",
]


def save_enriched_csv(items: list[EnrichedItem], path: Path) -> None:
    """Save enriched items to a CSV file.

    Parameters
    ----------
    items:
        List of enriched catalog items.
    path:
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ei in items:
        rows.append(
            {
                "item_code": ei.item.item_code,
                "item_name": ei.item.item_name,
                "vertical": ei.item.vertical,
                "family": ei.item.family,
                "gender": ei.item.gender.value,
                "image_url": ei.image_url,
                "color": ei.color,
                "category": ei.category.value,
                "embedding_ref": ei.embedding_ref,
                "embedding_model": ei.embedding_model,
                "embedding_dim": ei.embedding_dim,
            }
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(path, index=False)
    logger.info("Saved {} enriched items to {}", len(items), path)


def load_enriched_csv(path: Path) -> pd.DataFrame:
    """Load an enriched CSV back into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Enriched CSV not found: {path}")
    return pd.read_csv(path)
