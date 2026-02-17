"""Golden dataset loader for evaluation.

Loads the labeled dataset and provides query/expected pairs
for running the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from product_image_id.catalog.loader import load_catalog
from product_image_id.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    from product_image_id.domain.models import CatalogItem


@dataclass(frozen=True)
class GoldenEntry:
    """A single evaluation entry from the golden dataset."""

    item: CatalogItem
    expected_code: str  # The item code that should be matched


def load_golden_dataset(
    path: Path | str,
    *,
    limit: int | None = None,
) -> list[GoldenEntry]:
    """Load the golden dataset for evaluation.

    Each item in the golden dataset serves as both:
    - The query (its image is used as input)
    - The expected result (its item_code is the ground truth)

    This tests whether the system can correctly re-identify
    products from their own images.

    Parameters
    ----------
    path:
        Path to the golden dataset (XLSX or CSV).
    limit:
        Optional limit on number of entries.

    Returns
    -------
    list[GoldenEntry]
        Parsed golden entries.
    """
    items = load_catalog(path, limit=limit)

    entries = [GoldenEntry(item=item, expected_code=item.item_code) for item in items]

    logger.info("Loaded {} golden entries from {}", len(entries), path)
    return entries
