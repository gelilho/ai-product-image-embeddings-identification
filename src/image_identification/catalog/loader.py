"""Catalog loader — reads XLSX or CSV into a list of CatalogItem.

Handles column normalization via schema mapping, gender parsing,
and date parsing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from image_identification.catalog.schema import SchemaConfig, validate_columns
from image_identification.domain.errors import CatalogError
from image_identification.domain.models import CatalogItem, Category, Gender
from image_identification.logging import logger

# ---------------------------------------------------------------------------
# Gender mapping
# ---------------------------------------------------------------------------

_GENDER_MAP: dict[str, Gender] = {
    "men": Gender.MEN,
    "women": Gender.WOMEN,
    "unisex": Gender.UNISEX,
    "kids": Gender.KIDS,
    "kid": Gender.KIDS,
}


def _parse_gender(raw: str) -> Gender:
    return _GENDER_MAP.get(raw.strip().lower(), Gender.UNKNOWN)


# ---------------------------------------------------------------------------
# Category mapping — from golden dataset values to Category enum
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, Category] = {
    "shoe": Category.SHOE,
    "shoes": Category.SHOE,
    "footwear": Category.SHOE,
    "apparel": Category.APPAREL,
    "clothing": Category.APPAREL,
    "accessory": Category.ACCESSORY,
    "accessories": Category.ACCESSORY,
}


def _parse_category(raw: object) -> Category | None:
    """Parse a category value from the golden dataset.

    Returns None if the value is empty/NaN — the caller should then
    fall back to the rule-based classifier.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and pd.isna(raw):
        return None
    text = str(raw).strip().lower()
    if not text or text == "nan":
        return None
    return _CATEGORY_MAP.get(text)


def _parse_date(raw: object) -> datetime | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw).replace(" ", "T"))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_catalog(
    path: Path | str,
    *,
    schema: SchemaConfig | None = None,
    limit: int | None = None,
) -> list[CatalogItem]:
    """Load a product catalog from XLSX or CSV.

    Parameters
    ----------
    path:
        Path to the input file (.xlsx or .csv).
    schema:
        Optional schema config for column mapping.
    limit:
        If set, only load the first N rows (useful for POC).

    Returns
    -------
    list[CatalogItem]
        Parsed and validated catalog items.

    Raises
    ------
    CatalogError
        If the file cannot be loaded or has invalid data.
    """
    path = Path(path)
    logger.info("Loading catalog from {}", path)

    if not path.exists():
        raise CatalogError(f"Catalog file not found: {path}")

    try:
        if path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path, engine="openpyxl")
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise CatalogError(f"Unsupported file format: {path.suffix}")
    except CatalogError:
        raise
    except Exception as exc:
        raise CatalogError(f"Failed to read {path}: {exc}") from exc

    # Validate and map columns
    cfg = schema or SchemaConfig.default()
    mapped_cols = validate_columns(list(df.columns), cfg)
    df.columns = pd.Index(mapped_cols)

    if limit is not None:
        df = df.head(limit)

    # Parse rows into domain objects
    has_category_col = "category" in df.columns
    if has_category_col:
        logger.info("Category column found in dataset — will use golden dataset categories")
    else:
        logger.info("No category column in dataset — will fall back to rule-based classifier")

    items: list[CatalogItem] = []
    for idx, row in df.iterrows():
        try:
            category = _parse_category(row.get("category")) if has_category_col else None
            item = CatalogItem(
                item_code=str(row["item_code"]).strip(),
                item_name=str(row["item_name"]).strip(),
                vertical=str(row["vertical"]).strip(),
                family=str(row["family"]).strip(),
                gender=_parse_gender(str(row["gender"])),
                phase_in_date=_parse_date(row.get("phase_in_date")),
                category=category,
            )
            items.append(item)
        except Exception as exc:
            logger.warning("Skipping row {}: {}", idx, exc)

    logger.info("Loaded {} catalog items from {}", len(items), path.name)
    return items
