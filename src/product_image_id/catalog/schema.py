"""Schema definitions and validation for catalog input data.

Supports column mapping from the raw D365 export column names to
clean internal names.
"""

from __future__ import annotations

from dataclasses import dataclass

from product_image_id.domain.errors import SchemaValidationError

# ---------------------------------------------------------------------------
# Column mapping: raw D365 name → internal name
# ---------------------------------------------------------------------------

COLUMN_MAP: dict[str, str] = {
    "dim_d365_item_vertical_name": "vertical",
    "dim_d365_item_retail_level_3_family": "family",
    "dim_d365_item_gender_name": "gender",
    "dim_d365_item_item_name_en_us": "item_name",
    "dim_d365_item_item_code": "item_code",
    "dim_d365_item_phase_in_date": "phase_in_date",
    "dim_d365_item_item_image": "item_image",
    "dim_d365_item_category": "category",
}

REQUIRED_INTERNAL_COLUMNS: frozenset[str] = frozenset(
    {"vertical", "family", "gender", "item_name", "item_code"}
)


@dataclass(frozen=True)
class SchemaConfig:
    """Configurable schema mapping — allows overriding the default D365 map."""

    column_map: dict[str, str]
    required_columns: frozenset[str]

    @classmethod
    def default(cls) -> SchemaConfig:
        return cls(
            column_map=COLUMN_MAP,
            required_columns=REQUIRED_INTERNAL_COLUMNS,
        )


def validate_columns(columns: list[str], schema: SchemaConfig | None = None) -> list[str]:
    """Validate that all required columns are present after mapping.

    Parameters
    ----------
    columns:
        The column names found in the loaded data (raw or already mapped).
    schema:
        Schema configuration. Uses default D365 mapping if not provided.

    Returns
    -------
    list[str]
        The mapped column names (internal names).

    Raises
    ------
    SchemaValidationError
        If any required columns are missing.
    """
    cfg = schema or SchemaConfig.default()

    # Map raw → internal; keep unmapped columns as-is
    mapped = [cfg.column_map.get(c, c) for c in columns]

    missing = cfg.required_columns - set(mapped)
    if missing:
        raise SchemaValidationError(missing_columns=sorted(missing))

    return mapped
