"""Tests for schema validation and column mapping."""

import pytest

from product_image_id.catalog.schema import COLUMN_MAP, SchemaConfig, validate_columns
from product_image_id.domain.errors import SchemaValidationError


class TestValidateColumns:
    """Test column validation with the D365 column mapping."""

    def test_valid_d365_columns(self) -> None:
        """All D365 raw columns should map correctly."""
        raw_columns = list(COLUMN_MAP.keys())
        mapped = validate_columns(raw_columns)
        assert "item_code" in mapped
        assert "item_name" in mapped
        assert "vertical" in mapped
        assert "family" in mapped
        assert "gender" in mapped

    def test_already_mapped_columns(self) -> None:
        """Pre-mapped internal column names should pass."""
        columns = ["vertical", "family", "gender", "item_name", "item_code", "extra"]
        mapped = validate_columns(columns)
        assert "item_code" in mapped

    def test_missing_required_column_raises(self) -> None:
        """Missing required columns should raise SchemaValidationError."""
        columns = ["vertical", "family"]  # missing gender, item_name, item_code
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_columns(columns)
        assert len(exc_info.value.missing_columns) > 0

    def test_extra_columns_preserved(self) -> None:
        """Extra columns beyond required should be kept."""
        columns = [*list(COLUMN_MAP.keys()), "extra_column"]
        mapped = validate_columns(columns)
        assert "extra_column" in mapped

    def test_custom_schema(self) -> None:
        """Custom schema config should override default mapping."""
        custom = SchemaConfig(
            column_map={"col_a": "item_code", "col_b": "item_name"},
            required_columns=frozenset({"item_code", "item_name"}),
        )
        mapped = validate_columns(["col_a", "col_b", "col_c"], schema=custom)
        assert "item_code" in mapped
        assert "item_name" in mapped

    def test_error_message_lists_missing(self) -> None:
        """Error message should include which columns are missing."""
        with pytest.raises(SchemaValidationError, match="item_code"):
            validate_columns(["vertical", "family", "gender", "item_name"])
