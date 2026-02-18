"""Tests for the catalog loader."""

from pathlib import Path

import pandas as pd
import pytest

from image_identification.catalog.loader import load_catalog
from image_identification.domain.errors import CatalogError
from image_identification.domain.models import Category, Gender


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal valid CSV for testing."""
    df = pd.DataFrame(
        {
            "dim_d365_item_vertical_name": ["Performance Running", "Performance All Day"],
            "dim_d365_item_retail_level_3_family": ["Cloud", "All Day"],
            "dim_d365_item_gender_name": ["Men", "Women"],
            "dim_d365_item_item_name_en_us": ["Cloud 5 M Black", "Active Jacket 1 W White"],
            "dim_d365_item_item_code": ["59.98842", "130.01071"],
            "dim_d365_item_phase_in_date": ["2023-01-15", "2023-07-06"],
            "dim_d365_item_item_image": ["59.98842", "130.01071"],
        }
    )
    path = tmp_path / "test_catalog.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_csv_with_category(tmp_path: Path) -> Path:
    """Create a CSV that includes a category column."""
    df = pd.DataFrame(
        {
            "dim_d365_item_vertical_name": [
                "Performance Running",
                "Performance All Day",
                "Accessories",
            ],
            "dim_d365_item_retail_level_3_family": ["Cloud", "All Day", "Socks"],
            "dim_d365_item_gender_name": ["Men", "Women", "Unisex"],
            "dim_d365_item_item_name_en_us": [
                "Cloud 5 M Black",
                "Active Jacket 1 W White",
                "All-Day Sock 1 U Doe",
            ],
            "dim_d365_item_item_code": ["59.98842", "130.01071", "140.00001"],
            "dim_d365_item_phase_in_date": ["2023-01-15", "2023-07-06", "2023-09-01"],
            "dim_d365_item_item_image": ["59.98842", "130.01071", "140.00001"],
            "dim_d365_item_category": ["Shoe", "Apparel", "Accessory"],
        }
    )
    path = tmp_path / "test_catalog_with_category.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadCatalog:
    """Test catalog loading from CSV files."""

    def test_load_valid_csv(self, sample_csv: Path) -> None:
        items = load_catalog(sample_csv)
        assert len(items) == 2

    def test_item_fields(self, sample_csv: Path) -> None:
        items = load_catalog(sample_csv)
        item = items[0]
        assert item.item_code == "59.98842"
        assert item.item_name == "Cloud 5 M Black"
        assert item.vertical == "Performance Running"
        assert item.family == "Cloud"
        assert item.gender == Gender.MEN

    def test_limit_parameter(self, sample_csv: Path) -> None:
        items = load_catalog(sample_csv, limit=1)
        assert len(items) == 1

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogError, match="not found"):
            load_catalog(tmp_path / "nonexistent.csv")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        path.write_text("{}")
        with pytest.raises(CatalogError, match="Unsupported"):
            load_catalog(path)

    def test_gender_parsing(self, sample_csv: Path) -> None:
        items = load_catalog(sample_csv)
        assert items[0].gender == Gender.MEN
        assert items[1].gender == Gender.WOMEN

    # ------------------------------------------------------------------
    # Category from golden dataset
    # ------------------------------------------------------------------

    def test_no_category_column_returns_none(self, sample_csv: Path) -> None:
        """Items loaded from a CSV without a category column should have category=None."""
        items = load_catalog(sample_csv)
        for item in items:
            assert item.category is None

    def test_category_column_parsed(self, sample_csv_with_category: Path) -> None:
        """Items loaded from a CSV with a category column should have the correct category."""
        items = load_catalog(sample_csv_with_category)
        assert len(items) == 3
        assert items[0].category == Category.SHOE
        assert items[1].category == Category.APPAREL
        assert items[2].category == Category.ACCESSORY

    def test_category_case_insensitive(self, tmp_path: Path) -> None:
        """Category values should be parsed case-insensitively."""
        df = pd.DataFrame(
            {
                "dim_d365_item_vertical_name": ["V1", "V2", "V3"],
                "dim_d365_item_retail_level_3_family": ["F1", "F2", "F3"],
                "dim_d365_item_gender_name": ["Men", "Women", "Men"],
                "dim_d365_item_item_name_en_us": ["P1", "P2", "P3"],
                "dim_d365_item_item_code": ["001", "002", "003"],
                "dim_d365_item_category": ["SHOE", "apparel", "Accessory"],
            }
        )
        path = tmp_path / "case_test.csv"
        df.to_csv(path, index=False)
        items = load_catalog(path)
        assert items[0].category == Category.SHOE
        assert items[1].category == Category.APPAREL
        assert items[2].category == Category.ACCESSORY

    def test_category_synonyms(self, tmp_path: Path) -> None:
        """Alternative category names (footwear, clothing, accessories) should map correctly."""
        df = pd.DataFrame(
            {
                "dim_d365_item_vertical_name": ["V1", "V2", "V3"],
                "dim_d365_item_retail_level_3_family": ["F1", "F2", "F3"],
                "dim_d365_item_gender_name": ["Men", "Women", "Men"],
                "dim_d365_item_item_name_en_us": ["P1", "P2", "P3"],
                "dim_d365_item_item_code": ["001", "002", "003"],
                "dim_d365_item_category": ["Footwear", "Clothing", "Accessories"],
            }
        )
        path = tmp_path / "synonym_test.csv"
        df.to_csv(path, index=False)
        items = load_catalog(path)
        assert items[0].category == Category.SHOE
        assert items[1].category == Category.APPAREL
        assert items[2].category == Category.ACCESSORY

    def test_empty_category_returns_none(self, tmp_path: Path) -> None:
        """Empty or NaN category values should result in category=None."""
        df = pd.DataFrame(
            {
                "dim_d365_item_vertical_name": ["V1", "V2"],
                "dim_d365_item_retail_level_3_family": ["F1", "F2"],
                "dim_d365_item_gender_name": ["Men", "Women"],
                "dim_d365_item_item_name_en_us": ["P1", "P2"],
                "dim_d365_item_item_code": ["001", "002"],
                "dim_d365_item_category": ["Shoe", ""],
            }
        )
        path = tmp_path / "empty_cat_test.csv"
        df.to_csv(path, index=False)
        items = load_catalog(path)
        assert items[0].category == Category.SHOE
        assert items[1].category is None

    def test_unknown_category_returns_none(self, tmp_path: Path) -> None:
        """Unrecognized category values should result in category=None (fallback to classifier)."""
        df = pd.DataFrame(
            {
                "dim_d365_item_vertical_name": ["V1"],
                "dim_d365_item_retail_level_3_family": ["F1"],
                "dim_d365_item_gender_name": ["Men"],
                "dim_d365_item_item_name_en_us": ["P1"],
                "dim_d365_item_item_code": ["001"],
                "dim_d365_item_category": ["SomethingWeird"],
            }
        )
        path = tmp_path / "unknown_cat_test.csv"
        df.to_csv(path, index=False)
        items = load_catalog(path)
        assert items[0].category is None
