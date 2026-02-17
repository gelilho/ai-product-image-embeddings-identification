"""Tests for domain models."""

from product_image_id.domain.errors import (
    BackendNotFoundError,
    CatalogError,
    EmbeddingError,
    FetchError,
    ImageTooLargeError,
    PIIError,
    SchemaValidationError,
    StorageError,
)
from product_image_id.domain.models import (
    CatalogItem,
    Category,
    Gender,
    IdentificationMatch,
    IdentificationResult,
    PipelineReport,
)


class TestCatalogItem:
    """Test CatalogItem domain model."""

    def test_display_name(self) -> None:
        item = CatalogItem(
            item_code="59.98842",
            item_name="Cloud 5 M Black",
            vertical="Performance Running",
            family="Cloud",
            gender=Gender.MEN,
        )
        assert "Cloud 5 M Black" in item.display_name
        assert "59.98842" in item.display_name

    def test_category_default_none(self) -> None:
        """Category defaults to None when not provided (backward compatible)."""
        item = CatalogItem(
            item_code="59.98842",
            item_name="Cloud 5 M Black",
            vertical="Performance Running",
            family="Cloud",
            gender=Gender.MEN,
        )
        assert item.category is None

    def test_category_from_dataset(self) -> None:
        """Category can be set from the golden dataset."""
        item = CatalogItem(
            item_code="59.98842",
            item_name="Cloud 5 M Black",
            vertical="Performance Running",
            family="Cloud",
            gender=Gender.MEN,
            category=Category.SHOE,
        )
        assert item.category == Category.SHOE


class TestIdentificationResult:
    """Test identification result."""

    def test_top_match(self) -> None:
        item = CatalogItem(
            item_code="a", item_name="test", vertical="v", family="f", gender=Gender.MEN
        )
        match = IdentificationMatch(item=item, score=0.95, rank=1)
        result = IdentificationResult(
            query_path="test.png",
            matches=[match],
            embedding_model="test",
            embedding_dim=128,
        )
        assert result.top_match is not None
        assert result.top_match.score == 0.95

    def test_empty_matches(self) -> None:
        result = IdentificationResult(
            query_path="test.png",
            matches=[],
            embedding_model="test",
            embedding_dim=128,
        )
        assert result.top_match is None


class TestPipelineReport:
    """Test pipeline report."""

    def test_success_rate(self) -> None:
        report = PipelineReport(total_items=10, successful=8, failed=2, skipped=0)
        assert report.success_rate == 0.8

    def test_success_rate_zero_items(self) -> None:
        report = PipelineReport(total_items=0, successful=0, failed=0, skipped=0)
        assert report.success_rate == 0.0


class TestEnums:
    """Test enum values."""

    def test_category_values(self) -> None:
        assert Category.SHOE.value == "shoe"
        assert Category.APPAREL.value == "apparel"
        assert Category.ACCESSORY.value == "accessory"

    def test_gender_values(self) -> None:
        assert Gender.MEN.value == "Men"
        assert Gender.WOMEN.value == "Women"


class TestErrors:
    """Test error hierarchy."""

    def test_pii_error_base(self) -> None:
        err = PIIError("test")
        assert str(err) == "test"

    def test_catalog_error(self) -> None:
        err = CatalogError("bad data")
        assert isinstance(err, PIIError)

    def test_schema_validation_error(self) -> None:
        err = SchemaValidationError(missing_columns=["a", "b"])
        assert "a" in str(err)
        assert err.missing_columns == ["a", "b"]

    def test_fetch_error(self) -> None:
        err = FetchError("code", "http://url", "timeout")
        assert err.item_code == "code"
        assert "timeout" in str(err)

    def test_image_too_large(self) -> None:
        err = ImageTooLargeError("code", "http://url", 999, 100)
        assert err.size_bytes == 999
        assert err.max_bytes == 100

    def test_embedding_error(self) -> None:
        err = EmbeddingError("code", "model failed")
        assert err.item_code == "code"

    def test_backend_not_found(self) -> None:
        err = BackendNotFoundError("xyz")
        assert err.backend_name == "xyz"

    def test_storage_error(self) -> None:
        assert issubclass(StorageError, PIIError)
