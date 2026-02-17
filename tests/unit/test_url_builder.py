"""Tests for the URL builder module."""

import pytest

from product_image_id.config import FetcherConfig
from product_image_id.images.url_builder import UrlBuilder

BASE_URL = "https://onretailimages.blob.core.windows.net/prod-mediaserver/Products"
SUFFIX = "_000_001.png"


@pytest.fixture
def builder() -> UrlBuilder:
    config = FetcherConfig(
        base_url=BASE_URL,
        image_suffix=SUFFIX,
        concurrency=1,
        rate_limit=1.0,
        max_bytes=10_000_000,
        timeout=10,
    )
    return UrlBuilder(config)


class TestUrlBuilder:
    """Test URL construction for both item code formats."""

    def test_legacy_code(self, builder: UrlBuilder) -> None:
        """Legacy codes like 59.98842 should produce valid URLs."""
        url = builder.build("59.98842")
        assert url == f"{BASE_URL}/59.98842{SUFFIX}"

    def test_new_code_men(self, builder: UrlBuilder) -> None:
        """New format codes like 1MD10060553 should produce valid URLs."""
        url = builder.build("1MD10060553")
        assert url == f"{BASE_URL}/1MD10060553{SUFFIX}"

    def test_new_code_women(self, builder: UrlBuilder) -> None:
        url = builder.build("1WD10090585")
        assert url == f"{BASE_URL}/1WD10090585{SUFFIX}"

    def test_new_code_kids(self, builder: UrlBuilder) -> None:
        url = builder.build("3KD11430070")
        assert url == f"{BASE_URL}/3KD11430070{SUFFIX}"

    def test_strips_whitespace(self, builder: UrlBuilder) -> None:
        url = builder.build("  311.00218  ")
        assert url == f"{BASE_URL}/311.00218{SUFFIX}"

    def test_empty_code_raises(self, builder: UrlBuilder) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            builder.build("")

    def test_blank_code_raises(self, builder: UrlBuilder) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            builder.build("   ")

    def test_build_many(self, builder: UrlBuilder) -> None:
        codes = ["59.98842", "1MD10060553", "311.00218"]
        result = builder.build_many(codes)
        assert len(result) == 3
        for code in codes:
            assert code in result
            assert result[code].endswith(SUFFIX)

    def test_trailing_slash_in_base_url(self) -> None:
        """Base URL with trailing slash should not produce double slash."""
        config = FetcherConfig(
            base_url=f"{BASE_URL}/",
            image_suffix=SUFFIX,
            concurrency=1,
            rate_limit=1.0,
            max_bytes=10_000_000,
            timeout=10,
        )
        builder = UrlBuilder(config)
        url = builder.build("59.98842")
        assert "//" not in url.replace("https://", "")
