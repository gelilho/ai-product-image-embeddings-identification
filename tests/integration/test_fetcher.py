"""Integration tests for the image fetcher with mocked HTTP."""

import asyncio

import httpx
import pytest
import respx

from product_image_id.config import FetcherConfig
from product_image_id.images.fetcher import ImageFetcher

BASE_URL = "https://onretailimages.blob.core.windows.net/prod-mediaserver/Products"
SUFFIX = "_000_001.png"

# Minimal valid PNG bytes (1x1 transparent pixel)
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def config() -> FetcherConfig:
    return FetcherConfig(
        base_url=BASE_URL,
        image_suffix=SUFFIX,
        concurrency=2,
        rate_limit=10.0,  # Fast for tests
        max_bytes=10_000_000,
        timeout=5,
    )


@pytest.fixture
def fetcher(config: FetcherConfig, tmp_path: object) -> ImageFetcher:
    from pathlib import Path

    return ImageFetcher(config=config, cache_dir=Path(str(tmp_path)) / "cache")


@pytest.mark.integration
class TestImageFetcher:
    """Integration tests with mocked HTTP responses."""

    @respx.mock
    def test_fetch_success(self, fetcher: ImageFetcher) -> None:
        url = f"{BASE_URL}/59.98842{SUFFIX}"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=MINIMAL_PNG,
                headers={"content-type": "image/png"},
            )
        )

        result = asyncio.run(fetcher.fetch_one("59.98842"))
        assert result.success is True
        assert result.image_bytes == MINIMAL_PNG
        assert result.size_bytes == len(MINIMAL_PNG)

    @respx.mock
    def test_fetch_404(self, fetcher: ImageFetcher) -> None:
        url = f"{BASE_URL}/missing.item{SUFFIX}"
        respx.get(url).mock(return_value=httpx.Response(404))

        result = asyncio.run(fetcher.fetch_one("missing.item"))
        assert result.success is False
        assert result.error is not None

    @respx.mock
    def test_fetch_rate_limit_retry(self, fetcher: ImageFetcher) -> None:
        url = f"{BASE_URL}/rate.limited{SUFFIX}"
        # First call: 429, second: 200
        respx.get(url).mock(
            side_effect=[
                httpx.Response(429),
                httpx.Response(429),
                httpx.Response(200, content=MINIMAL_PNG, headers={"content-type": "image/png"}),
            ]
        )

        result = asyncio.run(fetcher.fetch_one("rate.limited"))
        assert result.success is True

    @respx.mock
    def test_fetch_too_large(self, config: FetcherConfig, tmp_path: object) -> None:
        from pathlib import Path

        small_config = FetcherConfig(
            base_url=config.base_url,
            image_suffix=config.image_suffix,
            concurrency=1,
            rate_limit=10.0,
            max_bytes=10,  # Very small limit
            timeout=5,
        )
        fetcher = ImageFetcher(config=small_config, cache_dir=Path(str(tmp_path)) / "cache2")
        url = f"{BASE_URL}/big.item{SUFFIX}"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=MINIMAL_PNG,  # Larger than 10 bytes
                headers={"content-type": "image/png"},
            )
        )

        result = asyncio.run(fetcher.fetch_one("big.item"))
        assert result.success is False

    @respx.mock
    def test_fetch_batch(self, fetcher: ImageFetcher) -> None:
        codes = ["item.001", "item.002", "item.003"]
        for code in codes:
            url = f"{BASE_URL}/{code}{SUFFIX}"
            respx.get(url).mock(
                return_value=httpx.Response(
                    200,
                    content=MINIMAL_PNG,
                    headers={"content-type": "image/png"},
                )
            )

        results = asyncio.run(fetcher.fetch_batch(codes))
        assert len(results) == 3
        assert all(r.success for r in results)

    @respx.mock
    def test_cache_hit(self, fetcher: ImageFetcher) -> None:
        url = f"{BASE_URL}/cached.item{SUFFIX}"
        route = respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=MINIMAL_PNG,
                headers={"content-type": "image/png"},
            )
        )

        # First fetch
        result1 = asyncio.run(fetcher.fetch_one("cached.item"))
        assert result1.success is True
        assert result1.cached is False

        # Second fetch should hit cache
        result2 = asyncio.run(fetcher.fetch_one("cached.item"))
        assert result2.success is True
        assert result2.cached is True
        assert route.call_count == 1  # Only one HTTP call
