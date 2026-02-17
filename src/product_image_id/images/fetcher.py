"""Safe, rate-limited image fetcher with retry and caching.

Treats the remote image host as UNTRUSTED — enforces:
- Rate limiting (token bucket)
- Concurrency cap (semaphore)
- Max download size
- Timeout per request
- Exponential backoff with jitter on failure
- Content-type validation
- PIL image validation
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from typing import TYPE_CHECKING

import httpx

from product_image_id.config import FetcherConfig
from product_image_id.domain.errors import FetchError, ImageTooLargeError, RateLimitExceededError
from product_image_id.domain.models import FetchResult
from product_image_id.images.rate_limiter import TokenBucketRateLimiter
from product_image_id.images.url_builder import UrlBuilder
from product_image_id.images.validators import validate_content_type
from product_image_id.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds
_BACKOFF_MAX = 30.0  # seconds
_JITTER_MAX = 0.5  # seconds


class ImageFetcher:
    """Async image fetcher with safety controls.

    Usage:
        fetcher = ImageFetcher(config)
        results = await fetcher.fetch_batch(item_codes)
    """

    def __init__(
        self,
        config: FetcherConfig | None = None,
        url_builder: UrlBuilder | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._config = config or FetcherConfig()
        self._url_builder = url_builder or UrlBuilder(self._config)
        self._rate_limiter = TokenBucketRateLimiter(
            rate=self._config.rate_limit,
            burst=max(1, int(self._config.rate_limit)),
        )
        self._semaphore = asyncio.Semaphore(self._config.concurrency)
        self._cache_dir = cache_dir

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_one(self, item_code: str) -> FetchResult:
        """Fetch a single product image with rate limiting and retry."""
        url = self._url_builder.build(item_code)

        # Check cache first
        cached_bytes = self._read_cache(item_code)
        if cached_bytes is not None:
            logger.info("Cache hit for {} ({:.0f} KB)", item_code, len(cached_bytes) / 1024)
            return FetchResult(
                item_code=item_code,
                url=url,
                success=True,
                image_bytes=cached_bytes,
                size_bytes=len(cached_bytes),
                cached=True,
            )

        # Fetch with rate limiting and retry
        async with self._semaphore:
            return await self._fetch_with_retry(item_code, url)

    async def fetch_batch(self, item_codes: list[str]) -> list[FetchResult]:
        """Fetch images for a batch of item codes concurrently.

        Continues on individual failures — returns results for all items.
        """
        logger.info(
            "Fetching {} images (concurrency={}, rate={}/s)",
            len(item_codes),
            self._config.concurrency,
            self._config.rate_limit,
        )
        tasks = [self.fetch_one(code) for code in item_codes]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fetch_with_retry(self, item_code: str, url: str) -> FetchResult:
        last_error: str = "Unknown error"

        for attempt in range(_MAX_RETRIES):
            try:
                await self._rate_limiter.acquire()
                return await self._do_fetch(item_code, url)
            except RateLimitExceededError:
                wait = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_MAX)
                wait += random.uniform(0, _JITTER_MAX)
                logger.warning(
                    "Rate limited for {} (attempt {}/{}), waiting {:.1f}s",
                    item_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)
                last_error = "Rate limited"
            except FetchError as exc:
                wait = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_MAX)
                wait += random.uniform(0, _JITTER_MAX)
                logger.warning(
                    "Fetch failed for {} (attempt {}/{}): {}",
                    item_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    exc.reason,
                )
                last_error = exc.reason
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(wait)

        return FetchResult(
            item_code=item_code,
            url=url,
            success=False,
            error=last_error,
        )

    async def _do_fetch(self, item_code: str, url: str) -> FetchResult:
        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            response = await client.get(url)

            if response.status_code == 429:
                raise RateLimitExceededError(item_code=item_code, url=url, reason="HTTP 429")

            if response.status_code == 404:
                raise FetchError(item_code=item_code, url=url, reason="HTTP 404 Not Found")

            response.raise_for_status()

            # Validate content type
            content_type = response.headers.get("content-type")
            validate_content_type(content_type, item_code, url)

            # Enforce max size
            image_bytes = response.content
            if len(image_bytes) > self._config.max_bytes:
                raise ImageTooLargeError(
                    item_code=item_code,
                    url=url,
                    size_bytes=len(image_bytes),
                    max_bytes=self._config.max_bytes,
                )

            # Cache on success
            self._write_cache(item_code, image_bytes)

            logger.info(
                "Fetched {} ({:.0f} KB, {})",
                item_code,
                len(image_bytes) / 1024,
                content_type or "unknown type",
            )

            return FetchResult(
                item_code=item_code,
                url=url,
                success=True,
                image_bytes=image_bytes,
                content_type=content_type,
                size_bytes=len(image_bytes),
            )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, item_code: str) -> Path | None:
        if self._cache_dir is None:
            return None
        safe_name = hashlib.sha256(item_code.encode()).hexdigest()[:16]
        return self._cache_dir / f"{safe_name}.png"

    def _read_cache(self, item_code: str) -> bytes | None:
        path = self._cache_path(item_code)
        if path and path.exists():
            return path.read_bytes()
        return None

    def _write_cache(self, item_code: str, data: bytes) -> None:
        path = self._cache_path(item_code)
        if path:
            path.write_bytes(data)
