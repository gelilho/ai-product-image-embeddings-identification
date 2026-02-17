"""Tests for the token bucket rate limiter."""

import asyncio
import time

import pytest

from product_image_id.images.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """Test rate limiter behavior."""

    def test_invalid_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucketRateLimiter(rate=0)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucketRateLimiter(rate=-1.0)

    def test_acquire_immediate(self) -> None:
        """First acquire should be immediate if burst is available."""
        limiter = TokenBucketRateLimiter(rate=100.0, burst=10)
        start = time.monotonic()
        asyncio.run(limiter.acquire())
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # Should be nearly instant

    def test_available_tokens(self) -> None:
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)
        assert limiter.available_tokens <= 5.0

    def test_burst_depletes(self) -> None:
        """After exhausting burst, acquire should wait."""
        limiter = TokenBucketRateLimiter(rate=100.0, burst=2)

        async def exhaust_and_check() -> float:
            # Use up the burst
            await limiter.acquire()
            await limiter.acquire()
            # Next acquire should wait
            start = time.monotonic()
            await limiter.acquire()
            return time.monotonic() - start

        elapsed = asyncio.run(exhaust_and_check())
        # Should have waited some time (at least a tiny bit)
        assert elapsed >= 0.005

    def test_default_burst_equals_rate(self) -> None:
        limiter = TokenBucketRateLimiter(rate=5.0)
        assert limiter._burst == 5
