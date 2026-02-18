"""Token-bucket rate limiter for safe image fetching.

Ensures we never exceed the configured requests-per-second,
preventing accidental DoS against the image host.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """Async-compatible token bucket rate limiter.

    Parameters
    ----------
    rate:
        Maximum requests per second.
    burst:
        Maximum burst size (tokens available at once). Defaults to rate.
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = rate
        self._burst = burst or max(1, int(rate))
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait = (1.0 - self._tokens) / self._rate
                # Release lock while sleeping
                self._lock.release()
                try:
                    await asyncio.sleep(wait)
                finally:
                    await self._lock.acquire()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens (approximate)."""
        self._refill()
        return self._tokens
