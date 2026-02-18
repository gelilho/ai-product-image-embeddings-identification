"""Centralised configuration loaded from environment variables.

All config is read-only after initialization. No secrets in code — only
env vars and .env files (via a caller's dotenv loader if desired).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw is not None else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw is not None else default


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetcherConfig:
    """Image fetcher configuration."""

    base_url: str = field(
        default_factory=lambda: _env(
            "PII_IMAGE_BASE_URL",
            "https://onretailimages.blob.core.windows.net/prod-mediaserver/Products",
        )
    )
    image_suffix: str = field(default_factory=lambda: _env("PII_IMAGE_SUFFIX", "_000_001.png"))
    concurrency: int = field(default_factory=lambda: _env_int("PII_FETCH_CONCURRENCY", 3))
    rate_limit: float = field(default_factory=lambda: _env_float("PII_FETCH_RATE_LIMIT", 2.0))
    max_bytes: int = field(default_factory=lambda: _env_int("PII_FETCH_MAX_BYTES", 10_485_760))
    timeout: int = field(default_factory=lambda: _env_int("PII_FETCH_TIMEOUT", 30))


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding backend configuration."""

    backend: str = field(default_factory=lambda: _env("PII_EMBEDDING_BACKEND", "marqo"))
    model_name: str = field(
        default_factory=lambda: _env("PII_EMBEDDING_MODEL", "Marqo/marqo-ecommerce-embeddings-L")
    )
    device: str = field(default_factory=lambda: _env("PII_EMBEDDING_DEVICE", ""))


@dataclass(frozen=True)
class StorageConfig:
    """Storage paths configuration."""

    golden_dataset: Path = field(
        default_factory=lambda: Path(
            _env("PII_GOLDEN_DATASET", "data/golden/product_items_golden.xlsx")
        )
    )
    output_dir: Path = field(default_factory=lambda: Path(_env("PII_OUTPUT_DIR", "data/output")))
    embedding_dir: Path = field(
        default_factory=lambda: Path(_env("PII_EMBEDDING_DIR", "data/embeddings"))
    )
    cache_dir: Path = field(default_factory=lambda: Path(_env("PII_CACHE_DIR", "data/cache")))


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: _env("PII_LOG_LEVEL", "INFO"))
    format: str = field(default_factory=lambda: _env("PII_LOG_FORMAT", "text"))


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration — aggregates all sub-configs."""

    fetcher: FetcherConfig = field(default_factory=FetcherConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def log_summary(self) -> dict[str, object]:
        """Return a non-sensitive summary for startup logging."""
        return {
            "image_base_url": self.fetcher.base_url,
            "image_suffix": self.fetcher.image_suffix,
            "fetch_concurrency": self.fetcher.concurrency,
            "fetch_rate_limit": self.fetcher.rate_limit,
            "embedding_backend": self.embedding.backend,
            "embedding_model": self.embedding.model_name,
            "embedding_device": self.embedding.device or "auto",
            "golden_dataset": str(self.storage.golden_dataset),
            "output_dir": str(self.storage.output_dir),
            "log_level": self.logging.level,
        }
