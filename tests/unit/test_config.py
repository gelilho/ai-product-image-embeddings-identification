"""Tests for application configuration."""

from image_identification.config import (
    AppConfig,
    EmbeddingConfig,
    FetcherConfig,
    LoggingConfig,
    StorageConfig,
)


class TestAppConfig:
    """Test configuration loading."""

    def test_default_config(self) -> None:
        config = AppConfig()
        assert config.fetcher.concurrency == 3
        assert config.fetcher.rate_limit == 2.0
        assert config.embedding.backend == "marqo"
        assert config.logging.level == "INFO"

    def test_log_summary(self) -> None:
        config = AppConfig()
        summary = config.log_summary()
        assert "embedding_backend" in summary
        assert "fetch_rate_limit" in summary
        assert summary["embedding_backend"] == "marqo"

    def test_fetcher_defaults(self) -> None:
        config = FetcherConfig()
        assert "onretailimages.blob.core.windows.net" in config.base_url
        assert config.image_suffix == "_000_001.png"
        assert config.max_bytes == 10_485_760

    def test_embedding_defaults(self) -> None:
        config = EmbeddingConfig()
        assert config.backend == "marqo"
        assert "marqo-ecommerce" in config.model_name.lower()

    def test_storage_defaults(self) -> None:
        config = StorageConfig()
        assert config.golden_dataset.name.endswith(".xlsx")

    def test_logging_defaults(self) -> None:
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "text"

    def test_env_override(self, monkeypatch: object) -> None:
        """Environment variables should override defaults."""
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("PII_LOG_LEVEL", "DEBUG")
        config = LoggingConfig()
        assert config.level == "DEBUG"
        mp.undo()
