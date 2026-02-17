"""Tests for logging setup."""

from product_image_id.config import LoggingConfig
from product_image_id.logging import logger, setup_logging


class TestLoggingSetup:
    """Test logging configuration."""

    def test_setup_text_format(self) -> None:
        config = LoggingConfig(level="DEBUG", format="text")
        setup_logging(config)
        # Should not raise
        logger.info("test text logging")

    def test_setup_json_format(self) -> None:
        config = LoggingConfig(level="INFO", format="json")
        setup_logging(config)
        logger.info("test json logging")

    def test_setup_default(self) -> None:
        setup_logging()
        logger.info("test default logging")

    def test_logger_is_importable(self) -> None:
        assert logger is not None
