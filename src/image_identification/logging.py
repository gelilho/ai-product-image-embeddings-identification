"""Logging setup using loguru — consistent across all modules.

Usage:
    from image_identification.logging import logger

    logger.info("Processing item", item_code="311.00218")
"""

from __future__ import annotations

import sys

from loguru import logger as _loguru_logger

from image_identification.config import LoggingConfig

# Re-export so every module imports from here
logger = _loguru_logger


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure loguru based on application config.

    Call once at startup (e.g. in CLI main).
    """
    cfg = config or LoggingConfig()

    # Remove default handler
    logger.remove()

    if cfg.format == "json":
        logger.add(
            sys.stderr,
            level=cfg.level.upper(),
            serialize=True,  # JSON output
        )
    else:
        logger.add(
            sys.stderr,
            level=cfg.level.upper(),
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )
