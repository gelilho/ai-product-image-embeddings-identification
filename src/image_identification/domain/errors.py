"""Domain errors for the product image identification system.

All custom exceptions inherit from PIIError so callers can catch broadly
or narrowly as needed.
"""

from __future__ import annotations


class PIIError(Exception):
    """Base error for the product image identification system."""


# ---------------------------------------------------------------------------
# Catalog errors
# ---------------------------------------------------------------------------


class CatalogError(PIIError):
    """Error loading or validating catalog data."""


class SchemaValidationError(CatalogError):
    """The input data does not match the expected schema."""

    def __init__(self, missing_columns: list[str] | None = None, message: str = "") -> None:
        self.missing_columns = missing_columns or []
        detail = message or f"Missing required columns: {', '.join(self.missing_columns)}"
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Image fetch errors
# ---------------------------------------------------------------------------


class FetchError(PIIError):
    """Error fetching a product image."""

    def __init__(self, item_code: str, url: str, reason: str) -> None:
        self.item_code = item_code
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to fetch image for {item_code} from {url}: {reason}")


class ImageTooLargeError(FetchError):
    """Downloaded image exceeds the configured maximum size."""

    def __init__(self, item_code: str, url: str, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            item_code,
            url,
            f"Image size {size_bytes} bytes exceeds max {max_bytes} bytes",
        )


class ImageValidationError(FetchError):
    """The fetched bytes are not a valid image."""


class RateLimitExceededError(FetchError):
    """Remote host returned a rate-limit response (429)."""


# ---------------------------------------------------------------------------
# Embedding errors
# ---------------------------------------------------------------------------


class EmbeddingError(PIIError):
    """Error generating embeddings."""

    def __init__(self, item_code: str, reason: str) -> None:
        self.item_code = item_code
        self.reason = reason
        super().__init__(f"Embedding failed for {item_code}: {reason}")


class BackendNotFoundError(EmbeddingError):
    """The requested embedding backend is not registered."""

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name
        super().__init__(item_code="N/A", reason=f"Backend '{backend_name}' not found")


# ---------------------------------------------------------------------------
# Storage errors
# ---------------------------------------------------------------------------


class StorageError(PIIError):
    """Error reading or writing storage artifacts."""


# ---------------------------------------------------------------------------
# Service-layer errors
# ---------------------------------------------------------------------------


class EmbeddingsNotFoundError(PIIError):
    """No embeddings found on disk — user must run embedding-generation first."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"No embeddings found at {path}. Run 'embedding-generation' first.")


class NoCategoryItemsError(PIIError):
    """The detected category has zero items in the index."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(f"No items of category '{category}' found in the index.")
