"""Domain models for the product image identification system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Category(StrEnum):
    """Product category classification."""

    SHOE = "shoe"
    APPAREL = "apparel"
    ACCESSORY = "accessory"
    UNKNOWN = "unknown"


class Gender(StrEnum):
    """Product gender classification."""

    MEN = "Men"
    WOMEN = "Women"
    UNISEX = "Unisex"
    KIDS = "Kids"
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogItem:
    """A single product from the internal catalog.

    Maps 1:1 with a row in the golden dataset.
    """

    item_code: str
    item_name: str
    vertical: str
    family: str
    gender: Gender
    phase_in_date: datetime | None = None
    category: Category | None = None  # from golden dataset; None = use classifier

    @property
    def display_name(self) -> str:
        return f"{self.item_name} ({self.item_code})"


@dataclass(frozen=True)
class EnrichedItem:
    """A catalog item enriched with derived attributes and embedding reference."""

    item: CatalogItem
    image_url: str
    color: str
    category: Category
    embedding_ref: str  # path or key to stored embedding
    embedding_model: str
    embedding_dim: int


@dataclass
class EmbeddingResult:
    """Result of embedding generation for a single image."""

    item_code: str
    vector: NDArray[np.float32]
    model_name: str
    model_version: str
    embedding_dim: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class IdentificationMatch:
    """A single match from the similarity search."""

    item: CatalogItem
    score: float  # cosine similarity [0, 1]
    rank: int


@dataclass(frozen=True)
class IdentificationResult:
    """Full identification result for a query image."""

    query_path: str | Path
    matches: list[IdentificationMatch]
    embedding_model: str
    embedding_dim: int

    @property
    def top_match(self) -> IdentificationMatch | None:
        return self.matches[0] if self.matches else None


@dataclass
class FetchResult:
    """Result of fetching a single product image."""

    item_code: str
    url: str
    success: bool
    image_bytes: bytes | None = None
    content_type: str | None = None
    size_bytes: int = 0
    error: str | None = None
    cached: bool = False


@dataclass
class PipelineReport:
    """Summary report for an embedding-generation run."""

    total_items: int
    successful: int
    failed: int
    skipped: int
    errors: list[dict[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_items if self.total_items > 0 else 0.0
