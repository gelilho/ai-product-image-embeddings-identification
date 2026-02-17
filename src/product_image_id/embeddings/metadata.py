"""Embedding metadata — tracks model version, dimensions, and timestamps.

Stored alongside embedding artifacts to enable reproducibility and
detect embedding drift when models change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Metadata for a batch of generated embeddings."""

    model_name: str
    model_version: str
    embedding_dim: int
    total_items: int
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    backend_key: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> EmbeddingMetadata:
        data = json.loads(raw)
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save metadata to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> EmbeddingMetadata:
        """Load metadata from a JSON file."""
        return cls.from_json(path.read_text())
