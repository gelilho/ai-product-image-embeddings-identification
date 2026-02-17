"""Tests for storage adapters."""

from pathlib import Path

import numpy as np
import pytest

from product_image_id.domain.models import CatalogItem, Category, EnrichedItem, Gender
from product_image_id.embeddings.metadata import EmbeddingMetadata
from product_image_id.storage.csv_store import load_enriched_csv, save_enriched_csv
from product_image_id.storage.embedding_store import EmbeddingStore


class TestCsvStore:
    """Test enriched CSV storage."""

    @pytest.fixture
    def enriched_items(self) -> list[EnrichedItem]:
        item = CatalogItem(
            item_code="59.98842",
            item_name="Cloud 5 M Black",
            vertical="Performance Running",
            family="Cloud",
            gender=Gender.MEN,
        )
        return [
            EnrichedItem(
                item=item,
                image_url="https://example.com/59.98842_000_001.png",
                color="Black",
                category=Category.SHOE,
                embedding_ref="59.98842.npy",
                embedding_model="test-model",
                embedding_dim=1024,
            )
        ]

    def test_save_and_load(self, enriched_items: list[EnrichedItem], tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        save_enriched_csv(enriched_items, path)

        df = load_enriched_csv(path)
        assert len(df) == 1
        assert str(df.iloc[0]["item_code"]) == "59.98842"
        assert df.iloc[0]["color"] == "Black"
        assert df.iloc[0]["category"] == "shoe"

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_enriched_csv(tmp_path / "nonexistent.csv")


class TestEmbeddingStore:
    """Test embedding storage."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        store = EmbeddingStore(tmp_path / "emb")
        embeddings = np.random.randn(3, 128).astype(np.float32)
        codes = ["a", "b", "c"]
        meta = EmbeddingMetadata(
            model_name="test",
            model_version="1.0",
            embedding_dim=128,
            total_items=3,
            backend_key="test",
        )

        store.save(embeddings, codes, meta)

        assert store.exists()
        loaded = store.load_embeddings()
        loaded_codes = store.load_item_codes()
        loaded_meta = store.load_metadata()

        np.testing.assert_array_almost_equal(loaded, embeddings)
        assert loaded_codes == codes
        assert loaded_meta.model_name == "test"
        assert loaded_meta.total_items == 3

    def test_exists_false_when_empty(self, tmp_path: Path) -> None:
        store = EmbeddingStore(tmp_path / "empty")
        assert not store.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        store = EmbeddingStore(tmp_path / "missing")
        with pytest.raises(FileNotFoundError):
            store.load_embeddings()
