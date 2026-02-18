"""Integration test for the end-to-end pipeline with fixtures.

Uses mocked HTTP responses and a fake embedding backend to test
the full pipeline flow without real model loading or network calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fixture_catalog(tmp_path: Path) -> Path:
    """Create a minimal fixture catalog CSV."""
    df = pd.DataFrame(
        {
            "dim_d365_item_vertical_name": [
                "Performance Running",
                "Performance Running",
                "Performance All Day",
            ],
            "dim_d365_item_retail_level_3_family": ["Cloud", "Cloud", "All Day"],
            "dim_d365_item_gender_name": ["Men", "Women", "Unisex"],
            "dim_d365_item_item_name_en_us": [
                "Cloud 5 M Black",
                "Cloud 5 W All Black",
                "Merino Beanie 1 U Lunar",
            ],
            "dim_d365_item_item_code": ["59.98842", "59.98843", "311.00218"],
            "dim_d365_item_phase_in_date": ["2023-01-15", "2023-01-15", "2019-09-12"],
            "dim_d365_item_item_image": ["59.98842", "59.98843", "311.00218"],
        }
    )
    path = tmp_path / "fixture_catalog.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.integration
class TestPipelineIntegration:
    """End-to-end pipeline test with mocked dependencies."""

    def test_catalog_to_enriched(self, fixture_catalog: Path, tmp_path: Path) -> None:
        """Test loading catalog → extracting features → saving enriched CSV."""
        from image_identification.catalog.loader import load_catalog
        from image_identification.domain.models import EnrichedItem
        from image_identification.features.category_classifier import classify_category
        from image_identification.features.color_extractor import extract_color
        from image_identification.images.url_builder import UrlBuilder
        from image_identification.storage.csv_store import save_enriched_csv

        # Load
        items = load_catalog(fixture_catalog)
        assert len(items) == 3

        # Process
        url_builder = UrlBuilder()
        enriched: list[EnrichedItem] = []

        for item in items:
            url = url_builder.build(item.item_code)
            color = extract_color(item.item_name)
            category = classify_category(item.item_name, item.family, item.vertical)

            enriched.append(
                EnrichedItem(
                    item=item,
                    image_url=url,
                    color=color,
                    category=category,
                    embedding_ref=f"{item.item_code}.npy",
                    embedding_model="test-model",
                    embedding_dim=128,
                )
            )

        # Save
        output_path = tmp_path / "enriched.csv"
        save_enriched_csv(enriched, output_path)

        # Verify
        df = pd.read_csv(output_path)
        assert len(df) == 3
        assert "color" in df.columns
        assert "category" in df.columns
        assert df.iloc[0]["color"] == "Black"
        assert df.iloc[0]["category"] == "shoe"
        assert df.iloc[2]["category"] == "accessory"

    def test_similarity_search_end_to_end(self, tmp_path: Path) -> None:
        """Test building index → searching → getting correct results."""
        from image_identification.embeddings.metadata import EmbeddingMetadata
        from image_identification.index.in_memory import InMemoryIndex
        from image_identification.storage.embedding_store import EmbeddingStore

        # Create fake embeddings
        embeddings = np.random.randn(5, 128).astype(np.float32)
        codes = ["item_a", "item_b", "item_c", "item_d", "item_e"]

        # Save
        store = EmbeddingStore(tmp_path / "embeddings")
        metadata = EmbeddingMetadata(
            model_name="test",
            model_version="0.1",
            embedding_dim=128,
            total_items=5,
            backend_key="test",
        )
        store.save(embeddings, codes, metadata)

        # Load and search
        loaded_embeddings = store.load_embeddings()
        loaded_codes = store.load_item_codes()

        index = InMemoryIndex()
        index.build(loaded_embeddings, loaded_codes)

        # Query with first item's embedding — should match itself
        matches = index.search(embeddings[0], top_k=3)
        assert matches[0].item_code == "item_a"
        assert pytest.approx(matches[0].score, abs=1e-4) == 1.0
