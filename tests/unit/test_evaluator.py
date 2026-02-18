"""Tests for the evaluation harness."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from image_identification.embeddings.metadata import EmbeddingMetadata
from image_identification.eval.evaluator import evaluate_from_files
from image_identification.eval.golden_loader import load_golden_dataset
from image_identification.storage.embedding_store import EmbeddingStore


@pytest.fixture
def golden_csv(tmp_path: Path) -> Path:
    """Create a minimal golden CSV."""
    df = pd.DataFrame(
        {
            "dim_d365_item_vertical_name": ["Performance Running", "Performance Running"],
            "dim_d365_item_retail_level_3_family": ["Cloud", "Cloud"],
            "dim_d365_item_gender_name": ["Men", "Women"],
            "dim_d365_item_item_name_en_us": ["Cloud 5 M Black", "Cloud 5 W White"],
            "dim_d365_item_item_code": ["item_a", "item_b"],
            "dim_d365_item_phase_in_date": ["2023-01-15", "2023-01-15"],
            "dim_d365_item_item_image": ["item_a", "item_b"],
        }
    )
    path = tmp_path / "golden.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def embedding_dir(tmp_path: Path) -> Path:
    """Create embeddings matching the golden CSV."""
    emb_dir = tmp_path / "embeddings"
    store = EmbeddingStore(emb_dir)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = EmbeddingMetadata(
        model_name="test",
        model_version="1.0",
        embedding_dim=4,
        total_items=2,
        backend_key="test",
    )
    store.save(embeddings, ["item_a", "item_b"], metadata)
    return emb_dir


class TestGoldenLoader:
    """Test golden dataset loading."""

    def test_load_golden(self, golden_csv: Path) -> None:
        entries = load_golden_dataset(golden_csv)
        assert len(entries) == 2
        assert entries[0].expected_code == "item_a"

    def test_load_golden_with_limit(self, golden_csv: Path) -> None:
        entries = load_golden_dataset(golden_csv, limit=1)
        assert len(entries) == 1


class TestEvaluateFromFiles:
    """Test the convenience evaluate_from_files function."""

    def test_perfect_self_identification(self, golden_csv: Path, embedding_dir: Path) -> None:
        """Self-identification should yield 100% Top-1 accuracy."""
        metrics = evaluate_from_files(
            golden_path=golden_csv,
            embedding_dir=embedding_dir,
            top_k=5,
        )
        assert metrics.top_1_accuracy == 1.0
        assert metrics.total_queries == 2

    def test_with_limit(self, golden_csv: Path, embedding_dir: Path) -> None:
        metrics = evaluate_from_files(
            golden_path=golden_csv,
            embedding_dir=embedding_dir,
            limit=1,
        )
        assert metrics.total_queries == 1
