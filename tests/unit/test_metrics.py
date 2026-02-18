"""Tests for evaluation metrics."""

from image_identification.eval.metrics import EvalMetrics


class TestEvalMetrics:
    """Test the metrics recording and computation."""

    def test_top_1_correct(self) -> None:
        m = EvalMetrics()
        m.record("A", ["A", "B", "C"])
        assert m.top_1_accuracy == 1.0

    def test_top_1_incorrect(self) -> None:
        m = EvalMetrics()
        m.record("A", ["B", "A", "C"])
        assert m.top_1_accuracy == 0.0

    def test_top_3_correct(self) -> None:
        m = EvalMetrics()
        m.record("A", ["B", "C", "A", "D"])
        assert m.top_3_accuracy == 1.0

    def test_top_3_incorrect(self) -> None:
        m = EvalMetrics()
        m.record("A", ["B", "C", "D", "A"])
        assert m.top_3_accuracy == 0.0

    def test_not_found(self) -> None:
        m = EvalMetrics()
        m.record("A", ["B", "C", "D"])
        assert m.top_1_accuracy == 0.0
        assert m.top_3_accuracy == 0.0
        assert m.mrr == 0.0

    def test_mrr_calculation(self) -> None:
        m = EvalMetrics()
        m.record("A", ["A", "B", "C"])  # rank 1 → RR = 1.0
        m.record("B", ["A", "B", "C"])  # rank 2 → RR = 0.5
        assert abs(m.mrr - 0.75) < 1e-6

    def test_multiple_queries(self) -> None:
        m = EvalMetrics()
        m.record("A", ["A", "B"])  # Top-1 ✓
        m.record("B", ["A", "B"])  # Top-1 ✗, Top-3 ✓
        m.record("C", ["A", "B"])  # All ✗
        assert m.total_queries == 3
        assert m.top_1_accuracy == pytest.approx(1 / 3)
        assert m.top_3_accuracy == pytest.approx(2 / 3)

    def test_empty_metrics(self) -> None:
        m = EvalMetrics()
        assert m.top_1_accuracy == 0.0
        assert m.mrr == 0.0

    def test_summary_keys(self) -> None:
        m = EvalMetrics()
        m.record("A", ["A"])
        summary = m.summary()
        assert "total_queries" in summary
        assert "top_1_accuracy" in summary
        assert "top_3_accuracy" in summary
        assert "top_5_accuracy" in summary
        assert "mrr" in summary


# Need pytest import for approx
import pytest  # noqa: E402
