"""Evaluation metrics for the identification system.

Supports Top-K accuracy, MRR, and per-item result tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics across a golden dataset."""

    total_queries: int = 0
    top_1_correct: int = 0
    top_3_correct: int = 0
    top_5_correct: int = 0
    reciprocal_ranks: list[float] = field(default_factory=list)

    @property
    def top_1_accuracy(self) -> float:
        return self.top_1_correct / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def top_3_accuracy(self) -> float:
        return self.top_3_correct / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def top_5_accuracy(self) -> float:
        return self.top_5_correct / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank."""
        if not self.reciprocal_ranks:
            return 0.0
        return sum(self.reciprocal_ranks) / len(self.reciprocal_ranks)

    def record(self, expected_code: str, result_codes: list[str]) -> None:
        """Record a single identification result against the expected code.

        Parameters
        ----------
        expected_code:
            The correct item code (ground truth).
        result_codes:
            Ordered list of returned item codes (best match first).
        """
        self.total_queries += 1

        if expected_code in result_codes[:1]:
            self.top_1_correct += 1
        if expected_code in result_codes[:3]:
            self.top_3_correct += 1
        if expected_code in result_codes[:5]:
            self.top_5_correct += 1

        # Reciprocal rank
        try:
            rank = result_codes.index(expected_code) + 1
            self.reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            self.reciprocal_ranks.append(0.0)

    def summary(self) -> dict[str, float]:
        """Return a summary dictionary."""
        return {
            "total_queries": float(self.total_queries),
            "top_1_accuracy": self.top_1_accuracy,
            "top_3_accuracy": self.top_3_accuracy,
            "top_5_accuracy": self.top_5_accuracy,
            "mrr": self.mrr,
        }
