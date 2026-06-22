"""Evaluation helpers matching the official baseline's metric definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    pauc: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def binary_metrics(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    threshold: float,
    max_fpr: float = 0.1,
) -> BinaryMetrics:
    """Calculate AUC, standardized partial AUC, and thresholded F1."""
    try:
        from sklearn import metrics
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Evaluation needs scikit-learn; install aegis/requirements.txt"
        ) from exc

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size != 2:
        raise ValueError("AUC requires at least one normal and one anomalous sample")

    prediction = (y_score > threshold).astype(np.int64)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true,
        prediction,
        average="binary",
        zero_division=0,
    )
    return BinaryMetrics(
        auc=float(metrics.roc_auc_score(y_true, y_score)),
        pauc=float(metrics.roc_auc_score(y_true, y_score, max_fpr=max_fpr)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )

