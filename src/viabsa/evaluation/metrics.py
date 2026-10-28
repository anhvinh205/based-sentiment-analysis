"""Metrics for multi-aspect, 4-class sentiment (torch-free, numpy + scikit-learn only).

Definitions
-----------
* ``macro_f1``      4-class macro-F1 per aspect, averaged over classes present in the gold labels.
* ``detection_f1``  binary F1 of "aspect is mentioned" vs "not mentioned".
* ``sentiment_f1``  macro-F1 over {negative, neutral, positive}, only on reviews that gold-mention
                    the aspect (isolates polarity errors from detection errors).
* ``mean_macro_f1`` headline number: mean of ``macro_f1`` over aspects.
* ``ece``           expected calibration error of the predicted class probability.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from viabsa.constants import ASPECTS, ID2LABEL, N_LABELS, NONE_ID


def _macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    """Macro-F1 over the classes that occur in ``y_true`` (absent classes would add fake zeros)."""
    present = [c for c in labels if (y_true == c).any()]
    if not present:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(confidence)
    for lo, hi in pairwise(edges):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.sum() / n * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None
) -> dict[str, Any]:
    """Compute per-aspect and aggregate metrics.

    Args:
        y_true: ``(n, n_aspects)`` gold label ids.
        y_pred: ``(n, n_aspects)`` predicted label ids.
        proba: optional ``(n, n_aspects, n_labels)`` probabilities, enables calibration (ECE).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    all_labels = list(range(N_LABELS))
    polarity = [c for c in all_labels if c != NONE_ID]
    per_aspect: dict[str, dict[str, Any]] = {}

    for j, aspect in enumerate(ASPECTS):
        t, p = y_true[:, j], y_pred[:, j]
        mentioned_t, mentioned_p = t != NONE_ID, p != NONE_ID
        m = mentioned_t
        entry: dict[str, Any] = {
            "macro_f1": _macro_f1_present(t, p, all_labels),
            "accuracy": float(accuracy_score(t, p)),
            "detection_f1": float(f1_score(mentioned_t, mentioned_p, zero_division=0)),
            "sentiment_f1": _macro_f1_present(t[m], p[m], polarity) if m.any() else float("nan"),
            "support": {ID2LABEL[c]: int((t == c).sum()) for c in all_labels},
            "confusion_matrix": confusion_matrix(t, p, labels=all_labels).tolist(),
        }
        if proba is not None:
            conf = proba[:, j, :].max(axis=1)
            entry["ece"] = expected_calibration_error(conf, (t == p).astype(float))
        per_aspect[aspect] = entry

    def mean_of(key: str) -> float:
        vals = [v[key] for v in per_aspect.values() if not np.isnan(v[key])]
        return float(np.mean(vals)) if vals else float("nan")

    result: dict[str, Any] = {
        "mean_macro_f1": mean_of("macro_f1"),
        "mean_detection_f1": mean_of("detection_f1"),
        "mean_sentiment_f1": mean_of("sentiment_f1"),
        "mean_accuracy": mean_of("accuracy"),
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
        "per_aspect": per_aspect,
    }
    if proba is not None:
        result["mean_ece"] = mean_of("ece")
    return result


def metrics_table(metrics: dict[str, Any]) -> str:
    """Render per-aspect metrics as a GitHub-flavoured markdown table."""
    lines = [
        "| Aspect | Macro-F1 | Detection F1 | Sentiment F1 | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for aspect, m in metrics["per_aspect"].items():
        lines.append(
            f"| {aspect} | {m['macro_f1']:.3f} | {m['detection_f1']:.3f} | "
            f"{m['sentiment_f1']:.3f} | {m['accuracy']:.3f} |"
        )
    lines.append(
        f"| **mean** | **{metrics['mean_macro_f1']:.3f}** | {metrics['mean_detection_f1']:.3f} | "
        f"{metrics['mean_sentiment_f1']:.3f} | {metrics['mean_accuracy']:.3f} |"
    )
    return "\n".join(lines)
