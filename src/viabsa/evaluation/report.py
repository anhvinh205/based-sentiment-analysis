"""Turn predictions into artefacts: metrics.json, slice metrics, confusion matrices, error CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viabsa.constants import ASPECTS, ID2LABEL, LABELS
from viabsa.data.load import label_matrix
from viabsa.evaluation.metrics import compute_metrics

SLICES = {
    "template_reviews": "is_template",
    "free_text_reviews": "~is_template",
    "had_gibberish": "had_gibberish",
}


def slice_metrics(df: pd.DataFrame, proba: np.ndarray) -> dict[str, Any]:
    """Headline metrics on data slices (template reviews are probably easier than free text)."""
    y_true, y_pred = label_matrix(df), proba.argmax(-1)
    out: dict[str, Any] = {}
    for name, expr in SLICES.items():
        negated = expr.startswith("~")
        column = expr[1:] if negated else expr
        mask = df[column].to_numpy()
        if negated:
            mask = ~mask
        if mask.sum() < 20:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        out[name] = {"n": int(mask.sum()), "mean_macro_f1": m["mean_macro_f1"]}
    return out


def plot_confusions(metrics: dict[str, Any], path: str | Path, title: str = "") -> None:
    """One normalised confusion matrix per aspect in a 2x4 grid."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, aspect in zip(axes.ravel(), ASPECTS, strict=True):
        cm = np.array(metrics["per_aspect"][aspect]["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(
                j,
                i,
                int(v),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] > 0.5 else "black",
            )
        ax.set_xticks(range(len(LABELS)), [s[:3] for s in LABELS])
        ax.set_yticks(range(len(LABELS)), [s[:3] for s in LABELS])
        ax.set_title(f"{aspect}  (F1 {metrics['per_aspect'][aspect]['macro_f1']:.2f})", fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
    fig.suptitle(title or "Confusion matrices (cell = count, colour = row-normalised)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_table(df: pd.DataFrame, proba: np.ndarray, top_k: int = 300) -> pd.DataFrame:
    """Most confidently wrong (review, aspect) pairs, for manual error analysis."""
    y_true, y_pred = label_matrix(df), proba.argmax(-1)
    rows = []
    for j, aspect in enumerate(ASPECTS):
        wrong = np.where(y_true[:, j] != y_pred[:, j])[0]
        for i in wrong:
            rows.append(
                {
                    "text": df["text"].iloc[i],
                    "aspect": aspect,
                    "gold": ID2LABEL[int(y_true[i, j])],
                    "pred": ID2LABEL[int(y_pred[i, j])],
                    "confidence": float(proba[i, j].max()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("confidence", ascending=False).head(top_k).reset_index(drop=True)


def write_report(
    name: str,
    df: pd.DataFrame,
    proba: np.ndarray,
    reports_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute metrics for a model on ``df`` and write all report artefacts."""
    reports = Path(reports_dir)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(label_matrix(df), proba.argmax(-1), proba)
    metrics["slices"] = slice_metrics(df, proba)
    metrics["model"] = name
    metrics["n_test"] = len(df)
    if extra:
        metrics.update(extra)
    (reports / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusions(metrics, reports / "figures" / f"confusion_{name}.png", f"{name}: test set")
    error_table(df, proba).to_csv(reports / f"errors_{name}.csv", index=False)
    return metrics
