"""Compare accuracy and latency of several saved models on the test split.

Usage::

    python -m viabsa.benchmark --models models/baseline models/phobert models/phobert_onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from viabsa.data.load import label_matrix
from viabsa.evaluation.metrics import compute_metrics
from viabsa.inference import load_predictor


def latency_ms(predict, texts: list[str], batch_size: int, repeats: int = 3) -> dict[str, float]:
    """Per-review latency (ms) over several passes; the first pass is a warm-up."""
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    predict(batches[0])
    per_review = []
    for _ in range(repeats):
        for batch in batches:
            start = time.perf_counter()
            predict(batch)
            per_review.append((time.perf_counter() - start) * 1000 / len(batch))
    return {
        "p50": float(np.percentile(per_review, 50)),
        "p95": float(np.percentile(per_review, 95)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", default="data/processed/test.parquet")
    parser.add_argument("--n", type=int, default=300, help="reviews used for timing")
    parser.add_argument("--out", default="reports/benchmark.md")
    args = parser.parse_args(argv)

    test = pd.read_parquet(args.data)
    texts_raw = test["text"].tolist()
    y = label_matrix(test)
    rows = []
    for path in args.models:
        pred = load_predictor(path)
        proba = pred.predict_proba(texts_raw)
        f1 = compute_metrics(y, proba.argmax(-1), proba)["mean_macro_f1"]
        sample = texts_raw[: args.n]
        single = latency_ms(pred.predict_proba, sample[:100], 1, repeats=1)
        batched = latency_ms(pred.predict_proba, sample, 32)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6
        rows.append(
            f"| {Path(path).name} | {pred.kind} | {f1:.3f} | {size:.0f} | "
            f"{single['p50']:.1f} | {batched['p50']:.1f} |"
        )
    table = "\n".join(
        [
            "| Model | Type | Test mean macro-F1 | Size (MB) "
            "| ms/review (batch 1) | ms/review (batch 32) |",
            "|---|---|---:|---:|---:|---:|",
            *rows,
        ]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
