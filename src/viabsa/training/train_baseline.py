"""Train the TF-IDF + logistic-regression baseline.

Usage::

    python -m viabsa.training.train_baseline --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from viabsa.config import load_config
from viabsa.data.load import label_matrix
from viabsa.evaluation.metrics import metrics_table
from viabsa.evaluation.report import write_report
from viabsa.models.baseline import BaselineModel

logger = logging.getLogger(__name__)
TEXT = "text_clean"


def run(cfg: dict) -> dict:
    np.random.seed(cfg.get("seed", 42))
    data_dir = Path(cfg["data_dir"])
    train, val, test = (
        pd.read_parquet(data_dir / f"{s}.parquet") for s in ("train", "val", "test")
    )

    tf, lr = cfg["tfidf"], cfg["logreg"]
    model = BaselineModel(
        word_ngram=tuple(tf["word_ngram"]),
        word_max_features=tf["word_max_features"],
        char_ngram=tuple(tf["char_ngram"]),
        char_max_features=tf["char_max_features"],
        min_df=tf["min_df"],
        c_values=lr["c_grid"],
        class_weight=lr["class_weight"],
        max_iter=lr["max_iter"],
        seed=cfg.get("seed", 42),
    )
    start = time.time()
    model.fit(train[TEXT].tolist(), label_matrix(train), val[TEXT].tolist(), label_matrix(val))
    train_seconds = time.time() - start
    logger.info("trained in %.1fs, best C per aspect: %s", train_seconds, model.best_c)

    out_dir = Path(cfg["out_dir"])
    model.save(out_dir)
    stats = json.loads((data_dir / "data_stats.json").read_text())
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "type": "baseline",
                "best_c": model.best_c,
                "cleaning": stats["cleaning"],
                "word_segment": False,
            },
            indent=2,
        )
    )

    proba = model.predict_proba(test[TEXT].tolist())
    metrics = write_report(
        "baseline", test, proba, cfg["reports_dir"], {"train_seconds": round(train_seconds, 1)}
    )
    logger.info("test mean macro-F1 = %.4f", metrics["mean_macro_f1"])
    print(metrics_table(metrics))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
