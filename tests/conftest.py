"""Shared fixtures: a small synthetic, learnable dataset that mimics the real CSV layout."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from viabsa.constants import ASPECTS
from viabsa.training import train_baseline

PHRASES = {
    "colour": {
        "positive": ["màu rất đẹp", "lên màu chuẩn", "màu xinh quá"],
        "negative": ["màu không giống hình", "màu xấu"],
        "neutral": ["màu bình thường"],
    },
    "texture": {
        "positive": ["son mịn", "chất son mướt"],
        "negative": ["son khô môi", "chất son bị vón cục"],
        "neutral": ["chất son tạm được"],
    },
    "smell": {
        "positive": ["mùi thơm nhẹ", "thơm lắm"],
        "negative": ["mùi hơi hôi", "mùi khó chịu"],
        "neutral": ["mùi bình thường"],
    },
    "price": {
        "positive": ["giá rẻ", "giá hạt dẻ"],
        "negative": ["giá hơi đắt", "giá cao"],
        "neutral": ["giá tạm ổn"],
    },
    "shipping": {
        "positive": ["giao hàng nhanh", "ship rất nhanh"],
        "negative": ["giao hàng chậm", "ship lâu quá"],
        "neutral": ["giao hàng bình thường"],
    },
    "packing": {
        "positive": ["đóng gói cẩn thận", "đóng gói chắc chắn"],
        "negative": ["đóng gói sơ sài", "hộp bị móp"],
        "neutral": ["đóng gói tạm được"],
    },
    "stayingpower": {
        "positive": ["lâu trôi", "bám màu tốt"],
        "negative": ["trôi nhanh", "nhanh phai"],
        "neutral": ["độ bền trung bình"],
    },
    "others": {"neutral": ["không có gì đặc biệt", "shop tư vấn nhiệt tình"]},
}


def make_reviews(n: int, seed: int = 0) -> pd.DataFrame:
    """Raw-format frame: column ``data`` plus one string-or-NaN column per aspect."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        chosen = rng.sample(ASPECTS, k=rng.randint(1, 3))
        row, parts = dict.fromkeys(ASPECTS), []
        for aspect in chosen:
            label = rng.choice(list(PHRASES[aspect]))
            row[aspect] = label
            parts.append(rng.choice(PHRASES[aspect][label]))
        rng.shuffle(parts)
        rows.append({"data": ", ".join(parts), **row})
    return pd.DataFrame(rows)


def write_raw(raw_dir: Path, n_train: int = 700, n_eval: int = 120) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    make_reviews(n_train, 1).to_csv(raw_dir / "data_train.csv", index=False)
    make_reviews(n_eval, 2).to_csv(raw_dir / "data_val.csv", index=False)
    make_reviews(n_eval, 3).to_csv(raw_dir / "data_test.csv", index=False)


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Raw CSVs -> processed parquet -> trained baseline, all inside a temp directory."""
    from viabsa.data.prepare import prepare

    root = tmp_path_factory.mktemp("viabsa")
    write_raw(root / "raw")
    prepare({"raw_dir": str(root / "raw"), "out_dir": str(root / "processed")})
    train_baseline.run(
        {
            "data_dir": str(root / "processed"),
            "out_dir": str(root / "baseline"),
            "reports_dir": str(root / "reports"),
            "seed": 0,
            "tfidf": {
                "word_ngram": [1, 2],
                "word_max_features": 2000,
                "char_ngram": [2, 3],
                "char_max_features": 4000,
                "min_df": 1,
            },
            "logreg": {"c_grid": [1.0, 4.0], "class_weight": "balanced", "max_iter": 200},
        }
    )
    return root


@pytest.fixture(scope="session")
def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"
