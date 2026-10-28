import json

import pandas as pd

from viabsa.constants import ASPECTS
from viabsa.data.load import read_raw_split
from viabsa.data.prepare import prepare


def _write(path, rows):
    pd.DataFrame(rows, columns=["data", *ASPECTS]).to_csv(path, index=False)


def row(text, **labels):
    return [text, *[labels.get(a) for a in ASPECTS]]


def test_multiline_reviews_survive_csv_roundtrip(tmp_path):
    _write(tmp_path / "x.csv", [row("dòng một\ndòng hai", colour="positive")])
    df = read_raw_split(tmp_path / "x.csv")
    assert len(df) == 1 and "\n" in df["text"].iloc[0]
    assert df["colour"].iloc[0] == 3 and df["price"].iloc[0] == 0


def test_prepare_removes_duplicates_conflicts_empties_and_leakage(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(
        raw / "data_train.csv",
        [
            row("son đẹp", colour="positive"),
            row("son đẹp", colour="positive"),  # exact duplicate
            row("son khô", texture="negative"),
            row("son khô", texture="positive"),  # conflicting duplicate
            row("x" * 40, others="neutral"),  # empty after cleaning
            row("giao nhanh", shipping="positive"),
        ],
    )
    _write(
        raw / "data_val.csv",
        [row("Son ĐẸP!!!", colour="positive"), row("mùi thơm", smell="positive")],
    )
    _write(
        raw / "data_test.csv", [row("mùi thơm", smell="positive"), row("giá rẻ", price="positive")]
    )

    stats = prepare({"raw_dir": str(raw), "out_dir": str(tmp_path / "out")})
    train = pd.read_parquet(tmp_path / "out" / "train.parquet")
    val = pd.read_parquet(tmp_path / "out" / "val.parquet")
    test = pd.read_parquet(tmp_path / "out" / "test.parquet")

    assert sorted(train["text_clean"]) == ["giao nhanh", "son đẹp"]
    assert stats["per_split"]["train"]["conflicting_texts"] == 1
    assert stats["per_split"]["train"]["empty_after_cleaning"] == 1
    assert stats["per_split"]["train"]["exact_duplicates_removed"] == 1
    assert list(val["text_clean"]) == ["mùi thơm"]  # "Son ĐẸP!!!" leaked from train
    assert list(test["text_clean"]) == ["giá rẻ"]  # "mùi thơm" leaked from val
    assert set(train["text_clean"]).isdisjoint(val["text_clean"]) and set(
        val["text_clean"]
    ).isdisjoint(test["text_clean"])
    saved = json.loads((tmp_path / "out" / "data_stats.json").read_text())
    assert saved["cleaning"]["lowercase"] is True
