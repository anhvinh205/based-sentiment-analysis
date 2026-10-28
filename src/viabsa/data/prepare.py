"""Raw CSVs -> clean, de-duplicated, leakage-free parquet splits + a statistics report.

Usage::

    python -m viabsa.data.prepare --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from viabsa.config import load_config
from viabsa.constants import ASPECTS, ID2LABEL, TEXT_COLUMN
from viabsa.data.clean import has_gibberish, is_template_review, normalize_text
from viabsa.data.load import read_raw_split

logger = logging.getLogger(__name__)

SPLIT_FILES = {"train": "data_train.csv", "val": "data_val.csv", "test": "data_test.csv"}
CLEAN_COL = "text_clean"
KEY_COL = "dedupe_key"  # punctuation-insensitive key: "Son đẹp!!!" and "son đẹp" are one review
_NON_WORD = re.compile(r"[^\w\s]")
DEFAULT_CLEANING = {
    "lowercase": True,
    "max_token_len": 25,
    "collapse_repeats": True,
    "expand_teencode": True,
}


def add_clean_columns(df: pd.DataFrame, cleaning: dict[str, Any]) -> pd.DataFrame:
    """Add ``text_clean`` plus the ``is_template`` / ``had_gibberish`` slice flags."""
    out = df.copy()
    max_len = int(cleaning.get("max_token_len", 25))
    out["is_template"] = out[TEXT_COLUMN].map(is_template_review)
    out["had_gibberish"] = out[TEXT_COLUMN].map(lambda t: has_gibberish(t, max_len))
    out[CLEAN_COL] = out[TEXT_COLUMN].map(
        lambda t: normalize_text(
            t,
            lowercase=cleaning.get("lowercase", True),
            max_token_len=max_len,
            collapse_repeats=cleaning.get("collapse_repeats", True),
            expand_teencode=cleaning.get("expand_teencode", True),
        )
    )
    out[KEY_COL] = out[CLEAN_COL].map(_dedupe_key)
    return out


def _dedupe_key(text: str) -> str:
    key = " ".join(_NON_WORD.sub(" ", text).split())
    return key or text  # emoji-only / symbol-only reviews keep their own text as key


def dedupe_split(df: pd.DataFrame, drop_conflicting: bool = True) -> tuple[pd.DataFrame, dict]:
    """Remove empty reviews and exact duplicates; optionally drop copies with conflicting labels."""
    stats = {"input": len(df)}
    df = df[df[CLEAN_COL].str.len() > 0]
    stats["empty_after_cleaning"] = stats["input"] - len(df)

    n_unique_labels = df.groupby(KEY_COL)[ASPECTS].nunique().max(axis=1)
    conflicting = set(n_unique_labels[n_unique_labels > 1].index)
    stats["conflicting_texts"] = len(conflicting)
    if drop_conflicting and conflicting:
        df = df[~df[KEY_COL].isin(conflicting)]

    before = len(df)
    df = df.drop_duplicates(subset=KEY_COL, keep="first")
    stats["exact_duplicates_removed"] = before - len(df)
    return df.reset_index(drop=True), stats


def remove_leakage(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Drop val/test reviews whose cleaned text also appears in an earlier split."""
    seen = set(splits["train"][KEY_COL])
    removed: dict[str, int] = {}
    out = {"train": splits["train"]}
    for name in ("val", "test"):
        df = splits[name]
        keep = ~df[KEY_COL].isin(seen)
        removed[name] = int((~keep).sum())
        out[name] = df[keep].reset_index(drop=True)
        seen |= set(out[name][KEY_COL])
    return out, removed


def label_distribution(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for aspect in ASPECTS:
        values, counts = np.unique(df[aspect].to_numpy(), return_counts=True)
        out[aspect] = {ID2LABEL[int(v)]: int(c) for v, c in zip(values, counts, strict=True)}
    return out


def prepare(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_dir, out_dir = Path(cfg["raw_dir"]), Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaning = {**DEFAULT_CLEANING, **cfg.get("cleaning", {})}
    dedupe_cfg = cfg.get("dedupe", {})

    # The effective cleaning settings are stored so training and serving reuse them exactly.
    splits, stats = {}, {"cleaning": cleaning, "per_split": {}}
    for name, filename in SPLIT_FILES.items():
        df = read_raw_split(raw_dir / filename, cfg.get("text_column", "data"))
        df = add_clean_columns(df, cleaning)
        df, split_stats = dedupe_split(df, dedupe_cfg.get("drop_conflicting", True))
        splits[name] = df
        stats["per_split"][name] = split_stats

    if dedupe_cfg.get("remove_cross_split_leakage", True):
        splits, removed = remove_leakage(splits)
        stats["cross_split_leakage_removed"] = removed

    for name, df in splits.items():
        df.drop(columns=[KEY_COL]).to_parquet(out_dir / f"{name}.parquet", index=False)
        stats["per_split"][name]["final"] = len(df)
        stats["per_split"][name]["label_distribution"] = label_distribution(df)
        stats["per_split"][name]["template_share"] = round(float(df["is_template"].mean()), 4)
        stats["per_split"][name]["gibberish_share"] = round(float(df["had_gibberish"].mean()), 4)

    payload = json.dumps(stats, indent=2, ensure_ascii=False)
    (out_dir / "data_stats.json").write_text(payload)
    if cfg.get("reports_dir"):  # statistics only (no review text), safe to commit
        Path(cfg["reports_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(cfg["reports_dir"]) / "data_stats.json").write_text(payload)
    logger.info("Wrote %s", out_dir)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stats = prepare(load_config(args.config, args.overrides))
    for name, s in stats["per_split"].items():
        logger.info("%s: %s -> %s rows", name, s["input"], s["final"])
    logger.info("leakage removed: %s", stats.get("cross_split_leakage_removed"))


if __name__ == "__main__":
    main()
