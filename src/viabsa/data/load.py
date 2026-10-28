"""Read the raw review CSVs and convert string labels to integer ids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from viabsa.constants import ASPECTS, LABEL2ID, NONE_ID, TEXT_COLUMN


def read_raw_split(path: str | Path, text_column: str = "data") -> pd.DataFrame:
    """Read one raw CSV split.

    Reviews can contain line breaks inside quoted cells (about one in five does), so the file
    must be parsed as CSV rather than line by line. Empty aspect cells mean "not mentioned".
    """
    df = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""])
    missing = [c for c in [text_column, *ASPECTS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.rename(columns={text_column: TEXT_COLUMN})
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    for aspect in ASPECTS:
        df[aspect] = _encode_labels(df[aspect], aspect, path)
    return df[[TEXT_COLUMN, *ASPECTS]]


def _encode_labels(col: pd.Series, aspect: str, path: str | Path) -> pd.Series:
    cleaned = col.astype("string").str.strip().str.lower()
    unknown = set(cleaned.dropna().unique()) - set(LABEL2ID)
    if unknown:
        raise ValueError(f"{path}: unknown labels {sorted(unknown)} in column {aspect!r}")
    return cleaned.map(LABEL2ID).fillna(NONE_ID).astype("int64")


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return an ``(n_reviews, n_aspects)`` int array of label ids."""
    return df[ASPECTS].to_numpy(dtype="int64")
