"""Helpers for selecting valid reviews from uploaded batch CSVs."""

from __future__ import annotations

import pandas as pd


def select_review_rows(df: pd.DataFrame, column: str, limit: int) -> tuple[pd.DataFrame, int]:
    """Return up to ``limit`` rows with non-blank reviews and count skipped rows."""
    selected = df.head(limit)
    texts = selected[column].fillna("").astype(str)
    valid = texts.str.strip().ne("")
    return selected.loc[valid].copy(), int((~valid).sum())
