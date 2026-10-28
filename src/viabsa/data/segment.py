"""Vietnamese word segmentation (needed by PhoBERT, which expects "sản_phẩm" style input)."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from pyvi import ViTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Word segmentation needs pyvi: pip install 'viabsa[train]'") from exc
    return ViTokenizer


def segment_words(text: str) -> str:
    """Join multi-syllable words with underscores using pyvi."""
    return _tokenizer().tokenize(text)
