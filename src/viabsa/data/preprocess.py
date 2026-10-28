"""One preprocessing path shared by training and serving, so they can never drift apart."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from viabsa.data.clean import normalize_text
from viabsa.data.segment import segment_words


@dataclass
class Preprocessor:
    cleaning: dict[str, Any] = field(default_factory=dict)
    word_segment: bool = False

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> Preprocessor:
        return cls(cleaning=meta.get("cleaning", {}), word_segment=bool(meta.get("word_segment")))

    def clean(self, text: str) -> str:
        return normalize_text(text, **self.cleaning)

    def finalize(self, cleaned: str) -> str:
        """Model-specific step applied after cleaning (word segmentation for PhoBERT)."""
        return segment_words(cleaned) if self.word_segment and cleaned else cleaned

    def __call__(self, texts: Iterable[str], already_clean: bool = False) -> list[str]:
        return [self.finalize(t if already_clean else self.clean(t)) for t in texts]
