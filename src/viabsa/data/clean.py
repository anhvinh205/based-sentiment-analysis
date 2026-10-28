"""Text normalisation for noisy Vietnamese e-commerce reviews."""

from __future__ import annotations

import re
import unicodedata

# Deliberately small and conservative: only unambiguous abbreviations.
TEENCODE: dict[str, str] = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "không",
    "dc": "được",
    "đc": "được",
    "vs": "với",
    "mn": "mọi người",
    "sp": "sản phẩm",
    "z": "vậy",
    "j": "gì",
    "r": "rồi",
    "mk": "mình",
    "mik": "mình",
    "nv": "vậy",
    "lun": "luôn",
    "bik": "biết",
    "wa": "quá",
    "oke": "ok",
}

TEMPLATE_PATTERN = re.compile(r"(công dụng|kết cấu|độ bền màu)\s*:", re.IGNORECASE)
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACES = re.compile(r"\s+")


def is_template_review(text: str) -> bool:
    """Marketplace auto-form reviews ("Công dụng: ... Kết cấu: ... Độ bền màu: ...")."""
    return bool(TEMPLATE_PATTERN.search(text))


def has_gibberish(text: str, max_token_len: int = 25) -> bool:
    """True when a whitespace-delimited token is too long to be a Vietnamese word."""
    return any(len(tok) > max_token_len for tok in text.split())


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    max_token_len: int = 25,
    collapse_repeats: bool = True,
    expand_teencode: bool = True,
) -> str:
    """Normalise one review: Unicode NFC, drop keyboard mash, squeeze repeats, expand teencode."""
    text = unicodedata.normalize("NFC", str(text))
    if lowercase:
        text = text.lower()
    tokens = [tok for tok in _SPACES.split(text.strip()) if tok and len(tok) <= max_token_len]
    text = " ".join(tokens)
    if collapse_repeats:
        text = _REPEAT.sub(r"\1", text)
    if expand_teencode:
        text = " ".join(_expand_token(tok) for tok in text.split())
    return text


def _expand_token(token: str) -> str:
    stripped = token.strip(".,!?;:()\"'")
    if stripped in TEENCODE:
        return token.replace(stripped, TEENCODE[stripped], 1)
    return token
