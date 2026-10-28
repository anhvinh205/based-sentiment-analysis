"""Torch-free helpers to turn sub-word attributions into word-level scores for display."""

from __future__ import annotations

SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", "<mask>", "[CLS]", "[SEP]", "[PAD]", "[UNK]"}


def _scheme(tokens: list[str]) -> str:
    if any(t.startswith("\u2581") for t in tokens):
        return "sentencepiece"  # XLM-R, ViSoBERT
    if any(t.endswith("@@") for t in tokens):
        return "bpe"  # PhoBERT (fastBPE)
    return "wordpiece"


def merge_subword_scores(tokens: list[str], scores: list[float]) -> list[tuple[str, float]]:
    """Merge sub-word pieces into words; a word's score is the sum of its pieces' scores.

    Special tokens are dropped and underscores of word-segmented Vietnamese become spaces.
    """
    scheme = _scheme(tokens)
    words: list[tuple[str, float]] = []
    current, score, joins_next = "", 0.0, False

    def flush() -> None:
        nonlocal current, score
        if current:
            words.append((current.replace("_", " "), score))
        current, score = "", 0.0

    for tok, s in zip(tokens, scores, strict=True):
        if tok in SPECIAL_TOKENS:
            flush()
            joins_next = False
            continue
        piece, starts_new = tok, False
        if scheme == "sentencepiece":
            starts_new = tok.startswith("\u2581")
            piece = tok.lstrip("\u2581")
        elif scheme == "bpe":
            starts_new = not joins_next
            joins_next = tok.endswith("@@")
            piece = tok[:-2] if joins_next else tok
        else:
            starts_new = not tok.startswith("##")
            piece = tok[2:] if tok.startswith("##") else tok
        if starts_new:
            flush()
        current += piece
        score += float(s)
    flush()
    return words
