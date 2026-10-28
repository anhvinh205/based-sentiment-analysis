import pytest

from viabsa.explain_utils import merge_subword_scores


def test_sentencepiece_pieces_are_merged():
    tokens = ["<s>", "▁son", "▁đẹ", "p", "▁quá", "</s>"]
    out = merge_subword_scores(tokens, [9.0, 1.0, 2.0, 3.0, 4.0, 9.0])
    assert out == [("son", 1.0), ("đẹp", 5.0), ("quá", 4.0)]


def test_phobert_bpe_with_segmented_words():
    tokens = ["<s>", "son_môi", "đẹ@@", "p", "</s>"]
    out = merge_subword_scores(tokens, [0.0, 1.0, 2.0, 3.0, 0.0])
    assert out == [("son môi", 1.0), ("đẹp", 5.0)]


def test_wordpiece_continuations():
    tokens = ["[CLS]", "son", "đẹ", "##p", "[SEP]"]
    out = merge_subword_scores(tokens, [0.0, 1.0, 2.0, 3.0, 0.0])
    assert out == [("son", 1.0), ("đẹp", 5.0)]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        merge_subword_scores(["a", "b"], [1.0])
