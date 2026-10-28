from viabsa.data.clean import has_gibberish, is_template_review, normalize_text


def test_collapses_repeated_characters():
    assert normalize_text("đẹppppp quá") == "đẹp quá"


def test_drops_keyboard_mash_tokens():
    assert normalize_text("son đẹp " + "x" * 30) == "son đẹp"
    assert has_gibberish("son " + "x" * 30)
    assert not has_gibberish("son đẹp lắm")


def test_expands_unambiguous_teencode_only():
    assert normalize_text("ko đẹp, k thơm") == "không đẹp, không thơm"
    assert normalize_text("son ok") == "son ok"


def test_unicode_is_nfc_normalised():
    decomposed = "e\u0301"  # e + combining acute
    assert normalize_text(decomposed) == "\u00e9"


def test_lowercase_switch_and_whitespace():
    assert normalize_text("  SON   Đẹp \n\r Lắm ") == "son đẹp lắm"
    assert normalize_text("Son Đẹp", lowercase=False) == "Son Đẹp"
    assert normalize_text("   ") == ""


def test_template_detection():
    assert is_template_review("Công dụng: tốt\nKết cấu: mịn")
    assert not is_template_review("son đẹp lắm")
