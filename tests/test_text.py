from katha.text import normalize, romanize


def test_normalize_strips_dandas_punctuation_and_collapses_whitespace():
    assert normalize("ਤੋ ਸੋ  ਨਹੀ ਦਾਤਾ ਕੋਇ ॥") == "ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ"
    assert normalize("  ਭਾਈ, ਸੰਤੋਖ ਸਿੰਘ।") == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"


def test_normalize_casefolds_latin_leakage():
    assert normalize("Guru ਗੱਦੀ") == "guru ਗੱਦੀ"


def test_romanize_produces_latin():
    out = romanize("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ")
    assert out.startswith("bh")
    assert "ਭ" not in out


def test_romanize_never_raises_on_odd_input():
    assert romanize("") == ""
    assert isinstance(romanize("Guru ਗੱਦੀ ॥"), str)
