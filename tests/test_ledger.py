import math

import pytest

from katha.embed import HashingEmbedder
from katha.stages.ledger import (
    LedgerEntry,
    apply_span,
    context_text,
    match,
    merge_embeddings,
    surface_score,
)

E = HashingEmbedder()


def entry(canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", forms=("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ",), ctx=None, model=E.name,
          threshold=0.8, id=1):
    return LedgerEntry(id=id, canonical=canonical, surface_forms=tuple(forms),
                       embedding=tuple(E.embed(ctx)) if ctx else None,
                       embedding_model=model if ctx else None, threshold=threshold)


def test_surface_score_uses_best_of_canonical_and_surface_forms():
    e = entry()
    assert surface_score("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", e) == 100.0
    assert surface_score("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ।", e) == 100.0   # punctuation ignored
    assert 80 < surface_score("ਭਾਈ ਸੰਤੋਕ ਸਿੰਗ", e) < 100
    assert surface_score("ਤਲਵੰਡੀ", e) < 40
    assert surface_score("   ", e) == 0.0


def test_context_text_returns_words_around_the_span_excluding_it():
    text = "ਕੱਲ ਆਪ ਜੀ ਨੇ ਸਰਵਣ ਕੀਤਾ ਸੀ ਮੰਗਲਾਚਰਨ ਚਰਨ ਨਾਮ ਗੁਰੂ ਦੇ ਚਰਨ"
    ctx = context_text(text, "ਸੀ ਮੰਗਲਾਚਰਨ", words=2)
    assert ctx == "ਸਰਵਣ ਕੀਤਾ ਚਰਨ ਨਾਮ"
    assert context_text(text, "ਨਹੀਂ ਮੌਜੂਦ") == ""
    assert context_text("ਸਿਰਫ਼ ਸਪੈਨ", "ਸਿਰਫ਼ ਸਪੈਨ") == ""
    assert context_text(text, "") == ""


CTX = "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ"


def test_match_auto_when_surface_and_context_both_pass():
    e = entry(ctx="ਕਵੀ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ")
    flagged = CTX.replace("ਸਿੰਘ", "ਸਿੰਗ")
    vec = E.embed(context_text(flagged, "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ"))
    m = match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", vec, E.name, [e], surface_threshold=88)
    assert m is not None and m.decision == "auto"
    assert m.entry.id == 1 and m.surface_score == 100.0
    assert m.context_similarity is not None and m.context_similarity >= 0.8


def test_match_suggests_when_context_is_dissimilar_or_missing():
    e = entry(ctx="ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਗਰਮ ਹੈ ਅਤੇ ਧੁੱਪ ਤੇਜ਼ ਹੈ")
    m = match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", E.embed("ਕਵੀ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ"), E.name, [e], 88)
    assert m is not None and m.decision == "suggest"
    m2 = match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", None, E.name, [entry()], 88)
    assert m2 is not None and m2.decision == "suggest" and m2.context_similarity is None


def test_match_never_autos_across_embedding_models():
    e = entry(ctx="ਕਵੀ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ", model="bge-m3")
    m = match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", E.embed("ਕਵੀ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ"), E.name, [e], 88)
    assert m is not None and m.decision == "suggest" and m.context_similarity is None


def test_match_returns_none_below_surface_threshold_or_when_span_already_canonical():
    assert match("ਤਲਵੰਡੀ", None, E.name, [entry()], 88) is None
    assert match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", None, E.name, [entry()], 88) is None
    assert match("x", None, E.name, [], 88) is None


def test_match_picks_the_best_surface_score():
    a = entry(canonical="ਸੰਤੋਖ", forms=("ਸੰਤੋਕ",), id=1)
    b = entry(canonical="ਸੰਤੋਸ਼", forms=("ਸੰਤੋਖ ਸਿੰਗ",), id=2)
    m = match("ਸੰਤੋਖ ਸਿੰਗ", None, E.name, [a, b], 88)
    assert m is not None and m.entry.id == 2


def test_apply_span_replaces_first_occurrence_only():
    assert apply_span("ਏ ਬੀ ਏ", "ਏ", "ਸੀ") == "ਸੀ ਬੀ ਏ"
    assert apply_span("ਏ ਬੀ", "ਡੀ", "ਸੀ") is None
    assert apply_span("ਏ ਬੀ", "", "ਸੀ") is None


def test_merge_embeddings_is_a_unit_mean():
    old = [1.0, 0.0]
    new = [0.0, 1.0]
    merged = merge_embeddings(old, new)
    assert merged == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert merge_embeddings(None, new) == new
    assert merge_embeddings([1.0], new) == new  # dimension mismatch -> replace
