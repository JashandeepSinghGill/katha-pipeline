import math

import pytest

from katha.embed import HashingEmbedder, cosine, vector_from_text, vector_to_text


def test_embedding_is_unit_length_and_deterministic():
    e = HashingEmbedder()
    v1 = e.embed("ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦੀ ਕਥਾ")
    v2 = e.embed("ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦੀ ਕਥਾ")
    assert v1 == v2
    assert len(v1) == e.dim == 256
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)


def test_similar_contexts_score_higher_than_unrelated_ones():
    e = HashingEmbedder()
    a = e.embed("ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦਾ ਜਨਮ ਤਲਵੰਡੀ ਵਿੱਚ ਹੋਇਆ")
    b = e.embed("ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਜੀ ਦਾ ਜਨਮ ਤਲਵੰਡੀ ਵਿਖੇ ਹੋਇਆ")
    c = e.embed("ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਗਰਮ ਹੈ")
    assert cosine(a, b) > 0.6
    assert cosine(a, c) < 0.3


def test_empty_text_embeds_to_zero_vector_and_cosine_is_zero():
    e = HashingEmbedder()
    z = e.embed("   ")
    assert all(x == 0.0 for x in z)
    assert cosine(z, e.embed("ਕੁਝ")) == 0.0


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])


def test_vector_text_round_trip():
    v = [0.5, -0.25, 1e-7]
    assert vector_from_text(vector_to_text(v)) == pytest.approx(v)
    assert vector_to_text(v).startswith("[") and vector_to_text(v).endswith("]")
    assert vector_from_text("[]") == []
