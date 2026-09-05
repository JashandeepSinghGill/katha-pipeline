from pathlib import Path

import pytest

from katha import repo, review
from katha.embed import HashingEmbedder
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

ENTRIES = [DiarizedEntry("0", "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ", 3.5, 5.5)]
WORDS = [AlignedWord(w, 3.5 + i * 0.2, 3.6 + i * 0.2, 0.9)
         for i, w in enumerate(ENTRIES[0].text.split())]
E = HashingEmbedder()


@pytest.fixture
def flag(db):
    cid = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    repo.save_segments(db, cid, assemble.assemble_segments(ENTRIES, WORDS), reciter_id=None)
    sid = repo.list_segment_texts(db, cid)[0]["id"]
    repo.save_flags(db, cid, [(sid, "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "proper_noun", "r", ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"])])
    return repo.list_open_flags(db, cid)[0]


@pytest.mark.db
def test_accept_rewrites_segment_logs_correction_and_teaches_ledger(db, flag):
    out = review.accept_flag(db, E, flag["id"], "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ",
                             remember={"entity_type": "person", "english_gloss": "the poet"})
    assert out["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ ਨੇ ਲਿਖਿਆ"
    seg = repo.get_segment(db, flag["segment_id"])
    assert seg["text_pa"] == out["text_pa"] and seg["edited"] is True
    f = repo.get_flag(db, flag["id"])
    assert f["status"] == "accepted" and f["resolved_text"] == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"
    c = repo.get_correction(db, out["correction_id"])
    assert (c["source"], c["before_text"], c["after_text"], c["ledger_id"]) == \
        ("admin", "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", out["ledger_id"])
    led = repo.get_ledger(db, out["ledger_id"])
    assert led["surface_forms"] == ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ"] and led["entity_type"] == "person"
    assert led["romanized"].startswith("bh") and led["english_gloss"] == "the poet"
    assert led["embedding_model"] == E.name
    assert led["confidence_threshold"] == pytest.approx(0.8)
    assert f["ledger_id"] == out["ledger_id"]


@pytest.mark.db
def test_accept_without_remember_leaves_ledger_alone(db, flag):
    out = review.accept_flag(db, E, flag["id"], "ਕੁਝ ਹੋਰ", remember=None)
    assert out["ledger_id"] is None and repo.list_ledger(db, None, True) == []
    assert repo.get_segment(db, flag["segment_id"])["text_pa"] == "ਕਵੀ ਕੁਝ ਹੋਰ ਜੀ ਨੇ ਲਿਖਿਆ"


@pytest.mark.db
def test_accept_rejects_blank_text_and_non_open_flags(db, flag):
    with pytest.raises(review.ReviewError) as exc:
        review.accept_flag(db, E, flag["id"], "   ", None)
    assert exc.value.status == 400
    review.reject_flag(db, flag["id"])
    with pytest.raises(review.ReviewError) as exc:
        review.accept_flag(db, E, flag["id"], "x", None)
    assert exc.value.status == 409
    with pytest.raises(review.ReviewError) as exc:
        review.accept_flag(db, E, 999999, "x", None)
    assert exc.value.status == 404
    with pytest.raises(review.ReviewError) as exc:
        review.reject_flag(db, flag["id"])
    assert exc.value.status == 409
    with pytest.raises(review.ReviewError) as exc:
        review.reject_flag(db, 999999)
    assert exc.value.status == 404


@pytest.mark.db
def test_accept_marks_flag_stale_when_span_was_edited_away(db, flag):
    repo.update_segment_text(db, flag["segment_id"], "ਬਿਲਕੁਲ ਵੱਖਰਾ ਪਾਠ")
    with pytest.raises(review.ReviewError) as exc:
        review.accept_flag(db, E, flag["id"], "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", None)
    assert exc.value.status == 409
    assert repo.get_flag(db, flag["id"])["status"] == "stale"


@pytest.mark.db
def test_revert_restores_text_reopens_flag_and_refuses_twice(db, flag):
    out = review.accept_flag(db, E, flag["id"], "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", None)
    back = review.revert_correction(db, out["correction_id"])
    assert back["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ"
    assert repo.get_flag(db, flag["id"])["status"] == "open"
    assert repo.get_correction(db, out["correction_id"])["reverted_at"] is not None
    with pytest.raises(review.ReviewError) as exc:
        review.revert_correction(db, out["correction_id"])
    assert exc.value.status == 409


@pytest.mark.db
def test_revert_refuses_when_the_corrected_text_is_gone(db, flag):
    out = review.accept_flag(db, E, flag["id"], "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", None)
    repo.update_segment_text(db, flag["segment_id"], "ਹੁਣ ਕੁਝ ਹੋਰ")
    with pytest.raises(review.ReviewError) as exc:
        review.revert_correction(db, out["correction_id"])
    assert exc.value.status == 409
    with pytest.raises(review.ReviewError) as exc:
        review.revert_correction(db, 999999)
    assert exc.value.status == 404
