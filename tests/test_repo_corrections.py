from pathlib import Path

import pytest

from katha import repo
from katha.embed import HashingEmbedder
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ", start_s=3.5, end_s=5.5),
]
WORDS = [
    AlignedWord("ਤੋ", 2.1, 2.3, 0.9), AlignedWord("ਸੋ", 2.4, 2.9, 0.9),
    AlignedWord("ਭਾਈ", 3.6, 3.8, 0.6), AlignedWord("ਸੰਤੋਖ", 3.9, 4.4, 0.6),
    AlignedWord("ਸਿੰਗ", 4.5, 4.9, 0.6), AlignedWord("ਜੀ", 5.0, 5.4, 0.6),
]
E = HashingEmbedder()


@pytest.fixture
def clip(db):
    cid = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    repo.save_segments(db, cid, assemble.assemble_segments(ENTRIES, WORDS), reciter_id="1")
    return cid


def seg_ids(db, clip_id):
    return [r["id"] for r in repo.list_segment_texts(db, clip_id)]


@pytest.mark.db
def test_settings_round_trip_and_typed_getters(db):
    assert repo.get_setting(db, "llm_model", "x") == "claude-sonnet-5"
    assert repo.get_setting(db, "missing", "dflt") == "dflt"
    assert repo.get_setting_float(db, "ledger_context_threshold", 0.0) == 0.8
    assert repo.get_setting_bool(db, "auto_apply_enabled", False) is True
    repo.set_setting(db, "auto_apply_enabled", "false")
    assert repo.get_setting_bool(db, "auto_apply_enabled", True) is False
    assert repo.get_all_settings(db)["auto_apply_enabled"] == "false"


@pytest.mark.db
def test_save_flags_replaces_open_flags_but_keeps_resolved_ones(db, clip):
    s1, s2 = seg_ids(db, clip)
    n = repo.save_flags(db, clip, [(s2, "ਸਿੰਗ", "proper_noun", "r", ["ਸਿੰਘ"]),
                                   (s1, "ਸੋ", "other", "r2", ["ਸੌ"])])
    assert n == 2
    open_ = repo.list_open_flags(db, clip)
    assert [(f["segment_id"], f["span_text"], f["candidates"]) for f in open_] == [
        (s1, "ਸੋ", ["ਸੌ"]), (s2, "ਸਿੰਗ", ["ਸਿੰਘ"])]
    assert open_[1]["text_pa"] == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ"
    repo.resolve_flag(db, open_[1]["id"], "accepted", resolved_text="ਸਿੰਘ")
    # re-running the stage: the open flag is replaced; the same span on the same
    # segment is NOT re-inserted because a resolved flag for it already exists
    n = repo.save_flags(db, clip, [(s2, "ਸਿੰਗ", "proper_noun", "r", ["ਸਿੰਘ"]),
                                   (s1, "ਤੋ", "other", "r3", ["ਤੌ"])])
    assert n == 1
    assert [f["span_text"] for f in repo.list_open_flags(db, clip)] == ["ਤੋ"]
    flag = repo.get_flag(db, open_[1]["id"])
    assert flag["status"] == "accepted" and flag["clip_id"] == clip


@pytest.mark.db
def test_flag_suggestion_reopen_and_segment_accessors(db, clip):
    s1, s2 = seg_ids(db, clip)
    repo.save_flags(db, clip, [(s2, "ਸਿੰਗ", "proper_noun", "r", ["ਸਿੰਘ", "ਸਿੰਘ੨"])])
    fid = repo.list_open_flags(db, clip)[0]["id"]
    lid = repo.upsert_ledger(db, canonical="ਸਿੰਘ", romanized="siṅgh", surface_form="ਸਿੰਗ",
                             entity_type="term", english_gloss=None, embedding=None,
                             embedding_model=None, threshold=0.8)
    repo.set_flag_suggestion(db, fid, lid, ["ਸਿੰਘ", "ਸਿੰਘ੨"])
    f = repo.get_flag(db, fid)
    assert f["ledger_id"] == lid and f["candidates"] == ["ਸਿੰਘ", "ਸਿੰਘ੨"]
    repo.resolve_flag(db, fid, "rejected")
    assert repo.get_flag(db, fid)["status"] == "rejected"
    repo.reopen_flag(db, fid)
    f = repo.get_flag(db, fid)
    assert f["status"] == "open" and f["resolved_text"] is None and f["resolved_at"] is None
    assert repo.get_segment(db, s2)["text_pa"] == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ"
    assert repo.get_segment(db, 999999) is None


@pytest.mark.db
def test_apply_auto_correction_is_one_transaction_and_shows_up_in_reads(db, clip):
    s1, s2 = seg_ids(db, clip)
    repo.save_flags(db, clip, [(s2, "ਸਿੰਗ", "proper_noun", "r", ["ਸਿੰਘ"])])
    fid = repo.list_open_flags(db, clip)[0]["id"]
    lid = repo.upsert_ledger(db, canonical="ਸਿੰਘ", romanized="siṅgh", surface_form="ਸਿੰਗ",
                             entity_type="term", english_gloss=None,
                             embedding=E.embed("ਭਾਈ ਸੰਤੋਖ ਜੀ"), embedding_model=E.name,
                             threshold=0.8)
    cid = repo.apply_auto_correction(db, flag_id=fid, segment_id=s2, ledger_id=lid,
                                     new_text="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ", before="ਸਿੰਗ", after="ਸਿੰਘ",
                                     similarity=0.93)
    seg = repo.get_segment(db, s2)
    assert seg["text_pa"] == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ" and seg["edited"] is True
    c = repo.get_correction(db, cid)
    assert (c["source"], c["before_text"], c["after_text"], c["flag_id"], c["ledger_id"]) == \
        ("auto", "ਸਿੰਗ", "ਸਿੰਘ", fid, lid)
    assert c["context_similarity"] == pytest.approx(0.93)
    assert repo.get_flag(db, fid)["status"] == "auto_applied"
    led = repo.get_ledger(db, lid)
    assert led["times_applied"] == 1 and led["last_used_at"] is not None
    segs = repo.list_segments(db, clip)
    assert segs[1]["flags"][0]["status"] == "auto_applied"
    assert segs[1]["corrections"][0]["after_text"] == "ਸਿੰਘ"
    assert segs[1]["corrections"][0]["reverted"] is False
    clips = repo.list_clips(db)
    assert clips[0]["open_flags"] == 0 and clips[0]["auto_applied"] == 1
    repo.mark_reverted(db, cid)
    assert repo.get_correction(db, cid)["reverted_at"] is not None
    assert repo.list_clips(db)[0]["auto_applied"] == 0
    assert repo.list_segments(db, clip)[1]["corrections"][0]["reverted"] is True


@pytest.mark.db
def test_ledger_upsert_merges_surface_forms_and_embeddings(db):
    v1 = E.embed("ਕਵੀ ਜੀ ਨੇ ਲਿਖਿਆ")
    lid = repo.upsert_ledger(db, canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", romanized="bhai",
                             surface_form="ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", entity_type="person",
                             english_gloss="the poet", embedding=v1,
                             embedding_model=E.name, threshold=0.8)
    v2 = E.embed("ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਦੇ ਰਚੇਤਾ")
    lid2 = repo.upsert_ledger(db, canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", romanized="bhai",
                              surface_form="ਭਾਈ ਸੰਤੋਕ ਸਿੰਘ", entity_type="person",
                              english_gloss=None, embedding=v2,
                              embedding_model=E.name, threshold=0.5)
    assert lid2 == lid
    entries = repo.ledger_entries(db)
    assert len(entries) == 1
    e = entries[0]
    assert set(e.surface_forms) == {"ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "ਭਾਈ ਸੰਤੋਕ ਸਿੰਘ"}
    assert e.threshold == pytest.approx(0.8)          # an existing threshold is kept
    assert e.embedding_model == E.name and len(e.embedding) == E.dim
    assert e.embedding != tuple(v1) and e.embedding != tuple(v2)   # merged
    row = repo.get_ledger(db, lid)
    assert row["english_gloss"] == "the poet"          # not overwritten by None
    assert repo.top_gazetteer(db) == [("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", "person")]
    assert repo.list_ledger(db, q="santokh", include_retired=False) == []
    assert [r["id"] for r in repo.list_ledger(db, q="bhai", include_retired=False)] == [lid]
    assert [r["id"] for r in repo.list_ledger(db, q="ਸੰਤੋਕ", include_retired=False)] == [lid]
    assert repo.update_ledger(db, lid, retired=True, confidence_threshold=0.9) is True
    assert repo.ledger_entries(db) == []
    assert repo.top_gazetteer(db) == []
    assert repo.list_ledger(db, q=None, include_retired=False) == []
    row = repo.list_ledger(db, q=None, include_retired=True)[0]
    assert row["confidence_threshold"] == pytest.approx(0.9)
    assert repo.update_ledger(db, 424242, retired=False) is False
    assert repo.update_ledger(db, lid) is False
    with pytest.raises(ValueError):
        repo.update_ledger(db, lid, times_applied=99)


@pytest.mark.db
def test_upsert_with_embedding_from_another_model_replaces_instead_of_merging(db):
    lid = repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="talvandi", surface_form="ਤਲਬੰਡੀ",
                             entity_type="place", english_gloss=None, embedding=[1.0, 0.0],
                             embedding_model="old-2", threshold=0.8)
    v = E.embed("ਜਨਮ ਅਸਥਾਨ")
    repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="talvandi", surface_form="ਤਲਵੰਢੀ",
                       entity_type="place", english_gloss=None, embedding=v,
                       embedding_model=E.name, threshold=0.8)
    e = repo.ledger_entries(db)[0]
    assert e.id == lid and e.embedding_model == E.name
    assert e.embedding == pytest.approx(tuple(v))


@pytest.mark.db
def test_upsert_revives_a_retired_entry(db):
    lid = repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="t", surface_form=None,
                             entity_type="place", english_gloss=None, embedding=None,
                             embedding_model=None, threshold=0.8)
    repo.update_ledger(db, lid, retired=True)
    repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="t", surface_form="ਤਲਬੰਡੀ",
                       entity_type="place", english_gloss=None, embedding=None,
                       embedding_model=None, threshold=0.8)
    assert repo.get_ledger(db, lid)["retired"] is False


def test_write_text_atomic(tmp_path):
    from katha.fsutil import write_text_atomic
    target = tmp_path / "x.json"
    write_text_atomic(target, "ਪਹਿਲਾ")
    write_text_atomic(target, "ਦੂਜਾ")
    assert target.read_text(encoding="utf-8") == "ਦੂਜਾ"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
