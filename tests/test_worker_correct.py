import json
from pathlib import Path

import pytest

from katha import repo, worker
from katha.embed import HashingEmbedder
from katha.stages.prepare import PrepareResult
from katha.stages.types import AlignedWord, DiarizedEntry
from tests.test_worker import FakeAligner, FakeAsr, FakeLlm, _write_sarvam_json

ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ", start_s=3.5, end_s=5.5),
    DiarizedEntry(speaker_id="0", text="ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦਾ ਜਨਮ ਤਲਬੰਡੀ ਵਿੱਚ ਹੋਇਆ", start_s=6.0, end_s=9.0),
]
WORDS = [AlignedWord(w, 2.0 + i * 0.3, 2.2 + i * 0.3, 0.9)
         for i, w in enumerate(" ".join(e.text for e in ENTRIES).split())]
E = HashingEmbedder()


class Asr(FakeAsr):
    def transcribe(self, wav, out_dir):
        self.calls += 1
        _write_sarvam_json(out_dir, ENTRIES)
        return ENTRIES, "sarvam-job-1"


class Aligner(FakeAligner):
    def align(self, wav, text):
        self.text = text
        return WORDS


def make_pipeline(db, settings, tmp_path, monkeypatch, llm):
    monkeypatch.setattr(settings, "work_dir", tmp_path / "work")
    monkeypatch.setattr(
        worker, "prepare_clip",
        lambda source, work_dir, s: PrepareResult(
            native=work_dir / "native.wav", denoised=None, asr_ready=work_dir / "asr.wav",
            duration_seconds=90.0, native_sample_rate=48000),
    )
    return worker.Pipeline(settings, db, aligner=Aligner(), asr=Asr(), llm=llm, embedder=E)


def run_to_end(db, pipeline):
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    while pipeline.run_once():
        pass
    return clip_id


FLAGS = [
    {"segment_idx": 1, "span_text": "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "kind": "proper_noun", "reason": "name",
     "candidates": ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"]},
    {"segment_idx": 2, "span_text": "ਤਲਬੰਡੀ", "kind": "place", "reason": "place",
     "candidates": ["ਤਲਵੰਡੀ"]},
    {"segment_idx": 2, "span_text": "ਨਹੀਂ ਹੈ", "kind": "garbled", "reason": "not verbatim -> dropped",
     "candidates": ["x"]},
]


@pytest.mark.db
def test_correct_stage_persists_validated_flags_and_completes(db, settings, tmp_path, monkeypatch):
    llm = FakeLlm(FLAGS)
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, llm)
    clip_id = run_to_end(db, pipeline)
    with db.cursor() as cur:
        cur.execute("SELECT stage, status, last_error FROM jobs WHERE clip_id = %s", (clip_id,))
        assert tuple(cur.fetchone().values()) == ("complete", "done", None)
    db.rollback()
    open_ = repo.list_open_flags(db, clip_id)
    assert [(f["span_text"], f["kind"], f["candidates"]) for f in open_] == [
        ("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "proper_noun", ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"]),
        ("ਤਲਬੰਡੀ", "other", ["ਤਲਵੰਡੀ"]),     # 'place' is not a flag kind -> 'other'
    ]
    assert llm.calls == 1  # 3 segments, window of 40 -> one call
    cache = tmp_path / "work" / str(clip_id) / "flags" / "0.json"
    assert cache.exists() and len(json.loads(cache.read_text(encoding="utf-8"))) == 2


@pytest.mark.db
def test_correct_stage_windows_by_setting_and_reuses_cached_windows_on_retry(
    db, settings, tmp_path, monkeypatch
):
    repo.set_setting(db, "correct_window_segments", "1")
    repo.set_setting(db, "correct_context_segments", "0")

    class Flaky(FakeLlm):
        def complete_json(self, *, system, user, schema):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transient LLM outage")
            return {"flags": []}

    llm = Flaky()
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, llm)
    monkeypatch.setattr(settings, "retry_backoff_seconds", 0)
    clip_id = run_to_end(db, pipeline)
    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        assert tuple(cur.fetchone().values()) == ("complete", "done")
    db.rollback()
    # 3 windows. Attempt 1: window 0 ok (cached), window 1 raises. The retry
    # replays window 0 from disk and asks only windows 1 and 2 -> 4 calls in
    # total. Without the cache the retry would re-ask all three -> 5.
    assert llm.calls == 4
    assert sorted(p.name for p in (tmp_path / "work" / str(clip_id) / "flags").iterdir()) == \
        ["0.json", "1.json", "2.json"]


@pytest.mark.db
def test_ledger_pass_auto_applies_when_context_matches_and_suggests_otherwise(
    db, settings, tmp_path, monkeypatch
):
    # Learned in a near-identical context -> auto. Learned elsewhere -> suggest only.
    repo.upsert_ledger(db, canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", romanized="bhai santokh singh",
                       surface_form="ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", entity_type="person", english_gloss=None,
                       embedding=E.embed("ਕਵੀ ਜੀ ਨੇ ਲਿਖਿਆ"), embedding_model=E.name, threshold=0.8)
    repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="talvandi", surface_form="ਤਲਬੰਡੀ",
                       entity_type="place", english_gloss=None,
                       embedding=E.embed("ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਗਰਮ ਹੈ ਅਤੇ ਧੁੱਪ ਤੇਜ਼ ਹੈ"),
                       embedding_model=E.name, threshold=0.8)
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, FakeLlm(FLAGS))
    clip_id = run_to_end(db, pipeline)

    segs = repo.list_segments(db, clip_id)
    assert segs[1]["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ ਨੇ ਲਿਖਿਆ"
    assert segs[1]["edited"] is True
    assert segs[1]["flags"][0]["status"] == "auto_applied"
    assert segs[1]["corrections"][0]["source"] == "auto"
    assert segs[1]["corrections"][0]["context_similarity"] >= 0.8

    assert segs[2]["text_pa"] == "ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦਾ ਜਨਮ ਤਲਬੰਡੀ ਵਿੱਚ ਹੋਇਆ"  # untouched
    f = segs[2]["flags"][0]
    assert f["status"] == "open" and f["candidates"][0] == "ਤਲਵੰਡੀ" and f["ledger_id"] is not None

    assert repo.list_clips(db)[0]["open_flags"] == 1
    assert repo.list_clips(db)[0]["auto_applied"] == 1
    assert repo.get_ledger(db, segs[1]["flags"][0]["ledger_id"])["times_applied"] == 1


@pytest.mark.db
def test_auto_apply_can_be_switched_off_but_suggestions_remain(db, settings, tmp_path, monkeypatch):
    repo.set_setting(db, "auto_apply_enabled", "false")
    repo.upsert_ledger(db, canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", romanized="b", surface_form="ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ",
                       entity_type="person", english_gloss=None,
                       embedding=E.embed("ਕਵੀ ਜੀ ਨੇ ਲਿਖਿਆ"), embedding_model=E.name, threshold=0.8)
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, FakeLlm(FLAGS[:1]))
    clip_id = run_to_end(db, pipeline)
    seg = repo.list_segments(db, clip_id)[1]
    assert seg["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ"
    assert seg["flags"][0]["status"] == "open" and seg["flags"][0]["ledger_id"] is not None


@pytest.mark.db
def test_llm_failure_is_a_normal_stage_failure_with_the_message_on_the_job(
    db, settings, tmp_path, monkeypatch
):
    from katha.llm import LLMError

    class Dead(FakeLlm):
        def complete_json(self, *, system, user, schema):
            raise LLMError("claude -p reported an error: Not logged in · Please run /login")

    monkeypatch.setattr(settings, "max_attempts", 1)
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, Dead())
    run_to_end(db, pipeline)
    row = repo.list_clips(db)[0]
    assert (row["stage"], row["status"]) == ("correct", "failed")
    assert "Not logged in" in row["last_error"]
    assert row["segment_count"] == 3  # transcript still reviewable


@pytest.mark.db
def test_system_prompt_carries_the_gazetteer(db, settings, tmp_path, monkeypatch):
    repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="t", surface_form=None, entity_type="place",
                       english_gloss=None, embedding=None, embedding_model=None, threshold=0.8)

    class Spy(FakeLlm):
        def complete_json(self, *, system, user, schema):
            self.system = system
            self.user = user
            return {"flags": []}

    llm = Spy()
    run_to_end(db, make_pipeline(db, settings, tmp_path, monkeypatch, llm))
    assert "ਤਲਵੰਡੀ (place)" in llm.system
    assert '"target": true' in llm.user


@pytest.mark.db
def test_clip_with_no_segments_completes_correct_stage_without_an_llm_call(
    db, settings, tmp_path, monkeypatch
):
    llm = FakeLlm()
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, llm)
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    with db.cursor() as cur:
        cur.execute("UPDATE jobs SET stage = 'correct' WHERE clip_id = %s", (clip_id,))
    db.commit()
    assert pipeline.run_once() is True
    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        assert tuple(cur.fetchone().values()) == ("complete", "done")
    db.rollback()
    assert llm.calls == 0
