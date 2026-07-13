from pathlib import Path

import pytest

from katha import repo, worker
from katha.stages.prepare import PrepareResult
from katha.stages.types import AlignedWord, DiarizedEntry

# As in test_assemble: speaker "0" must clearly out-talk speaker "1" so that the
# reciter heuristic has an unambiguous answer.
ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਐ ਪਾਤਸ਼ਾਹ", start_s=3.5, end_s=5.5),
]
WORDS = [
    AlignedWord(text="ਤੋ", start_s=2.1, end_s=2.3, score=0.9),
    AlignedWord(text="ਸੋ", start_s=2.4, end_s=2.9, score=0.9),      # -> 800 ms of reciter
    AlignedWord(text="ਐ", start_s=3.6, end_s=3.8, score=0.6),
    AlignedWord(text="ਪਾਤਸ਼ਾਹ", start_s=3.9, end_s=5.4, score=0.6),  # -> 1800 ms of kathakar
]


class FakeAsr:
    def __init__(self):
        self.calls = 0

    def transcribe(self, wav, out_dir):
        self.calls += 1
        return ENTRIES, "sarvam-job-1"


class FakeAligner:
    def __init__(self):
        self.text = None

    def align(self, wav, text):
        self.text = text
        return WORDS


@pytest.fixture
def pipeline(db, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "work_dir", tmp_path / "work")
    monkeypatch.setattr(
        worker, "prepare_clip",
        lambda source, work_dir, s: PrepareResult(
            native=work_dir / "native.wav",
            denoised=None,
            asr_ready=work_dir / "asr.wav",
            duration_seconds=90.0,
            native_sample_rate=48000,
        ),
    )
    return worker.Pipeline(settings, db, aligner=FakeAligner(), asr=FakeAsr())


@pytest.mark.db
def test_run_once_walks_a_clip_through_every_stage(db, pipeline):
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    stages = []
    while pipeline.run_once():
        with db.cursor() as cur:
            cur.execute("SELECT stage, status FROM jobs")
            stages.append(tuple(cur.fetchone().values()))

    assert stages == [
        ("asr", "queued"),
        ("align", "queued"),
        ("assemble", "queued"),
        ("complete", "done"),
    ]


@pytest.mark.db
def test_a_completed_clip_has_aligned_segments(db, pipeline):
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    while pipeline.run_once():
        pass

    with db.cursor() as cur:
        cur.execute(
            "SELECT speaker_role, start_ms, timing_source FROM segments"
            " WHERE clip_id = %s ORDER BY idx",
            (clip_id,),
        )
        rows = cur.fetchall()

    assert [r["speaker_role"] for r in rows] == ["reciter", "kathakar"]
    assert rows[0]["start_ms"] == 2100
    assert rows[0]["timing_source"] == "aligned"


@pytest.mark.db
def test_the_aligner_receives_text_built_from_the_diarized_entries(db, pipeline):
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    while pipeline.run_once():
        pass
    assert pipeline.aligner.text == "ਤੋ ਸੋ ਐ ਪਾਤਸ਼ਾਹ"


@pytest.mark.db
def test_a_failing_stage_is_recorded_and_retried_up_to_max_attempts(db, settings, pipeline, monkeypatch):
    monkeypatch.setattr(settings, "max_attempts", 2)
    monkeypatch.setattr(
        worker, "prepare_clip",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg exploded")),
    )
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    assert pipeline.run_once() is True   # attempt 1 -> requeued
    assert pipeline.run_once() is True   # attempt 2 -> exhausted
    assert pipeline.run_once() is False  # nothing claimable

    with db.cursor() as cur:
        cur.execute("SELECT status, attempts, last_error FROM jobs")
        job = cur.fetchone()
    assert job["status"] == "failed"
    assert job["attempts"] == 2
    assert "ffmpeg exploded" in job["last_error"]


@pytest.mark.db
def test_the_daily_throttle_stops_new_work(db, settings, pipeline):
    with db.cursor() as cur:
        cur.execute("UPDATE settings SET value = '0' WHERE key = 'files_per_day'")
    db.commit()
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    assert pipeline.run_once() is False


@pytest.mark.db
def test_every_clip_up_to_the_daily_quota_reaches_done_while_the_next_is_withheld(
    db, settings, pipeline
):
    """Regression test for the critical throttle bug: the quota must gate only
    the *start* of a new clip, never the continuation of one already started.

    files_per_day new clips are registered together (the normal case — the
    watcher registers up to files_per_day per scan). Every one of them must
    fully drain through all four stages to completion in the same day,
    because each only ever consumes one quota unit (stamped once by
    claim_next_job's COALESCE(started_at, now())), regardless of how many of
    its four stages get claimed. A clip registered after the quota is spent
    must not be started at all.
    """
    quota = repo.get_setting_int(db, "files_per_day", 3)
    clip_ids = [
        repo.register_clip(db, Path(f"/a/k{i}.mp3"), f"k{i}", f"sha{i}")
        for i in range(quota)
    ]

    while pipeline.run_once():
        pass

    with db.cursor() as cur:
        cur.execute(
            "SELECT clip_id, stage, status FROM jobs WHERE clip_id = ANY(%s)",
            (clip_ids,),
        )
        jobs_by_clip = {row["clip_id"]: row for row in cur.fetchall()}

    for clip_id in clip_ids:
        job = jobs_by_clip[clip_id]
        assert job["stage"] == "complete"
        assert job["status"] == "done"
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM segments WHERE clip_id = %s", (clip_id,)
            )
            assert cur.fetchone()["n"] > 0

    fourth_clip_id = repo.register_clip(db, Path("/a/k-extra.mp3"), "k-extra", "sha-extra")

    assert pipeline.run_once() is False  # quota spent; nothing left to claim

    with db.cursor() as cur:
        cur.execute(
            "SELECT stage, started_at FROM jobs WHERE clip_id = %s", (fourth_clip_id,)
        )
        row = cur.fetchone()
    assert row["stage"] == "prepare"
    assert row["started_at"] is None
