import json
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


def _write_sarvam_json(out_dir: Path, entries: list[DiarizedEntry]) -> None:
    """Mirror what SarvamAsr.transcribe() really does: download the diarized
    result as JSON into out_dir, in Sarvam's own wire shape (parseable by
    katha.stages.asr.parse_diarized). This is the durable artifact that
    worker._load_entries re-reads — real fakes must produce it too, or a test
    can't exercise the on-disk handoff at all."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "diarized_transcript": {
            "entries": [
                {
                    "speaker_id": e.speaker_id,
                    "transcript": e.text,
                    "start_time_seconds": e.start_s,
                    "end_time_seconds": e.end_s,
                }
                for e in entries
            ]
        }
    }
    (out_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


class FakeAsr:
    def __init__(self):
        self.calls = 0

    def transcribe(self, wav, out_dir):
        self.calls += 1
        _write_sarvam_json(out_dir, ENTRIES)
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
    # This test is about the attempts budget, not backoff (covered separately) —
    # a nonzero backoff would withhold the retried job from the second claim and
    # turn it into a false failure of this test.
    monkeypatch.setattr(settings, "retry_backoff_seconds", 0)
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


# --- B1: no stage may depend on in-memory state from a previous Pipeline -------
#
# The worker is a `Type=oneshot` systemd unit: a Pipeline object lives for
# exactly one process invocation. If a crash, reboot, or admin Retry hands a
# clip's next stage to a *different* Pipeline instance, that instance must be
# able to run the stage from durable, on-disk state alone.


@pytest.mark.db
def test_a_brand_new_pipeline_can_resume_align_and_assemble_after_a_different_process_ran_asr(
    db, settings, tmp_path, monkeypatch
):
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
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    # Process 1: runs prepare and asr, then "exits" (crash/reboot/end of the
    # 15-minute systemd timer window). Nothing about this Pipeline object
    # survives into process 2 below except what it wrote to Postgres and disk.
    process_one = worker.Pipeline(settings, db, aligner=FakeAligner(), asr=FakeAsr())
    assert process_one.run_once() is True  # prepare -> asr
    assert process_one.run_once() is True  # asr -> align

    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        mid = cur.fetchone()
    assert mid == {"stage": "align", "status": "queued"}

    # Process 2: a brand-new Pipeline, sharing nothing with process 1 but the DB
    # connection and the on-disk work directory. Against the old in-memory-cache
    # code this KeyErrors inside _align (self._entries[clip_id] was never
    # populated in *this* object), which the generic except in run_once turns
    # into a requeue-then-eventually-failed clip instead of a successful align.
    process_two = worker.Pipeline(settings, db, aligner=FakeAligner(), asr=FakeAsr())
    assert process_two.run_once() is True  # align
    assert process_two.run_once() is True  # assemble -> complete

    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        final = cur.fetchone()
    assert final == {"stage": "complete", "status": "done"}

    with db.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM segments WHERE clip_id = %s", (clip_id,))
        assert cur.fetchone()["n"] == 2


@pytest.mark.db
def test_admin_retry_after_an_align_failure_succeeds_in_a_fresh_pipeline(
    db, settings, tmp_path, monkeypatch
):
    """The admin-Retry regression from the review: a clip that fails at `align`
    goes terminal-failed; requeue_job resets attempts/status but leaves stage
    at 'align'. The worker process that eventually claims it again is, in
    production, always a fresh one (oneshot systemd). Old in-memory-cache code
    KeyErrors here every time, making the clip permanently unrecoverable."""
    monkeypatch.setattr(settings, "work_dir", tmp_path / "work")
    monkeypatch.setattr(settings, "max_attempts", 1)
    monkeypatch.setattr(settings, "retry_backoff_seconds", 0)
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
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    class BoomAligner:
        def align(self, wav, text):
            raise RuntimeError("CUDA OOM")

    doomed = worker.Pipeline(settings, db, aligner=BoomAligner(), asr=FakeAsr())
    assert doomed.run_once() is True  # prepare -> asr
    assert doomed.run_once() is True  # asr -> align
    assert doomed.run_once() is True  # align fails, max_attempts=1 -> terminal failed

    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        failed = cur.fetchone()
    assert failed == {"stage": "align", "status": "failed"}

    assert repo.requeue_job(db, clip_id) is True

    # The admin clicks Retry, and whatever worker process picks this up next is
    # a fresh one — never the one that ran asr.
    rescuer = worker.Pipeline(settings, db, aligner=FakeAligner(), asr=FakeAsr())
    assert rescuer.run_once() is True  # align, this time succeeding
    assert rescuer.run_once() is True  # assemble -> complete

    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        final = cur.fetchone()
    assert final == {"stage": "complete", "status": "done"}


# --- B2: attempts must reset on a successful stage transition ------------------


@pytest.mark.db
def test_align_gets_its_own_full_retry_budget_even_though_prepare_and_asr_already_succeeded(
    db, settings, pipeline, monkeypatch
):
    monkeypatch.setattr(settings, "max_attempts", 3)
    monkeypatch.setattr(settings, "retry_backoff_seconds", 0)
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    assert pipeline.run_once() is True  # prepare -> asr
    assert pipeline.run_once() is True  # asr -> align

    with db.cursor() as cur:
        cur.execute("SELECT stage, attempts FROM jobs")
        row = cur.fetchone()
    # Against the old code (no reset on success) this is already 2 here, one
    # short of max_attempts=3 before align has even been tried once.
    assert row == {"stage": "align", "attempts": 0}

    monkeypatch.setattr(
        pipeline._aligner, "align",
        lambda wav, text: (_ for _ in ()).throw(RuntimeError("CUDA OOM")),
    )

    assert pipeline.run_once() is True   # align attempt 1/3 -> requeued
    assert pipeline.run_once() is True   # align attempt 2/3 -> requeued
    assert pipeline.run_once() is True   # align attempt 3/3 -> exhausted -> failed
    assert pipeline.run_once() is False  # terminal; nothing left to claim

    with db.cursor() as cur:
        cur.execute("SELECT stage, status, attempts, last_error FROM jobs")
        job = cur.fetchone()
    assert job["stage"] == "align"
    assert job["status"] == "failed"
    assert job["attempts"] == 3
    assert "CUDA OOM" in job["last_error"]


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
    quota = 3
    with db.cursor() as cur:
        cur.execute(
            "UPDATE settings SET value = %s WHERE key = 'files_per_day'", (str(quota),)
        )
    db.commit()
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


# --- S1: the work directory must not grow without bound -------------------------


@pytest.mark.db
def test_assemble_reclaims_denoised_and_asr_wav_but_keeps_native(
    db, settings, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "work_dir", tmp_path / "work")
    monkeypatch.setattr(
        worker, "prepare_clip",
        lambda source, work_dir, s: PrepareResult(
            native=work_dir / "native.wav",
            denoised=work_dir / "native_DeepFilterNet3.wav",
            asr_ready=work_dir / "asr.wav",
            duration_seconds=90.0,
            native_sample_rate=48000,
        ),
    )
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    pipeline = worker.Pipeline(settings, db, aligner=FakeAligner(), asr=FakeAsr())
    work_dir = pipeline._work_dir(clip_id)
    work_dir.mkdir(parents=True)
    for name in ("native.wav", "native_DeepFilterNet3.wav", "asr.wav"):
        (work_dir / name).write_bytes(b"x" * 10)

    while pipeline.run_once():
        pass

    # native.wav is kept — /api/audio serves it to the admin review UI.
    assert (work_dir / "native.wav").exists()
    # denoised and the 16k ASR wav are dead weight once assemble has run.
    assert not (work_dir / "native_DeepFilterNet3.wav").exists()
    assert not (work_dir / "asr.wav").exists()

    with db.cursor() as cur:
        cur.execute("SELECT kind FROM audio_files WHERE clip_id = %s", (clip_id,))
        kinds = {r["kind"] for r in cur.fetchall()}
    assert kinds == {"original", "native"}  # denoised/asr_ready rows removed too


@pytest.mark.db
def test_a_cleanup_failure_does_not_fail_an_otherwise_successful_clip(
    db, settings, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "work_dir", tmp_path / "work")
    monkeypatch.setattr(
        worker, "prepare_clip",
        lambda source, work_dir, s: PrepareResult(
            native=work_dir / "native.wav",
            # Deliberately point "denoised" at a directory, not a file: unlink()
            # on a directory raises OSError (not FileNotFoundError) on both
            # Windows and POSIX, giving a deterministic, cross-platform way to
            # exercise the non-fatal-cleanup path without relying on OS-specific
            # file locking.
            denoised=work_dir / "denoised_dir",
            asr_ready=work_dir / "asr.wav",
            duration_seconds=90.0,
            native_sample_rate=48000,
        ),
    )
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    pipeline = worker.Pipeline(settings, db, aligner=FakeAligner(), asr=FakeAsr())
    work_dir = pipeline._work_dir(clip_id)
    (work_dir / "denoised_dir").mkdir(parents=True)

    while pipeline.run_once():
        pass

    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        job = cur.fetchone()
    # The clip still completes: a stuck denoised "file" must not fail a clip
    # that otherwise transcribed fine.
    assert job == {"stage": "complete", "status": "done"}
    assert (work_dir / "denoised_dir").exists()  # unlink failed; left in place

    with db.cursor() as cur:
        cur.execute(
            "SELECT kind FROM audio_files WHERE clip_id = %s AND kind = 'denoised'",
            (clip_id,),
        )
        # The row is kept too — deleting it would make the DB claim the file is
        # gone when it (still) isn't.
        assert cur.fetchone() is not None
