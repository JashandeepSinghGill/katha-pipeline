from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from katha import repo
from katha.api.main import app, get_conn, get_app_settings
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਐ ਪਾਤਸ਼ਾਹ", start_s=3.5, end_s=4.5),
]
WORDS = [
    AlignedWord(text="ਤੋ", start_s=2.1, end_s=2.3, score=0.9),
    AlignedWord(text="ਸੋ", start_s=2.4, end_s=2.9, score=0.9),
    AlignedWord(text="ਐ", start_s=3.6, end_s=3.8, score=0.6),
    AlignedWord(text="ਪਾਤਸ਼ਾਹ", start_s=3.9, end_s=4.4, score=0.6),
]


@pytest.fixture
def client(db, settings, tmp_path):
    audio = tmp_path / "native.wav"
    audio.write_bytes(bytes(range(256)) * 4)  # 1024 deterministic bytes

    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_app_settings] = lambda: settings
    yield TestClient(app), audio
    app.dependency_overrides.clear()


@pytest.fixture
def clip_id(db, tmp_path):
    cid = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    repo.save_segments(db, cid, assemble.assemble_segments(ENTRIES, WORDS), reciter_id="1")
    repo.add_audio_file(db, cid, "native", tmp_path / "native.wav", 48000)
    return cid


@pytest.mark.db
def test_get_clip_returns_segments_with_words(client, clip_id):
    api, _ = client
    body = api.get(f"/api/clips/{clip_id}").json()
    assert body["clip"]["reciter_speaker_id"] == "1"
    assert len(body["segments"]) == 2
    assert body["segments"][0]["speaker_role"] == "reciter"
    assert [w["text"] for w in body["segments"][0]["words"]] == ["ਤੋ", "ਸੋ"]


@pytest.mark.db
def test_patching_the_reciter_reassigns_every_role(client, clip_id):
    api, _ = client
    api.patch(f"/api/clips/{clip_id}", json={"reciter_speaker_id": "0"})
    segments = api.get(f"/api/clips/{clip_id}").json()["segments"]
    assert [s["speaker_role"] for s in segments] == ["kathakar", "reciter"]


@pytest.mark.db
def test_editing_a_segment_marks_it_edited(client, clip_id):
    api, _ = client
    segment_id = api.get(f"/api/clips/{clip_id}").json()["segments"][0]["id"]
    api.patch(f"/api/segments/{segment_id}", json={"text_pa": "ਤੋ ਸੋ ਨਹੀ"})
    segments = api.get(f"/api/clips/{clip_id}").json()["segments"]
    assert segments[0]["text_pa"] == "ਤੋ ਸੋ ਨਹੀ"
    assert segments[0]["edited"] is True


@pytest.mark.db
def test_audio_without_a_range_header_returns_the_whole_file(client, clip_id):
    api, _ = client
    r = api.get(f"/api/audio/{clip_id}")
    assert r.status_code == 200
    assert r.headers["accept-ranges"] == "bytes"
    assert len(r.content) == 1024


@pytest.mark.db
def test_audio_with_a_range_header_returns_206_and_only_that_slice(client, clip_id):
    api, audio = client
    r = api.get(f"/api/audio/{clip_id}", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206
    assert r.headers["content-range"] == "bytes 100-199/1024"
    assert r.content == audio.read_bytes()[100:200]


@pytest.mark.db
def test_an_unsatisfiable_range_returns_416(client, clip_id):
    api, _ = client
    r = api.get(f"/api/audio/{clip_id}", headers={"Range": "bytes=99999-"})
    assert r.status_code == 416


def _job_status(db, clip_id: int) -> dict:
    with db.cursor() as cur:
        cur.execute("SELECT status, attempts FROM jobs WHERE clip_id = %s", (clip_id,))
        row = cur.fetchone()
    db.rollback()
    return row


@pytest.mark.db
def test_list_clips_reports_correct_segment_and_loose_counts(client, db):
    api, _ = client
    cid = repo.register_clip(db, Path("/a/mixed.mp3"), "mixed", "mixed-sha")

    # One entry aligns cleanly (word count matches -> timing_source='aligned').
    aligned_entry = [DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0)]
    aligned_words = [
        AlignedWord(text="ਤੋ", start_s=2.1, end_s=2.3, score=0.9),
        AlignedWord(text="ਸੋ", start_s=2.4, end_s=2.9, score=0.9),
    ]
    aligned_segments = assemble.assemble_segments(aligned_entry, aligned_words)

    # One entry has a word-count mismatch -> falls back to timing_source='asr'
    # (this is the loose/untrustworthy case loose_count exists to surface).
    loose_entry = [DiarizedEntry(speaker_id="0", text="ਐ ਪਾਤਸ਼ਾਹ", start_s=3.5, end_s=4.5)]
    loose_segments = assemble.assemble_segments(loose_entry, words=[])

    combined = aligned_segments + [replace(loose_segments[0], idx=1)]
    assert [s.timing_source for s in combined] == ["aligned", "asr"]
    repo.save_segments(db, cid, combined, reciter_id="1")

    body = api.get("/api/clips").json()
    row = next(c for c in body["clips"] if c["id"] == cid)
    assert row["segment_count"] == 2
    assert row["loose_count"] == 1


@pytest.mark.db
def test_retry_requeues_a_failed_job_and_resets_attempts(client, clip_id, db):
    api, _ = client
    with db.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = 'failed', attempts = 3 WHERE clip_id = %s",
            (clip_id,),
        )
    db.commit()

    r = api.post(f"/api/clips/{clip_id}/retry")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "requeued": True}

    row = _job_status(db, clip_id)
    assert row["status"] == "queued"
    assert row["attempts"] == 0


@pytest.mark.db
def test_retry_is_a_no_op_on_a_queued_job(client, clip_id, db):
    api, _ = client
    before = _job_status(db, clip_id)
    assert before["status"] == "queued"  # freshly registered by the clip_id fixture

    r = api.post(f"/api/clips/{clip_id}/retry")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "requeued": False}

    assert _job_status(db, clip_id) == before


@pytest.mark.db
def test_retry_is_a_no_op_on_a_running_job(client, clip_id, db):
    api, _ = client
    with db.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = 'running', attempts = 1 WHERE clip_id = %s",
            (clip_id,),
        )
    db.commit()
    before = _job_status(db, clip_id)
    assert before["status"] == "running"

    r = api.post(f"/api/clips/{clip_id}/retry")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "requeued": False}

    assert _job_status(db, clip_id) == before


@pytest.mark.db
def test_retry_404s_when_the_clip_does_not_exist(client):
    api, _ = client
    r = api.post("/api/clips/999999/retry")
    assert r.status_code == 404


@pytest.mark.db
def test_patching_reciter_to_null_marks_every_segment_unknown(client, clip_id):
    api, _ = client
    r = api.patch(f"/api/clips/{clip_id}", json={"reciter_speaker_id": None})
    assert r.status_code == 200

    body = api.get(f"/api/clips/{clip_id}").json()
    assert body["clip"]["reciter_speaker_id"] is None
    assert len(body["segments"]) == 2
    assert all(s["speaker_role"] == "unknown" for s in body["segments"])


@pytest.mark.db
def test_get_clip_404s_for_an_unknown_id(client):
    api, _ = client
    r = api.get("/api/clips/999999")
    assert r.status_code == 404


@pytest.mark.db
def test_patch_clip_404s_for_an_unknown_id(client):
    api, _ = client
    r = api.patch("/api/clips/999999", json={"reciter_speaker_id": None})
    assert r.status_code == 404


@pytest.mark.db
def test_patch_segment_404s_for_an_unknown_id(client):
    api, _ = client
    r = api.patch("/api/segments/999999", json={"text_pa": "x"})
    assert r.status_code == 404
