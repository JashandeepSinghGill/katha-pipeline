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
