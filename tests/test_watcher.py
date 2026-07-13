from pathlib import Path

import pytest

from katha.ingest import watcher


def test_discover_finds_audio_and_ignores_everything_else(tmp_path):
    (tmp_path / "b.mp3").write_bytes(b"x")
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.mp3").write_bytes(b"x")
    assert watcher.discover(tmp_path) == [tmp_path / "a.wav", tmp_path / "b.mp3"]


def test_sha256_file_is_stable(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"hello")
    assert watcher.sha256_file(f) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


@pytest.mark.db
def test_scan_once_respects_the_daily_throttle(db, settings, tmp_path, monkeypatch):
    for i in range(5):
        (tmp_path / f"k{i}.wav").write_bytes(b"x")
    monkeypatch.setattr(settings, "watch_dir", tmp_path)
    with db.cursor() as cur:
        cur.execute("UPDATE settings SET value = '2' WHERE key = 'files_per_day'")
    db.commit()

    assert len(watcher.scan_once(db, settings)) == 2
    assert len(watcher.scan_once(db, settings)) == 2  # registration alone doesn't consume quota
