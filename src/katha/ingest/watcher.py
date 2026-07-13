import hashlib
from pathlib import Path

import psycopg

from katha import repo
from katha.config import Settings

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"}
_CHUNK = 1024 * 1024


def discover(watch_dir: Path) -> list[Path]:
    if not watch_dir.is_dir():
        return []
    return sorted(
        p
        for p in watch_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in AUDIO_SUFFIXES
        and not p.name.startswith(".")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def scan_once(conn: psycopg.Connection, settings: Settings) -> list[int]:
    """Register new clips, up to the daily quota, in sorted filename order.

    Idempotent — re-running over already-registered files is a no-op and does
    not re-hash them: already-registered source paths are looked up once
    up front and skipped before hashing. register_clip's
    ON CONFLICT DO NOTHING remains as the race-safety backstop for the case
    where a file is registered concurrently between that lookup and the
    insert.
    """
    quota = repo.get_setting_int(conn, "files_per_day", 3)
    known = repo.known_source_paths(conn)
    registered: list[int] = []
    for path in discover(settings.watch_dir):
        if str(path) in known:
            continue
        if len(registered) >= quota:
            break
        clip_id = repo.register_clip(conn, path, path.stem, sha256_file(path))
        if clip_id is not None:
            registered.append(clip_id)
    return registered
