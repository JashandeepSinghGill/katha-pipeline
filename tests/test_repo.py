from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from katha import repo


@pytest.mark.db
def test_register_clip_is_idempotent_on_source_path(db):
    first = repo.register_clip(db, Path("/audio/katha-01.mp3"), "katha-01", "abc")
    second = repo.register_clip(db, Path("/audio/katha-01.mp3"), "katha-01", "abc")
    assert isinstance(first, int)
    assert second is None


@pytest.mark.db
def test_register_clip_creates_a_queued_prepare_job(db):
    clip_id = repo.register_clip(db, Path("/audio/k.mp3"), "k", "abc")
    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        job = cur.fetchone()
    assert job == {"stage": "prepare", "status": "queued"}


@pytest.mark.db
def test_claim_next_job_marks_it_running_and_increments_attempts(db):
    repo.register_clip(db, Path("/audio/k.mp3"), "k", "abc")
    job = repo.claim_next_job(db)
    assert job["stage"] == "prepare"
    assert job["attempts"] == 1
    assert repo.claim_next_job(db) is None  # nothing left to claim


@pytest.mark.db
def test_count_jobs_started_since_ignores_older_jobs(db):
    repo.register_clip(db, Path("/audio/k.mp3"), "k", "abc")
    repo.claim_next_job(db)
    now = datetime.now(timezone.utc)
    assert repo.count_jobs_started_since(db, now - timedelta(hours=24)) == 1
    assert repo.count_jobs_started_since(db, now + timedelta(hours=1)) == 0
