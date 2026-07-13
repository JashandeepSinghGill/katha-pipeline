from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
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
    job = repo.claim_next_job(db, quota_left=1)
    assert job["stage"] == "prepare"
    assert job["attempts"] == 1
    assert repo.claim_next_job(db, quota_left=1) is None  # nothing left to claim


@pytest.mark.db
def test_claim_next_job_will_not_start_a_new_clip_when_quota_is_exhausted(db):
    repo.register_clip(db, Path("/audio/k.mp3"), "k", "abc")
    assert repo.claim_next_job(db, quota_left=0) is None


@pytest.mark.db
def test_claim_next_job_always_continues_an_in_flight_clip_regardless_of_quota(db):
    repo.register_clip(db, Path("/audio/k.mp3"), "k", "abc")
    first = repo.claim_next_job(db, quota_left=1)  # stamps started_at once
    repo.set_job(db, first["id"], stage="asr", status="queued")

    # Quota is exhausted, but this clip already paid its quota unit on the
    # first claim, so continuing it must not be gated.
    second = repo.claim_next_job(db, quota_left=0)
    assert second is not None
    assert second["stage"] == "asr"
    assert second["attempts"] == 2


@pytest.mark.db
def test_claim_next_job_prefers_in_flight_clips_over_new_ones(db):
    started = repo.register_clip(db, Path("/audio/k1.mp3"), "k1", "abc")
    repo.register_clip(db, Path("/audio/k2.mp3"), "k2", "def")

    first = repo.claim_next_job(db, quota_left=2)
    assert first["clip_id"] == started
    repo.set_job(db, first["id"], stage="asr", status="queued")

    # k2 (new, created after k1) would sort first by created_at alone, but the
    # already-started k1 must be claimed ahead of it.
    second = repo.claim_next_job(db, quota_left=2)
    assert second["clip_id"] == started
    assert second["stage"] == "asr"


@pytest.mark.db
def test_count_jobs_started_since_ignores_older_jobs(db):
    repo.register_clip(db, Path("/audio/k.mp3"), "k", "abc")
    repo.claim_next_job(db, quota_left=1)
    now = datetime.now(timezone.utc)
    assert repo.count_jobs_started_since(db, now - timedelta(hours=24)) == 1
    assert repo.count_jobs_started_since(db, now + timedelta(hours=1)) == 0


@pytest.mark.db
def test_reclaim_stale_jobs_requeues_old_running_jobs_without_resetting_attempts(db):
    repo.register_clip(db, Path("/audio/k.mp3"), "k", "abc")
    job = repo.claim_next_job(db, quota_left=1)  # status='running', attempts=1
    with db.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET updated_at = now() - interval '7 hours' WHERE id = %s",
            (job["id"],),
        )
    db.commit()

    older_than = datetime.now(timezone.utc) - timedelta(hours=6)
    assert repo.reclaim_stale_jobs(db, older_than) == 1

    with db.cursor() as cur:
        cur.execute("SELECT status, attempts FROM jobs WHERE id = %s", (job["id"],))
        row = cur.fetchone()
    assert row["status"] == "queued"
    assert row["attempts"] == 1  # unchanged — max_attempts still bounds total tries


@pytest.mark.db
def test_reclaim_stale_jobs_leaves_recently_updated_running_jobs_alone(db):
    repo.register_clip(db, Path("/audio/k.mp3"), "k", "abc")
    job = repo.claim_next_job(db, quota_left=1)  # status='running', updated_at=now()

    older_than = datetime.now(timezone.utc) - timedelta(hours=6)
    assert repo.reclaim_stale_jobs(db, older_than) == 0

    with db.cursor() as cur:
        cur.execute("SELECT status FROM jobs WHERE id = %s", (job["id"],))
        row = cur.fetchone()
    assert row["status"] == "running"


@pytest.mark.db
def test_get_setting_int_does_not_leave_an_open_transaction(db):
    repo.get_setting_int(db, "files_per_day", 3)
    assert db.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


@pytest.mark.db
def test_count_jobs_started_since_does_not_leave_an_open_transaction(db):
    repo.count_jobs_started_since(db, datetime.now(timezone.utc))
    assert db.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


@pytest.mark.db
def test_known_source_paths_returns_registered_paths_only(db):
    repo.register_clip(db, Path("/audio/k1.mp3"), "k1", "abc")
    repo.register_clip(db, Path("/audio/k2.mp3"), "k2", "def")
    assert repo.known_source_paths(db) == {
        str(Path("/audio/k1.mp3")),
        str(Path("/audio/k2.mp3")),
    }


@pytest.mark.db
def test_known_source_paths_does_not_leave_an_open_transaction(db):
    repo.known_source_paths(db)
    assert db.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
