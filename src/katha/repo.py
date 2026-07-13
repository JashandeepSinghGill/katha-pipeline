from datetime import datetime
from pathlib import Path

import psycopg

# Transaction contract: every function in this module opens and closes its own
# transaction on the connection it is given (commit on write, rollback on a
# read-only SELECT or a no-op ON CONFLICT branch). This keeps db.connect()'s
# autocommit=False connections from being left idle-in-transaction between
# calls. It also means the `conn.rollback()` used to discard a failed insert
# (see register_clip below) is connection-wide: it discards *everything*
# uncommitted on that connection, not just the failed statement. That is only
# safe because no repo function here ever leaves work uncommitted on entry —
# callers must not pass in a connection that has unmanaged uncommitted writes
# pending from outside this module.


def register_clip(
    conn: psycopg.Connection, source_path: Path, title: str, sha256: str
) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO clips (source_path, title, sha256) VALUES (%s, %s, %s)"
            " ON CONFLICT (source_path) DO NOTHING RETURNING id",
            (str(source_path), title, sha256),
        )
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            return None
        clip_id = row["id"]
        cur.execute("INSERT INTO jobs (clip_id) VALUES (%s)", (clip_id,))
    conn.commit()
    return clip_id


def get_setting_int(conn: psycopg.Connection, key: str, default: int) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
        row = cur.fetchone()
    conn.rollback()
    return int(row["value"]) if row else default


def count_jobs_started_since(conn: psycopg.Connection, since: datetime) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM jobs WHERE started_at >= %s", (since,))
        n = cur.fetchone()["n"]
    conn.rollback()
    return n


def known_source_paths(conn: psycopg.Connection) -> set[str]:
    """Return the source_path of every already-registered clip.

    Read-only; used by scan_once to skip re-hashing files that are already
    in the registry instead of relying solely on register_clip's
    ON CONFLICT DO NOTHING (which still exists as the race-safety backstop).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT source_path FROM clips")
        rows = cur.fetchall()
    conn.rollback()
    return {row["source_path"] for row in rows}


def claim_next_job(conn: psycopg.Connection) -> dict | None:
    """Atomically claim one queued job. SKIP LOCKED makes this safe for >1 worker.

    Only matches status='queued'. 'failed' is a terminal state reached once
    max_attempts is exhausted, and is not reclaimed here: a non-exhausted
    failure is put back to status='queued' by the caller (the Task 7 worker's
    retry path) and picked up normally on the next claim. A job that is
    truly 'failed' stays that way until an admin explicitly requeues it
    (requeue_job, Task 8), which resets it back to 'queued'.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE jobs SET
                status     = 'running',
                attempts   = attempts + 1,
                started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued' AND stage <> 'complete'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, clip_id, stage, attempts, sarvam_job_id
        """)
        row = cur.fetchone()
    conn.commit()
    return row


def set_job(
    conn: psycopg.Connection,
    job_id: int,
    *,
    stage: str | None = None,
    status: str | None = None,
    last_error: str | None = None,
    sarvam_job_id: str | None = None,
) -> None:
    sets, params = ["updated_at = now()"], []
    for column, value in (
        ("stage", stage),
        ("status", status),
        ("last_error", last_error),
        ("sarvam_job_id", sarvam_job_id),
    ):
        if value is not None:
            sets.append(f"{column} = %s")
            params.append(value)
    params.append(job_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s", params)
    conn.commit()
