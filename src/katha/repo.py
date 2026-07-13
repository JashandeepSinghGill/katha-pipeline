from datetime import datetime
from pathlib import Path

import psycopg


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
    return int(row["value"]) if row else default


def count_jobs_started_since(conn: psycopg.Connection, since: datetime) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM jobs WHERE started_at >= %s", (since,))
        return cur.fetchone()["n"]


def claim_next_job(conn: psycopg.Connection) -> dict | None:
    """Atomically take one runnable job. SKIP LOCKED makes this safe for >1 worker."""
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
