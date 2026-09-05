import json
from datetime import datetime
from pathlib import Path

import psycopg

from katha.embed import vector_from_text, vector_to_text
from katha.stages.assemble import _ms, role_for
from katha.stages.ledger import LedgerEntry, merge_embeddings
from katha.stages.types import AssembledSegment

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


def claim_next_job(conn: psycopg.Connection, quota_left: int, backoff_seconds: int) -> dict | None:
    """Atomically claim one queued job. SKIP LOCKED makes this safe for >1 worker.

    Only matches status='queued'. 'failed' is a terminal state reached once
    max_attempts is exhausted, and is not reclaimed here: a non-exhausted
    failure is put back to status='queued' by the caller (the Task 7 worker's
    retry path) and picked up normally on the next claim. A job that is
    truly 'failed' stays that way until an admin explicitly requeues it
    (requeue_job, Task 8), which resets it back to 'queued'.

    The daily throttle gates only the *start* of a new clip, never the
    continuation of one already started: a job with started_at IS NOT NULL
    has already paid its one quota unit (started_at is stamped exactly once
    per clip, via the COALESCE below), so it is always eligible to be
    claimed. A job that has never started (started_at IS NULL) is only
    eligible when quota_left > 0. In-flight jobs are ordered ahead of new
    ones so the pipeline drains what it started rather than fanning out.

    backoff_seconds gates re-claiming of a job that has already failed at
    least once (attempts > 0): it only becomes claimable again once
    updated_at is older than that many seconds, so a brief outage (e.g. a
    one-second Sarvam network blip) can't burn the whole retry budget in a
    tight `while pipeline.run_once(): pass` loop. A job with attempts == 0 —
    including one that just advanced a stage successfully, since the success
    path resets attempts to 0 — is always immediately claimable.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs SET
                status     = 'running',
                attempts   = attempts + 1,
                started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued' AND stage <> 'complete'
                  AND (started_at IS NOT NULL OR %s > 0)
                  AND (attempts = 0 OR updated_at < now() - (%s * interval '1 second'))
                ORDER BY (started_at IS NULL), created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, clip_id, stage, attempts, sarvam_job_id
            """,
            (quota_left, backoff_seconds),
        )
        row = cur.fetchone()
    conn.commit()
    return row


def reclaim_stale_jobs(conn: psycopg.Connection, older_than: datetime) -> int:
    """Requeue jobs stuck in 'running' because the worker died mid-stage
    (OOM, host reboot, a systemd timeout) between claim_next_job's commit
    and the terminating set_job. Without this, such a job matches neither
    claim_next_job (status <> 'queued') nor any other transition, and
    silently vanishes from the pipeline forever.

    Matches status='running' AND updated_at < older_than. Does not reset
    attempts — max_attempts must still bound the total number of tries, or a
    crash-looping clip would retry forever.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = 'queued', updated_at = now()"
            " WHERE status = 'running' AND updated_at < %s",
            (older_than,),
        )
        n = cur.rowcount
    conn.commit()
    return n


def get_clip(conn: psycopg.Connection, clip_id: int) -> dict | None:
    """API-facing accessor: only columns that are safe to ship to a browser
    client. Deliberately excludes source_path (a full server filesystem path)
    and sha256 — see get_clip_internal for the full row, which must never be
    returned directly from an HTTP endpoint."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, duration_seconds, reciter_speaker_id, created_at"
            " FROM clips WHERE id = %s",
            (clip_id,),
        )
        row = cur.fetchone()
    conn.rollback()
    return row


def get_clip_internal(conn: psycopg.Connection, clip_id: int) -> dict | None:
    """Full clip row, including source_path and sha256. For server-side stage
    code only (e.g. worker._prepare, which needs source_path to run ffmpeg) —
    never expose this dict verbatim over the API; use get_clip for that."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM clips WHERE id = %s", (clip_id,))
        row = cur.fetchone()
    conn.rollback()
    return row


def set_clip_meta(
    conn: psycopg.Connection,
    clip_id: int,
    *,
    duration_seconds: float | None = None,
    reciter_speaker_id: str | None = None,
) -> None:
    sets, params = [], []
    for column, value in (
        ("duration_seconds", duration_seconds),
        ("reciter_speaker_id", reciter_speaker_id),
    ):
        if value is not None:
            sets.append(f"{column} = %s")
            params.append(value)
    if not sets:
        return
    params.append(clip_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE clips SET {', '.join(sets)} WHERE id = %s", params)
    conn.commit()


def add_audio_file(
    conn: psycopg.Connection, clip_id: int, kind: str, path: Path, sample_rate: int | None
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO audio_files (clip_id, kind, path, sample_rate)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (clip_id, kind) DO UPDATE SET path = EXCLUDED.path,"
            " sample_rate = EXCLUDED.sample_rate RETURNING id",
            (clip_id, kind, str(path), sample_rate),
        )
        audio_id = cur.fetchone()["id"]
    conn.commit()
    return audio_id


def delete_audio_file(conn: psycopg.Connection, clip_id: int, kind: str) -> None:
    """Drop the row for an intermediate that the worker has unlinked from disk
    (see Pipeline._cleanup_intermediates), so the DB never points at a missing
    file."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM audio_files WHERE clip_id = %s AND kind = %s", (clip_id, kind)
        )
    conn.commit()


def save_segments(
    conn: psycopg.Connection,
    clip_id: int,
    segments: list[AssembledSegment],
    reciter_id: str | None,
) -> None:
    """Replace this clip's segments wholesale, so a retried assemble stage is idempotent."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM segments WHERE clip_id = %s", (clip_id,))  # words cascade
        for seg in segments:
            cur.execute(
                "INSERT INTO segments (clip_id, idx, speaker_id, speaker_role,"
                " start_ms, end_ms, text_pa, align_score, timing_source)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (
                    clip_id, seg.idx, seg.speaker_id,
                    role_for(seg.speaker_id, reciter_id),
                    seg.start_ms, seg.end_ms, seg.text_pa,
                    seg.align_score, seg.timing_source,
                ),
            )
            segment_id = cur.fetchone()["id"]
            for widx, w in enumerate(seg.words):
                cur.execute(
                    "INSERT INTO words (segment_id, idx, text, start_ms, end_ms, score)"
                    " VALUES (%s,%s,%s,%s,%s,%s)",
                    (segment_id, widx, w.text, _ms(w.start_s), _ms(w.end_s), w.score),
                )
        cur.execute(
            "UPDATE clips SET reciter_speaker_id = %s WHERE id = %s", (reciter_id, clip_id)
        )
    conn.commit()


def set_job(
    conn: psycopg.Connection,
    job_id: int,
    *,
    stage: str | None = None,
    status: str | None = None,
    last_error: str | None = None,
    sarvam_job_id: str | None = None,
    attempts: int | None = None,
    clear_error: bool = False,
) -> None:
    """Update only the columns explicitly provided — a parameter left at its
    None default is never written. last_error is the one exception: passing
    last_error=None also leaves it untouched (there would otherwise be no way
    to distinguish "not provided" from "clear it"), so clearing it requires
    the explicit clear_error=True, which emits `last_error = NULL`. This is
    what lets a successful stage transition wipe a stale failure message
    (e.g. a prior "align: CUDA OOM") without disturbing any other caller that
    never passes clear_error and relies on last_error being left alone."""
    sets, params = ["updated_at = now()"], []
    for column, value in (
        ("stage", stage),
        ("status", status),
        ("sarvam_job_id", sarvam_job_id),
        ("attempts", attempts),
    ):
        if value is not None:
            sets.append(f"{column} = %s")
            params.append(value)
    if clear_error:
        sets.append("last_error = NULL")
    elif last_error is not None:
        sets.append("last_error = %s")
        params.append(last_error)
    params.append(job_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s", params)
    conn.commit()


def list_clips(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id, c.title, c.duration_seconds, c.reciter_speaker_id,
                   j.stage, j.status, j.attempts, j.last_error,
                   (SELECT count(*) FROM segments s WHERE s.clip_id = c.id) AS segment_count,
                   (SELECT count(*) FROM segments s
                     WHERE s.clip_id = c.id AND s.timing_source = 'asr') AS loose_count,
                   (SELECT count(*) FROM flags f JOIN segments s ON s.id = f.segment_id
                     WHERE s.clip_id = c.id AND f.status = 'open') AS open_flags,
                   (SELECT count(*) FROM corrections x JOIN segments s ON s.id = x.segment_id
                     WHERE s.clip_id = c.id AND x.source = 'auto' AND x.reverted_at IS NULL)
                     AS auto_applied
            FROM clips c JOIN jobs j ON j.clip_id = c.id
            ORDER BY c.created_at DESC
        """)
        rows = cur.fetchall()
    conn.rollback()
    return rows


def list_segments(conn: psycopg.Connection, clip_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.id, s.idx, s.speaker_id, s.speaker_role, s.start_ms, s.end_ms,
                   s.text_pa, s.align_score, s.timing_source, s.edited,
                   COALESCE(
                       (SELECT json_agg(json_build_object(
                            'text', w.text, 'start_ms', w.start_ms, 'end_ms', w.end_ms
                        ) ORDER BY w.idx)
                        FROM words w WHERE w.segment_id = s.id),
                       '[]'::json
                   ) AS words,
                   COALESCE(
                       (SELECT json_agg(json_build_object(
                            'id', f.id, 'span_text', f.span_text, 'kind', f.kind,
                            'reason', f.reason, 'candidates', f.candidates,
                            'ledger_id', f.ledger_id, 'status', f.status,
                            'resolved_text', f.resolved_text
                        ) ORDER BY f.id)
                        FROM flags f WHERE f.segment_id = s.id),
                       '[]'::json
                   ) AS flags,
                   COALESCE(
                       (SELECT json_agg(json_build_object(
                            'id', x.id, 'flag_id', x.flag_id, 'ledger_id', x.ledger_id,
                            'source', x.source, 'before_text', x.before_text,
                            'after_text', x.after_text,
                            'context_similarity', x.context_similarity,
                            'reverted', x.reverted_at IS NOT NULL
                        ) ORDER BY x.id)
                        FROM corrections x WHERE x.segment_id = s.id),
                       '[]'::json
                   ) AS corrections
            FROM segments s WHERE s.clip_id = %s ORDER BY s.idx
        """, (clip_id,))
        rows = cur.fetchall()
    conn.rollback()
    return rows


def get_audio_file(conn: psycopg.Connection, clip_id: int, kind: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM audio_files WHERE clip_id = %s AND kind = %s", (clip_id, kind)
        )
        row = cur.fetchone()
    conn.rollback()
    return row


def update_segment_text(conn: psycopg.Connection, segment_id: int, text_pa: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE segments SET text_pa = %s, edited = true WHERE id = %s",
            (text_pa, segment_id),
        )
        changed = cur.rowcount == 1
    conn.commit()
    return changed


def set_reciter(conn: psycopg.Connection, clip_id: int, reciter_id: str | None) -> None:
    """Admin override. Re-derives every segment's role from the new reciter id."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE clips SET reciter_speaker_id = %s WHERE id = %s", (reciter_id, clip_id)
        )
        cur.execute("""
            UPDATE segments SET speaker_role = CASE
                WHEN %s::text IS NULL   THEN 'unknown'
                WHEN speaker_id = %s    THEN 'reciter'
                ELSE 'kathakar'
            END
            WHERE clip_id = %s
        """, (reciter_id, reciter_id, clip_id))
    conn.commit()


def requeue_job(conn: psycopg.Connection, clip_id: int) -> bool:
    """Reset a failed job back to queued. No-op (returns False) on any job that
    is not currently 'failed' — in particular 'queued' and 'running' must be
    left completely untouched, since 'failed' is only reached once max_attempts
    is exhausted and this is the sole path that clears it."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = 'queued', attempts = 0, last_error = NULL,"
            " updated_at = now() WHERE clip_id = %s AND status = 'failed'",
            (clip_id,),
        )
        requeued = cur.rowcount == 1
    conn.commit()
    return requeued


# --- settings ---------------------------------------------------------------

def get_setting(conn: psycopg.Connection, key: str, default: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
        row = cur.fetchone()
    conn.rollback()
    return row["value"] if row else default


def get_setting_float(conn: psycopg.Connection, key: str, default: float) -> float:
    try:
        return float(get_setting(conn, key, str(default)))
    except ValueError:
        return default


def get_setting_bool(conn: psycopg.Connection, key: str, default: bool) -> bool:
    raw = get_setting(conn, key, "true" if default else "false").strip().lower()
    return raw in ("true", "1", "yes", "on")


def get_all_settings(conn: psycopg.Connection) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT key, value FROM settings ORDER BY key")
        rows = cur.fetchall()
    conn.rollback()
    return {r["key"]: r["value"] for r in rows}


def set_setting(conn: psycopg.Connection, key: str, value: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (%s, %s)"
            " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )
    conn.commit()


# --- segments (text) --------------------------------------------------------

def list_segment_texts(conn: psycopg.Connection, clip_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, idx, speaker_role, text_pa FROM segments"
            " WHERE clip_id = %s ORDER BY idx",
            (clip_id,),
        )
        rows = cur.fetchall()
    conn.rollback()
    return rows


def get_segment(conn: psycopg.Connection, segment_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, clip_id, idx, speaker_role, text_pa, edited FROM segments WHERE id = %s",
            (segment_id,),
        )
        row = cur.fetchone()
    conn.rollback()
    return row


# --- flags ------------------------------------------------------------------

FlagRow = tuple[int, str, str, str, list[str]]  # segment_id, span_text, kind, reason, candidates


def save_flags(conn: psycopg.Connection, clip_id: int, rows: list[FlagRow]) -> int:
    """Replace this clip's OPEN flags with `rows`. Resolved flags are kept, and
    a row whose (segment, span) already has a non-open flag is skipped, so a
    re-run of the correct stage never duplicates something the admin or the
    ledger has already dealt with."""
    inserted = 0
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM flags WHERE status = 'open' AND segment_id IN"
            " (SELECT id FROM segments WHERE clip_id = %s)",
            (clip_id,),
        )
        for segment_id, span, kind, reason, candidates in rows:
            cur.execute(
                """
                INSERT INTO flags (segment_id, span_text, kind, reason, candidates)
                SELECT %s, %s, %s, %s, %s::jsonb
                WHERE NOT EXISTS (
                    SELECT 1 FROM flags WHERE segment_id = %s AND span_text = %s
                )
                """,
                (segment_id, span, kind, reason, json.dumps(candidates, ensure_ascii=False),
                 segment_id, span),
            )
            inserted += cur.rowcount
    conn.commit()
    return inserted


_FLAG_COLS = (
    "f.id, f.segment_id, f.span_text, f.kind, f.reason, f.candidates, f.ledger_id,"
    " f.status, f.resolved_text, f.resolved_at, f.created_at, s.text_pa, s.clip_id"
)


def list_open_flags(conn: psycopg.Connection, clip_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_FLAG_COLS} FROM flags f JOIN segments s ON s.id = f.segment_id"
            " WHERE s.clip_id = %s AND f.status = 'open' ORDER BY s.idx, f.id",
            (clip_id,),
        )
        rows = cur.fetchall()
    conn.rollback()
    return rows


def get_flag(conn: psycopg.Connection, flag_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_FLAG_COLS} FROM flags f JOIN segments s ON s.id = f.segment_id"
            " WHERE f.id = %s",
            (flag_id,),
        )
        row = cur.fetchone()
    conn.rollback()
    return row


def set_flag_suggestion(
    conn: psycopg.Connection, flag_id: int, ledger_id: int, candidates: list[str]
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE flags SET ledger_id = %s, candidates = %s::jsonb WHERE id = %s",
            (ledger_id, json.dumps(candidates, ensure_ascii=False), flag_id),
        )
    conn.commit()


def resolve_flag(
    conn: psycopg.Connection,
    flag_id: int,
    status: str,
    resolved_text: str | None = None,
    ledger_id: int | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE flags SET status = %s, resolved_text = %s, resolved_at = now(),"
            " ledger_id = COALESCE(%s, ledger_id) WHERE id = %s",
            (status, resolved_text, ledger_id, flag_id),
        )
    conn.commit()


def reopen_flag(conn: psycopg.Connection, flag_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE flags SET status = 'open', resolved_text = NULL, resolved_at = NULL"
            " WHERE id = %s",
            (flag_id,),
        )
    conn.commit()


# --- corrections ------------------------------------------------------------

def _insert_correction(
    cur, *, segment_id, flag_id, ledger_id, source, before, after, similarity
) -> int:
    cur.execute(
        "INSERT INTO corrections (segment_id, flag_id, ledger_id, source, before_text,"
        " after_text, context_similarity) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (segment_id, flag_id, ledger_id, source, before, after, similarity),
    )
    return cur.fetchone()["id"]


def add_correction(
    conn: psycopg.Connection, *, segment_id: int, flag_id: int | None, ledger_id: int | None,
    source: str, before: str, after: str, similarity: float | None,
) -> int:
    with conn.cursor() as cur:
        cid = _insert_correction(
            cur, segment_id=segment_id, flag_id=flag_id, ledger_id=ledger_id,
            source=source, before=before, after=after, similarity=similarity,
        )
    conn.commit()
    return cid


def get_correction(conn: psycopg.Connection, correction_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM corrections WHERE id = %s", (correction_id,))
        row = cur.fetchone()
    conn.rollback()
    return row


def mark_reverted(conn: psycopg.Connection, correction_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE corrections SET reverted_at = now() WHERE id = %s AND reverted_at IS NULL",
            (correction_id,),
        )
    conn.commit()


def apply_auto_correction(
    conn: psycopg.Connection, *, flag_id: int, segment_id: int, ledger_id: int,
    new_text: str, before: str, after: str, similarity: float,
) -> int:
    """Segment rewrite + correction log + flag status + ledger usage, atomically."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE segments SET text_pa = %s, edited = true WHERE id = %s",
            (new_text, segment_id),
        )
        cid = _insert_correction(
            cur, segment_id=segment_id, flag_id=flag_id, ledger_id=ledger_id,
            source="auto", before=before, after=after, similarity=similarity,
        )
        cur.execute(
            "UPDATE flags SET status = 'auto_applied', resolved_text = %s, resolved_at = now(),"
            " ledger_id = %s WHERE id = %s",
            (after, ledger_id, flag_id),
        )
        cur.execute(
            "UPDATE correction_ledger SET times_applied = times_applied + 1,"
            " last_used_at = now() WHERE id = %s",
            (ledger_id,),
        )
    conn.commit()
    return cid


# --- ledger -----------------------------------------------------------------

_LEDGER_COLS = (
    "id, canonical_gurmukhi, romanized, surface_forms, entity_type, english_gloss,"
    " context_embedding::text AS context_embedding, embedding_model, confidence_threshold,"
    " times_applied, retired, created_by, created_at, last_used_at"
)


def ledger_entries(conn: psycopg.Connection) -> list[LedgerEntry]:
    """Every non-retired entry, in the pure matcher's shape."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_LEDGER_COLS} FROM correction_ledger WHERE NOT retired ORDER BY id")
        rows = cur.fetchall()
    conn.rollback()
    return [
        LedgerEntry(
            id=r["id"],
            canonical=r["canonical_gurmukhi"],
            surface_forms=tuple(r["surface_forms"]),
            embedding=(
                tuple(vector_from_text(r["context_embedding"]))
                if r["context_embedding"] else None
            ),
            embedding_model=r["embedding_model"],
            threshold=float(r["confidence_threshold"]),
        )
        for r in rows
    ]


def list_ledger(conn: psycopg.Connection, q: str | None, include_retired: bool) -> list[dict]:
    where = [] if include_retired else ["NOT retired"]
    params: list = []
    if q:
        where.append(
            "(canonical_gurmukhi ILIKE %s OR romanized ILIKE %s"
            " OR EXISTS (SELECT 1 FROM unnest(surface_forms) sf WHERE sf ILIKE %s))"
        )
        like = f"%{q}%"
        params += [like, like, like]
    sql = f"SELECT {_LEDGER_COLS} FROM correction_ledger"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY times_applied DESC, created_at DESC"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.rollback()
    return rows


def get_ledger(conn: psycopg.Connection, ledger_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_LEDGER_COLS} FROM correction_ledger WHERE id = %s", (ledger_id,))
        row = cur.fetchone()
    conn.rollback()
    return row


def upsert_ledger(
    conn: psycopg.Connection, *, canonical: str, romanized: str, surface_form: str | None,
    entity_type: str, english_gloss: str | None, embedding: list[float] | None,
    embedding_model: str | None, threshold: float, created_by: str = "admin",
) -> int:
    """Insert, or teach an existing row: append the surface form, keep the
    existing threshold and gloss unless absent, and merge the context
    embedding (same model) or replace it (different model). A retired entry
    that gets taught again is revived."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_LEDGER_COLS} FROM correction_ledger WHERE canonical_gurmukhi = %s"
            " FOR UPDATE",
            (canonical,),
        )
        row = cur.fetchone()
        if row is None:
            forms = [surface_form] if surface_form and surface_form != canonical else []
            cur.execute(
                "INSERT INTO correction_ledger (canonical_gurmukhi, romanized, surface_forms,"
                " entity_type, english_gloss, context_embedding, embedding_model,"
                " confidence_threshold, created_by)"
                " VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s,%s) RETURNING id",
                (canonical, romanized, forms, entity_type, english_gloss,
                 vector_to_text(embedding) if embedding else None,
                 embedding_model if embedding else None, threshold, created_by),
            )
            ledger_id = cur.fetchone()["id"]
        else:
            forms = list(row["surface_forms"])
            if surface_form and surface_form != canonical and surface_form not in forms:
                forms.append(surface_form)
            old_vec = (
                vector_from_text(row["context_embedding"]) if row["context_embedding"] else None
            )
            if embedding is None:
                new_vec, new_model = old_vec, row["embedding_model"]
            elif old_vec is not None and row["embedding_model"] == embedding_model:
                new_vec, new_model = merge_embeddings(old_vec, embedding), embedding_model
            else:
                new_vec, new_model = list(embedding), embedding_model
            cur.execute(
                "UPDATE correction_ledger SET surface_forms = %s, romanized = %s,"
                " english_gloss = COALESCE(english_gloss, %s), context_embedding = %s::vector,"
                " embedding_model = %s, retired = false WHERE id = %s",
                (forms, romanized, english_gloss,
                 vector_to_text(new_vec) if new_vec else None, new_model, row["id"]),
            )
            ledger_id = row["id"]
    conn.commit()
    return ledger_id


_LEDGER_EDITABLE = {
    "canonical_gurmukhi", "romanized", "surface_forms", "entity_type",
    "english_gloss", "confidence_threshold", "retired",
}


def update_ledger(conn: psycopg.Connection, ledger_id: int, **fields) -> bool:
    sets, params = [], []
    for column, value in fields.items():
        if column not in _LEDGER_EDITABLE:
            raise ValueError(f"not an editable ledger column: {column}")
        sets.append(f"{column} = %s")
        params.append(value)
    if not sets:
        return False
    params.append(ledger_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE correction_ledger SET {', '.join(sets)} WHERE id = %s", params)
        changed = cur.rowcount == 1
    conn.commit()
    return changed


def bump_ledger_use(conn: psycopg.Connection, ledger_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE correction_ledger SET times_applied = times_applied + 1,"
            " last_used_at = now() WHERE id = %s",
            (ledger_id,),
        )
    conn.commit()


def top_gazetteer(conn: psycopg.Connection, limit: int = 200) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT canonical_gurmukhi, entity_type FROM correction_ledger WHERE NOT retired"
            " ORDER BY times_applied DESC, created_at LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    conn.rollback()
    return [(r["canonical_gurmukhi"], r["entity_type"]) for r in rows]
