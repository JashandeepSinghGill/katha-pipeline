import psycopg
import pytest


@pytest.mark.db
def test_every_table_exists(db):
    with db.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        names = {r["table_name"] for r in cur.fetchall()}
    assert {"clips", "audio_files", "jobs", "segments", "words", "settings"} <= names


@pytest.mark.db
def test_segment_rejects_unknown_speaker_role(db):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO clips (source_path, title, sha256) VALUES (%s,%s,%s) RETURNING id",
            ("/a.mp3", "a", "x"),
        )
        clip_id = cur.fetchone()["id"]
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO segments (clip_id, idx, speaker_role, start_ms, end_ms, text_pa)"
                " VALUES (%s,0,'narrator',0,10,'x')",
                (clip_id,),
            )


@pytest.mark.db
def test_jobs_stage_accepts_correct(db):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO clips (source_path, title, sha256) VALUES ('/x', 'x', 'h') RETURNING id"
        )
        clip_id = cur.fetchone()["id"]
        cur.execute("INSERT INTO jobs (clip_id, stage) VALUES (%s, 'correct')", (clip_id,))
    db.commit()


@pytest.mark.db
def test_correction_tables_exist_with_seeded_settings(db):
    from katha.appsettings import DEFAULTS

    with db.cursor() as cur:
        cur.execute("SELECT key, value FROM settings")
        rows = {r["key"]: r["value"] for r in cur.fetchall()}
        for table in ("correction_ledger", "flags", "corrections"):
            cur.execute("SELECT to_regclass(%s) AS t", (table,))
            assert cur.fetchone()["t"] == table
    db.rollback()
    assert rows == DEFAULTS
