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
