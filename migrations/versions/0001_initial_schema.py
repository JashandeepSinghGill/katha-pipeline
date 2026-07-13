"""initial schema

Revision ID: 0001
Revises:
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enabled now so the retrieval plan needs no schema surgery. Unused in this slice.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE clips (
            id                 bigserial PRIMARY KEY,
            source_path        text NOT NULL UNIQUE,
            title              text NOT NULL,
            sha256             text NOT NULL,
            duration_seconds   double precision,
            reciter_speaker_id text,
            created_at         timestamptz NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE audio_files (
            id          bigserial PRIMARY KEY,
            clip_id     bigint NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
            kind        text NOT NULL CHECK (kind IN ('original','native','denoised','asr_ready')),
            path        text NOT NULL,
            sample_rate integer,
            created_at  timestamptz NOT NULL DEFAULT now(),
            UNIQUE (clip_id, kind)
        )
    """)

    op.execute("""
        CREATE TABLE jobs (
            id            bigserial PRIMARY KEY,
            clip_id       bigint NOT NULL UNIQUE REFERENCES clips(id) ON DELETE CASCADE,
            stage         text NOT NULL DEFAULT 'prepare'
                          CHECK (stage IN ('prepare','asr','align','assemble','complete')),
            status        text NOT NULL DEFAULT 'queued'
                          CHECK (status IN ('queued','running','failed','done')),
            attempts      integer NOT NULL DEFAULT 0,
            last_error    text,
            sarvam_job_id text,
            started_at    timestamptz,
            updated_at    timestamptz NOT NULL DEFAULT now(),
            created_at    timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX jobs_status_stage_idx ON jobs (status, stage)")

    op.execute("""
        CREATE TABLE segments (
            id            bigserial PRIMARY KEY,
            clip_id       bigint NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
            idx           integer NOT NULL,
            speaker_id    text,
            speaker_role  text CHECK (speaker_role IN ('reciter','kathakar','unknown')),
            start_ms      integer NOT NULL,
            end_ms        integer NOT NULL,
            text_pa       text NOT NULL,
            align_score   double precision,
            timing_source text NOT NULL DEFAULT 'aligned'
                          CHECK (timing_source IN ('aligned','asr')),
            edited        boolean NOT NULL DEFAULT false,
            created_at    timestamptz NOT NULL DEFAULT now(),
            UNIQUE (clip_id, idx)
        )
    """)
    op.execute("CREATE INDEX segments_clip_start_idx ON segments (clip_id, start_ms)")

    op.execute("""
        CREATE TABLE words (
            id         bigserial PRIMARY KEY,
            segment_id bigint NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
            idx        integer NOT NULL,
            text       text NOT NULL,
            start_ms   integer NOT NULL,
            end_ms     integer NOT NULL,
            score      double precision,
            UNIQUE (segment_id, idx)
        )
    """)

    op.execute("""
        CREATE TABLE settings (
            key   text PRIMARY KEY,
            value text NOT NULL
        )
    """)
    op.execute("INSERT INTO settings (key, value) VALUES ('files_per_day', '3')")


def downgrade() -> None:
    for table in ("words", "segments", "jobs", "audio_files", "clips", "settings"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
