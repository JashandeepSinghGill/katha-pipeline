"""correction slice: correct stage, ledger, flags, corrections, settings seeds

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

ENTITY_TYPES = "('person','place','date','book','title','term','other')"
FLAG_KINDS = "('proper_noun','verse','date','number','script','garbled','other')"

# Duplicated from katha.appsettings.DEFAULTS on purpose: migrations must not
# import app code. tests/test_schema.py asserts the two stay equal.
SETTINGS_SEED = {
    "llm_model": "claude-sonnet-5",
    "correct_window_segments": "40",
    "correct_context_segments": "3",
    "ledger_surface_threshold": "88.0",
    "ledger_context_threshold": "0.8",
    "auto_apply_enabled": "true",
}


def upgrade() -> None:
    # Postgres names an inline column CHECK <table>_<column>_check.
    op.execute("ALTER TABLE jobs DROP CONSTRAINT jobs_stage_check")
    op.execute("""
        ALTER TABLE jobs ADD CONSTRAINT jobs_stage_check
        CHECK (stage IN ('prepare','asr','align','assemble','correct','complete'))
    """)

    op.execute(f"""
        CREATE TABLE correction_ledger (
            id                   bigserial PRIMARY KEY,
            canonical_gurmukhi   text NOT NULL UNIQUE,
            romanized            text NOT NULL,
            surface_forms        text[] NOT NULL DEFAULT '{{}}',
            entity_type          text NOT NULL CHECK (entity_type IN {ENTITY_TYPES}),
            english_gloss        text,
            context_embedding    vector,
            embedding_model      text,
            confidence_threshold real NOT NULL,
            times_applied        integer NOT NULL DEFAULT 0,
            retired              boolean NOT NULL DEFAULT false,
            created_by           text NOT NULL DEFAULT 'admin',
            created_at           timestamptz NOT NULL DEFAULT now(),
            last_used_at         timestamptz
        )
    """)

    op.execute(f"""
        CREATE TABLE flags (
            id            bigserial PRIMARY KEY,
            segment_id    bigint NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
            span_text     text NOT NULL,
            kind          text NOT NULL CHECK (kind IN {FLAG_KINDS}),
            reason        text NOT NULL,
            candidates    jsonb NOT NULL,
            ledger_id     bigint REFERENCES correction_ledger(id) ON DELETE SET NULL,
            status        text NOT NULL DEFAULT 'open'
                          CHECK (status IN ('open','accepted','rejected','auto_applied','stale')),
            resolved_text text,
            resolved_at   timestamptz,
            created_at    timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX flags_segment_idx ON flags (segment_id)")
    op.execute("CREATE INDEX flags_status_idx ON flags (status)")

    op.execute("""
        CREATE TABLE corrections (
            id                 bigserial PRIMARY KEY,
            segment_id         bigint NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
            flag_id            bigint REFERENCES flags(id) ON DELETE SET NULL,
            ledger_id          bigint REFERENCES correction_ledger(id) ON DELETE SET NULL,
            source             text NOT NULL CHECK (source IN ('auto','admin')),
            before_text        text NOT NULL,
            after_text         text NOT NULL,
            context_similarity real,
            applied_at         timestamptz NOT NULL DEFAULT now(),
            reverted_at        timestamptz
        )
    """)
    op.execute("CREATE INDEX corrections_segment_idx ON corrections (segment_id)")

    for key, value in SETTINGS_SEED.items():
        op.execute(
            f"INSERT INTO settings (key, value) VALUES ('{key}', '{value}')"
            " ON CONFLICT (key) DO NOTHING"
        )


def downgrade() -> None:
    for table in ("corrections", "flags", "correction_ledger"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    op.execute("DELETE FROM jobs WHERE stage = 'correct'")
    op.execute("ALTER TABLE jobs DROP CONSTRAINT jobs_stage_check")
    op.execute("""
        ALTER TABLE jobs ADD CONSTRAINT jobs_stage_check
        CHECK (stage IN ('prepare','asr','align','assemble','complete'))
    """)
    for key in SETTINGS_SEED:
        op.execute(f"DELETE FROM settings WHERE key = '{key}'")
