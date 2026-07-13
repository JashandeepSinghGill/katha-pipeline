# Katha Pipeline — Reviewable Transcript Slice: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop a katha audio file into a watch folder and, with no further intervention, get a speaker-labelled Punjabi transcript with word-accurate timestamps that an admin can review in a browser while the exact audio span plays back.

**Architecture:** A Python stage-machine worker walks each clip through `prepare → asr → align → assemble`, persisting to Postgres after each stage so any stage can be retried independently. Audio prep (ffmpeg + DeepFilterNet) and forced alignment (ctc-forced-aligner) run locally on the Proxmox host; only ASR leaves the box (Sarvam Batch API). A FastAPI service exposes the data plus an HTTP range endpoint for audio seeking; a Next.js admin UI consumes it. All pipeline stages are pure functions over dataclasses wherever possible, with I/O pushed to the edges, so the logic is testable without a GPU or an API key.

**Tech Stack:** Python 3.11, psycopg 3, Alembic, FastAPI, pydantic-settings, `sarvamai` SDK, DeepFilterNet 0.5.6 (isolated venv, invoked as a subprocess), ctc-forced-aligner (MMS), ffmpeg; Postgres 17 + pgvector 0.8.5 via Docker; Next.js (App Router) + `gurmukhi-utils@3.2.2`.

---

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec or from verified vendor documentation — **do not substitute from memory.**

- **Sarvam model ID is `saaras:v3`.** The `sarvamai` SDK's `create_job()` still defaults to the legacy `saarika:v2.5` and sends it *explicitly*, so the server-side default never applies. **Always pass `model="saaras:v3"`.**
- **Sarvam Batch API only.** The sync `/speech-to-text` endpoint caps at **30 seconds**; clips are 30–90 min. Batch caps at **2 hours per file** — clips fit natively, no chunking. Diarization and timestamps exist **only** on the Batch API.
- **Sarvam returns chunk-level (sentence) timestamps, never word-level.** The response key is named `words` but its elements are whole sentences. Word timing comes *only* from forced alignment.
- **`job.wait_until_complete()` defaults to `timeout=600` (10 min)** and will raise `TimeoutError` on a 90-minute clip. Always pass an explicit long timeout.
- **Punjabi language code differs per tool:** Sarvam uses **`pa-IN`**; ctc-forced-aligner uses ISO-639-3 **`pan`** (not `pa`). Getting these backwards silently degrades output.
- **ctc-forced-aligner requires `romanize=True`.** Its model vocab (`MahmoudAshraf/mms-300m-1130-forced-aligner`) is 26 Latin letters + apostrophe with **zero Gurmukhi codepoints**; without romanization every character maps to `<unk>` and alignment is garbage. Romanization is internal — the returned word text is your original Gurmukhi.
- **WhisperX must not be used for alignment.** Punjabi (`pa`) is absent from both `DEFAULT_ALIGN_MODELS_TORCH` and `DEFAULT_ALIGN_MODELS_HF`; it raises `ValueError` rather than falling back.
- **Denoiser is DeepFilterNet, invoked as a subprocess from its own venv.** It is predictive (mask-based) and cannot hallucinate speech. Do **not** use `resemble-enhance`'s generative `enhance()` path — it re-synthesizes the waveform and would corrupt both transcript and timestamps.
- **pgvector must be ≥ 0.8.2** (CVE-2026-3172, parallel HNSW build buffer overflow). Pin the image tag `pgvector/pgvector:0.8.5-pg17-trixie`; the floating `pg17` tag silently moves versions.
- **Speaker role is not derivable from diarization alone.** Verses appear inside kathakar segments (verified in `punjabi-transcript-PART-1.txt`: Speaker 2 recites `ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ ॥` at 2:25, and Speaker 1 re-quotes the same line inside commentary at 3:13). Diarization is a *prior*, not a classifier. Roles are inferred by a heuristic and **must be admin-overridable per clip.**
- **Throttle default is 3 files/day** (spec §14 baseline). Stored in the `settings` table, not hardcoded.
- **All timestamps are stored as integer milliseconds** in the database. Sarvam and the aligner both emit float seconds; convert at the boundary, never store floats.

### Locked architectural decisions (for follow-on plans, not built here)

- **All LLM calls go behind an `LLMProvider` interface** with two implementations (`claude -p` subprocess, Anthropic API), selected by config. Nothing else in the codebase learns which backend is live. This slice contains no LLM stage, so the interface is not built yet — but the correction plan must open with it.

### Out of scope for this plan

Correction ledger and auto-apply (spec §6), facts table (§7), Nanak Prakash verse DB (§3.1), BaniDB/GGS quote detection (§3.2), English translation (§4.9), embeddings and pgvector retrieval (§9), RAG bot (§4.11), `audit_log` table. The `vector` extension is enabled in the baseline migration so the retrieval plan needs no schema surgery.

---

## Prerequisites (human, before Task 1)

These are manual and block the plan. Do them first.

1. **Sarvam account + API key.** Sign up at sarvam.ai, create a key. New accounts get ₹100 free credit — enough for ~2 hours of diarized audio. Export as `KATHA_SARVAM_API_KEY`.
2. **Decide on the ctc-forced-aligner licence.** It and its model are **CC-BY-NC 4.0 (non-commercial)**. If this project is personal/parchar, proceed. If it will ever be commercial, stop and swap the aligner for NeMo Forced Aligner + `ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large` (heavier install, aligns native Gurmukhi with no romanization, likely more accurate). That swap only touches Task 5.
3. **Confirm GPU on Zira.** `nvidia-smi`. Alignment of a 90-min clip on CPU is slow but works; set `KATHA_DEVICE=cpu` if there is no GPU. DeepFilterNet is real-time on CPU either way.
4. **`ffmpeg` and `ffprobe` on PATH.** `sudo apt install ffmpeg`. Both are hard dependencies — ctc-forced-aligner's `load_audio()` shells out to ffmpeg.
5. **Pick the watch folder and work folder** on the existing media LV, e.g. `/srv/media/katha/incoming` and `/srv/media/katha/work`.

---

## File Structure

| File | Responsibility |
|---|---|
| `docker-compose.yml` | Postgres 17 + pgvector, bound to localhost |
| `.env.example` | Every `KATHA_*` variable, documented |
| `pyproject.toml` | Python deps + pytest config |
| `alembic.ini`, `migrations/env.py`, `migrations/versions/0001_initial_schema.py` | Schema versioning; raw SQL, no ORM |
| `src/katha/config.py` | `Settings` — all config from env, one source of truth |
| `src/katha/db.py` | psycopg 3 connection factory |
| `src/katha/stages/types.py` | `DiarizedEntry`, `AlignedWord`, `AssembledSegment` — shared dataclasses, zero imports from stages (prevents cycles) |
| `src/katha/stages/prepare.py` | ffmpeg/DeepFilterNet command builders (pure) + `prepare_clip` (I/O) |
| `src/katha/stages/asr.py` | Sarvam Batch client + `parse_diarized` (pure) |
| `src/katha/stages/align.py` | ctc-forced-aligner wrapper + `to_aligned_words` (pure) |
| `src/katha/stages/assemble.py` | `alignment_text`, `assemble_segments`, `infer_reciter_speaker_id` — all pure |
| `src/katha/repo.py` | Every SQL statement. Stages never touch the DB. |
| `src/katha/ingest/watcher.py` | Scan watch dir, register new clips, enforce daily throttle |
| `src/katha/worker.py` | Stage machine: dispatch, retry, resume |
| `src/katha/api/ranges.py` | HTTP Range header parsing (pure) |
| `src/katha/api/main.py` | FastAPI app: clips, segments, audio range endpoint |
| `web/` | Next.js admin: dashboard + review view |
| `tests/` | Mirrors `src/katha/` |

---

## Task 1: Scaffolding, config, database, schema

**Files:**
- Create: `pyproject.toml`, `docker-compose.yml`, `.env.example`, `alembic.ini`
- Create: `src/katha/__init__.py`, `src/katha/config.py`, `src/katha/db.py`
- Create: `migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/0001_initial_schema.py`
- Create: `tests/conftest.py`, `tests/test_config.py`, `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `katha.config.Settings` (pydantic-settings, env prefix `KATHA_`) and `get_settings() -> Settings`.
  - `katha.db.connect(settings: Settings) -> psycopg.Connection` — autocommit off, `row_factory=dict_row`.
  - Tables: `clips`, `audio_files`, `jobs`, `segments`, `words`, `settings`.

- [ ] **Step 1: Create the Python project definition**

`pyproject.toml`:

```toml
[project]
name = "katha"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic-settings>=2.0",
    "psycopg[binary]>=3.3",
    "pgvector>=0.5.0",
    "alembic>=1.18",
    "sqlalchemy>=2.0",
    "sarvamai>=0.1.28",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
]

[project.optional-dependencies]
align = [
    "torch>=2.2",
    "ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git",
]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "numpy>=1.26",
    "soundfile>=0.12",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = [
    "db: requires a live Postgres (docker compose up -d db)",
    "heavy: requires torch + the alignment model downloaded",
]
```

`align` is an extra because torch is a large install and the pure-logic tests must run without it. DeepFilterNet is deliberately **absent** — it gets its own venv (Step 3) to keep its torch pin from fighting the aligner's.

- [ ] **Step 2: Create the database service**

`docker-compose.yml`:

```yaml
services:
  db:
    image: pgvector/pgvector:0.8.5-pg17-trixie
    container_name: katha-db
    restart: unless-stopped
    environment:
      POSTGRES_DB: katha
      POSTGRES_USER: katha
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD in .env}
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U katha -d katha"]
      interval: 10s
      timeout: 5s
      retries: 5
    command: >
      postgres
      -c shared_buffers=1GB
      -c maintenance_work_mem=2GB

volumes:
  pgdata:
```

`.env.example`:

```bash
# Database
POSTGRES_PASSWORD=change-me
KATHA_DATABASE_URL=postgresql://katha:change-me@localhost:5432/katha

# Sarvam (https://sarvam.ai — see Prerequisites)
KATHA_SARVAM_API_KEY=

# Paths on the media LV
KATHA_WATCH_DIR=/srv/media/katha/incoming
KATHA_WORK_DIR=/srv/media/katha/work

# Compute
KATHA_DEVICE=cuda                                  # 'cpu' if Zira has no GPU
KATHA_DENOISE_BIN=/opt/venvs/deepfilternet/bin/deepFilter
```

- [ ] **Step 3: Install DeepFilterNet into its own venv**

DeepFilterNet 0.5.6 and ctc-forced-aligner both pin torch, and their pins conflict in practice. Isolating the denoiser and calling its binary means the pipeline never imports it.

```bash
python3.11 -m venv /opt/venvs/deepfilternet
/opt/venvs/deepfilternet/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
/opt/venvs/deepfilternet/bin/pip install deepfilternet==0.5.6
/opt/venvs/deepfilternet/bin/deepFilter --help
```

Expected: usage text listing `--output-dir`, `--model-base-dir`, `--pf`. Confirms the binary path used in `.env`.

- [ ] **Step 4: Write the failing config test**

`tests/test_config.py`:

```python
from pathlib import Path
from katha.config import Settings


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("KATHA_SARVAM_API_KEY", "sk-test")
    monkeypatch.setenv("KATHA_WATCH_DIR", "/tmp/in")
    s = Settings(_env_file=None)
    assert s.sarvam_api_key == "sk-test"
    assert s.watch_dir == Path("/tmp/in")


def test_sarvam_defaults_are_the_verified_values():
    s = Settings(_env_file=None)
    assert s.sarvam_model == "saaras:v3"      # SDK default is the legacy saarika:v2.5
    assert s.sarvam_language == "pa-IN"       # Sarvam uses ISO-639-1 + region
    assert s.align_language == "pan"          # aligner uses ISO-639-3
    assert s.sarvam_timeout_seconds >= 7200   # wait_until_complete default of 600s is too short
```

- [ ] **Step 5: Run it to confirm it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'katha.config'`

- [ ] **Step 6: Write the config module**

`src/katha/config.py`:

```python
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KATHA_", env_file=".env", extra="ignore"
    )

    database_url: str = "postgresql://katha:katha@localhost:5432/katha"

    watch_dir: Path = Path("/srv/media/katha/incoming")
    work_dir: Path = Path("/srv/media/katha/work")

    sarvam_api_key: str = ""
    sarvam_model: str = "saaras:v3"
    sarvam_mode: str = "transcribe"
    sarvam_language: str = "pa-IN"
    sarvam_timeout_seconds: int = 7200
    sarvam_poll_seconds: int = 15
    num_speakers: int = 2

    align_language: str = "pan"
    align_batch_size: int = 8

    device: str = "cuda"
    denoise_enabled: bool = True
    denoise_bin: str = "deepFilter"
    denoise_model: str = "DeepFilterNet3"

    max_audio_seconds: int = 7200
    max_attempts: int = 3


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 7: Run the config test to confirm it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 8: Write the connection helper**

`src/katha/db.py`:

```python
import psycopg
from psycopg.rows import dict_row

from katha.config import Settings


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.database_url, row_factory=dict_row)
```

- [ ] **Step 9: Initialise Alembic and write the baseline migration**

```bash
alembic init -t generic migrations
```

Leave `sqlalchemy.url` blank in `alembic.ini` so no credential is ever committed:

```ini
sqlalchemy.url =
```

Then replace the generated `migrations/env.py` wholesale — the URL comes from the environment, and there is no ORM metadata to autogenerate against:

```python
import os

from alembic import context
from sqlalchemy import create_engine

from katha.config import get_settings


def _url() -> str:
    return os.environ.get("KATHA_DATABASE_URL") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=_url(), literal_binds=True, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Reading `KATHA_DATABASE_URL` from the environment first is what lets `tests/conftest.py` point Alembic at the test database without touching any file.

`migrations/versions/0001_initial_schema.py`:

```python
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
```

- [ ] **Step 10: Write the test fixtures**

`tests/conftest.py`:

```python
import os
import subprocess

import pytest

from katha.config import Settings
from katha.db import connect

TEST_DB_URL = os.environ.get(
    "KATHA_TEST_DATABASE_URL",
    "postgresql://katha:change-me@localhost:5432/katha_test",
)
TABLES = ("words", "segments", "jobs", "audio_files", "clips")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None, database_url=TEST_DB_URL)


@pytest.fixture(scope="session", autouse=True)
def _migrate(settings):
    env = {**os.environ, "KATHA_DATABASE_URL": settings.database_url}
    subprocess.run(["alembic", "upgrade", "head"], env=env, check=True)


@pytest.fixture
def db(settings):
    conn = connect(settings)
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    conn.commit()
    conn.close()
```

`tests/test_schema.py`:

```python
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
        with pytest.raises(Exception):
            cur.execute(
                "INSERT INTO segments (clip_id, idx, speaker_role, start_ms, end_ms, text_pa)"
                " VALUES (%s,0,'narrator',0,10,'x')",
                (clip_id,),
            )
```

- [ ] **Step 11: Bring up the database and run the schema tests**

```bash
cp .env.example .env      # then fill in POSTGRES_PASSWORD and KATHA_SARVAM_API_KEY
docker compose up -d db
docker compose exec db createdb -U katha katha_test
pytest tests/ -v
```

Expected: PASS (4 tests — 2 config, 2 schema)

- [ ] **Step 12: Commit**

```bash
git init
git add .
git commit -m "feat: project scaffolding, config, and baseline schema"
```

---

## Task 2: Clip registry and ingest watcher

**Files:**
- Create: `src/katha/repo.py`, `src/katha/ingest/__init__.py`, `src/katha/ingest/watcher.py`
- Create: `tests/test_repo.py`, `tests/test_watcher.py`

**Interfaces:**
- Consumes: `katha.config.Settings`, `katha.db.connect`.
- Produces:
  - `katha.repo.register_clip(conn, source_path: Path, title: str, sha256: str) -> int | None` — returns new clip id, or `None` if `source_path` is already registered.
  - `katha.repo.get_setting_int(conn, key: str, default: int) -> int`
  - `katha.repo.count_jobs_started_since(conn, since: datetime) -> int`
  - `katha.repo.claim_next_job(conn) -> dict | None` — atomically marks one queued/failed job `running`, returns `{"id", "clip_id", "stage", "attempts", "sarvam_job_id"}`.
  - `katha.repo.set_job(conn, job_id: int, *, stage=None, status=None, last_error=None, sarvam_job_id=None) -> None`
  - `katha.ingest.watcher.sha256_file(path: Path) -> str`
  - `katha.ingest.watcher.discover(watch_dir: Path) -> list[Path]` — pure, sorted, audio extensions only.
  - `katha.ingest.watcher.scan_once(conn, settings) -> list[int]` — registers up to the remaining daily quota, returns new clip ids.

- [ ] **Step 1: Write the failing repo tests**

`tests/test_repo.py`:

```python
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
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_repo.py -v`
Expected: FAIL — `ImportError: cannot import name 'repo' from 'katha'`

- [ ] **Step 3: Write the repo module**

`src/katha/repo.py`:

```python
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
```

- [ ] **Step 4: Run the repo tests**

Run: `pytest tests/test_repo.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the failing watcher tests**

`tests/test_watcher.py`:

```python
from pathlib import Path

import pytest

from katha.ingest import watcher


def test_discover_finds_audio_and_ignores_everything_else(tmp_path):
    (tmp_path / "b.mp3").write_bytes(b"x")
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.mp3").write_bytes(b"x")
    assert watcher.discover(tmp_path) == [tmp_path / "a.wav", tmp_path / "b.mp3"]


def test_sha256_file_is_stable(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"hello")
    assert watcher.sha256_file(f) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


@pytest.mark.db
def test_scan_once_respects_the_daily_throttle(db, settings, tmp_path, monkeypatch):
    for i in range(5):
        (tmp_path / f"k{i}.wav").write_bytes(b"x")
    monkeypatch.setattr(settings, "watch_dir", tmp_path)
    with db.cursor() as cur:
        cur.execute("UPDATE settings SET value = '2' WHERE key = 'files_per_day'")
    db.commit()

    assert len(watcher.scan_once(db, settings)) == 2
    assert len(watcher.scan_once(db, settings)) == 2  # registration alone doesn't consume quota
```

The second `scan_once` still returns 2 because quota is spent when a job *starts*, not when a clip is registered — registering ahead of the worker is harmless and keeps the queue warm. The throttle that matters is enforced in Task 7.

- [ ] **Step 6: Run to confirm it fails**

Run: `pytest tests/test_watcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'katha.ingest'`

- [ ] **Step 7: Write the watcher**

`src/katha/ingest/__init__.py`: (empty file)

`src/katha/ingest/watcher.py`:

```python
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
    """Register new clips, newest quota first. Idempotent — re-running is a no-op."""
    quota = repo.get_setting_int(conn, "files_per_day", 3)
    registered: list[int] = []
    for path in discover(settings.watch_dir):
        if len(registered) >= quota:
            break
        clip_id = repo.register_clip(conn, path, path.stem, sha256_file(path))
        if clip_id is not None:
            registered.append(clip_id)
    return registered
```

- [ ] **Step 8: Run the watcher tests**

Run: `pytest tests/test_watcher.py -v`
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add src/katha/repo.py src/katha/ingest tests/test_repo.py tests/test_watcher.py
git commit -m "feat: clip registry and ingest watcher with daily throttle"
```

---

## Task 3: Audio preparation — transcode, denoise, ASR-ready wav

**Files:**
- Create: `src/katha/stages/__init__.py`, `src/katha/stages/prepare.py`
- Create: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `katha.config.Settings`.
- Produces:
  - `katha.stages.prepare.PrepareError` (exception).
  - `katha.stages.prepare.to_mono_wav_cmd(src: Path, dst: Path, sample_rate: int | None) -> list[str]`
  - `katha.stages.prepare.denoise_cmd(src: Path, out_dir: Path, bin_path: str, model: str) -> list[str]`
  - `katha.stages.prepare.denoised_path(src: Path, out_dir: Path, model: str) -> Path`
  - `katha.stages.prepare.probe_cmd(src: Path) -> list[str]` and `katha.stages.prepare.probe(src: Path) -> tuple[float, int]` returning `(duration_seconds, sample_rate)`.
  - `katha.stages.prepare.prepare_clip(source: Path, work_dir: Path, settings) -> PrepareResult` where `PrepareResult` is a frozen dataclass with `native: Path`, `denoised: Path | None`, `asr_ready: Path`, `duration_seconds: float`, `native_sample_rate: int`.

**Why subprocess, not the Python API:** DeepFilterNet's `enhance()` does **not** resample (its docstring: *"Sampling rate needs to match to `model` and `df_state`"*), and `init_df()` returns a 3-tuple on PyPI 0.5.6 but a 4-tuple on git main — an unpack that breaks on upgrade. The `deepFilter` CLI resamples correctly (16k → 48k → denoise → 16k) and its contract is stable. Shelling out also lets the denoiser live in a separate venv.

- [ ] **Step 1: Write the failing command-builder tests**

`tests/test_prepare.py`:

```python
from pathlib import Path

import pytest

from katha.stages import prepare


def test_denoised_path_matches_deepfilternet_naming():
    # DeepFilterNet writes <stem>_<model>.wav into --output-dir
    assert prepare.denoised_path(
        Path("/w/1/native.wav"), Path("/w/1"), "DeepFilterNet3"
    ) == Path("/w/1/native_DeepFilterNet3.wav")


def test_to_mono_wav_cmd_pins_16k_pcm_for_asr():
    cmd = prepare.to_mono_wav_cmd(Path("/a/in.wav"), Path("/a/asr.wav"), 16000)
    assert cmd[:2] == ["ffmpeg", "-y"]
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[-1] == "/a/asr.wav"


def test_to_mono_wav_cmd_omits_rate_when_preserving_native():
    cmd = prepare.to_mono_wav_cmd(Path("/a/in.mp3"), Path("/a/native.wav"), None)
    assert "-ar" not in cmd
    assert cmd[cmd.index("-ac") + 1] == "1"
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_prepare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'katha.stages'`

- [ ] **Step 3: Write the prepare module**

`src/katha/stages/__init__.py`: (empty file)

`src/katha/stages/prepare.py`:

```python
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from katha.config import Settings


class PrepareError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrepareResult:
    native: Path
    denoised: Path | None
    asr_ready: Path
    duration_seconds: float
    native_sample_rate: int


def to_mono_wav_cmd(src: Path, dst: Path, sample_rate: int | None) -> list[str]:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1"]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    return cmd + ["-c:a", "pcm_s16le", str(dst)]


def denoise_cmd(src: Path, out_dir: Path, bin_path: str, model: str) -> list[str]:
    return [bin_path, str(src), "-o", str(out_dir), "-m", model]


def denoised_path(src: Path, out_dir: Path, model: str) -> Path:
    return out_dir / f"{src.stem}_{model}.wav"


def probe_cmd(src: Path) -> list[str]:
    return [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "format=duration:stream=sample_rate",
        "-of", "json", str(src),
    ]


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PrepareError(f"{cmd[0]} failed: {result.stderr.strip()[:500]}")
    return result.stdout


def probe(src: Path) -> tuple[float, int]:
    payload = json.loads(_run(probe_cmd(src)))
    duration = float(payload["format"]["duration"])
    sample_rate = int(payload["streams"][0]["sample_rate"])
    return duration, sample_rate


def prepare_clip(source: Path, work_dir: Path, settings: Settings) -> PrepareResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    duration, sample_rate = probe(source)
    if duration > settings.max_audio_seconds:
        raise PrepareError(
            f"{source.name} is {duration / 3600:.2f}h; Sarvam's Batch API caps at "
            f"{settings.max_audio_seconds / 3600:.0f}h per file. Split it first."
        )

    # Keep the native rate: denoising at 48k then downsampling beats the reverse,
    # because upsampling invents no bandwidth for the model to work on.
    native = work_dir / "native.wav"
    _run(to_mono_wav_cmd(source, native, None))

    denoised: Path | None = None
    if settings.denoise_enabled:
        _run(denoise_cmd(native, work_dir, settings.denoise_bin, settings.denoise_model))
        denoised = denoised_path(native, work_dir, settings.denoise_model)
        if not denoised.exists():
            raise PrepareError(f"denoiser produced no output at {denoised}")

    asr_ready = work_dir / "asr.wav"
    _run(to_mono_wav_cmd(denoised or native, asr_ready, 16000))

    return PrepareResult(
        native=native,
        denoised=denoised,
        asr_ready=asr_ready,
        duration_seconds=duration,
        native_sample_rate=sample_rate,
    )
```

- [ ] **Step 4: Run the command-builder tests**

Run: `pytest tests/test_prepare.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Add an end-to-end prepare test against real ffmpeg**

Append to `tests/test_prepare.py`:

```python
import numpy as np
import soundfile as sf


@pytest.fixture
def tone_wav(tmp_path):
    path = tmp_path / "tone.wav"
    t = np.linspace(0, 2.0, 48000 * 2, endpoint=False)
    sf.write(path, 0.2 * np.sin(2 * np.pi * 220 * t), 48000)
    return path


def test_prepare_clip_produces_16k_mono_asr_wav(tone_wav, tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "denoise_enabled", False)  # denoiser covered separately
    result = prepare.prepare_clip(tone_wav, tmp_path / "work", settings)

    assert result.asr_ready.exists()
    assert result.native_sample_rate == 48000
    assert 1.9 < result.duration_seconds < 2.1

    data, rate = sf.read(result.asr_ready)
    assert rate == 16000
    assert data.ndim == 1  # mono


def test_prepare_clip_rejects_audio_longer_than_the_batch_cap(tone_wav, tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "max_audio_seconds", 1)
    with pytest.raises(prepare.PrepareError, match="caps at"):
        prepare.prepare_clip(tone_wav, tmp_path / "work", settings)
```

- [ ] **Step 6: Run the full prepare suite**

Run: `pytest tests/test_prepare.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Verify the denoiser on a real katha file by hand**

Pick one real clip. Run the pipeline with denoise on and off, and listen to both:

```bash
python - <<'PY'
from pathlib import Path
from katha.config import get_settings
from katha.stages import prepare

s = get_settings()
r = prepare.prepare_clip(Path("/srv/media/katha/incoming/sample.mp3"), Path("/tmp/dn-on"), s)
print("denoised:", r.denoised, "asr:", r.asr_ready)
PY
```

Expected: `/tmp/dn-on/native_DeepFilterNet3.wav` exists and `/tmp/dn-on/asr.wav` is 16 kHz mono. Listen to both `native.wav` and the denoised output. Spec §4 stage 2 calls denoise a per-file toggle to be A/B tested — if DeepFilterNet dulls the sung recitation, set `KATHA_DENOISE_ENABLED=false` and note it. This is a judgement call that must be made on real audio, not asserted in a test.

- [ ] **Step 8: Commit**

```bash
git add src/katha/stages tests/test_prepare.py
git commit -m "feat: audio preparation — transcode, denoise, 16k asr wav"
```

---

## Task 4: Sarvam Batch ASR client

**Files:**
- Create: `src/katha/stages/types.py`, `src/katha/stages/asr.py`
- Create: `tests/test_asr.py`

**Interfaces:**
- Consumes: `katha.config.Settings`.
- Produces:
  - `katha.stages.types.DiarizedEntry` — frozen dataclass: `speaker_id: str`, `text: str`, `start_s: float`, `end_s: float`.
  - `katha.stages.asr.AsrError` (exception).
  - `katha.stages.asr.parse_diarized(payload: dict) -> list[DiarizedEntry]` — pure.
  - `katha.stages.asr.SarvamAsr(settings, client=None)` with `.transcribe(wav: Path, out_dir: Path) -> tuple[list[DiarizedEntry], str]` returning entries and the Sarvam job id.

- [ ] **Step 1: Create the shared dataclasses**

`src/katha/stages/types.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiarizedEntry:
    """One Sarvam diarized chunk. Sentence-level, NOT word-level."""
    speaker_id: str
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class AlignedWord:
    """One word from forced alignment. The only source of word-level timing."""
    text: str
    start_s: float
    end_s: float
    score: float


@dataclass(frozen=True)
class AssembledSegment:
    idx: int
    speaker_id: str
    text_pa: str
    start_ms: int
    end_ms: int
    align_score: float | None
    timing_source: str  # 'aligned' | 'asr'
    words: list[AlignedWord] = field(default_factory=list)
```

- [ ] **Step 2: Write the failing parser tests**

`tests/test_asr.py`:

```python
import pytest

from katha.stages import asr

# Response shape verified against Sarvam's OpenAPI schema
# (Sarvam_Model_API_SpeechToTextResponse).
PAYLOAD = {
    "request_id": "req-1",
    "transcript": "ਕਵਿਤ। ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ ॥ ਐ ਪਾਤਸ਼ਾਹ ਜੀ।",
    "language_code": "pa-IN",
    "diarized_transcript": {
        "entries": [
            {"transcript": "ਕਵਿਤ।", "start_time_seconds": 2.25,
             "end_time_seconds": 2.28, "speaker_id": "1"},
            {"transcript": "ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ ॥", "start_time_seconds": 2.29,
             "end_time_seconds": 2.34, "speaker_id": "1"},
            {"transcript": "  ", "start_time_seconds": 2.35,
             "end_time_seconds": 2.36, "speaker_id": "0"},
            {"transcript": "ਐ ਪਾਤਸ਼ਾਹ ਜੀ।", "start_time_seconds": 2.41,
             "end_time_seconds": 2.44, "speaker_id": "0"},
        ]
    },
}


def test_parse_diarized_reads_entries_in_order():
    entries = asr.parse_diarized(PAYLOAD)
    assert [e.speaker_id for e in entries] == ["1", "1", "0"]
    assert entries[1].text == "ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ ॥"
    assert entries[0].start_s == 2.25


def test_parse_diarized_drops_blank_entries():
    assert len(asr.parse_diarized(PAYLOAD)) == 3


def test_parse_diarized_coerces_speaker_id_to_str():
    # Sarvam's docs disagree with themselves: schema says "0"/"1", one page says SPEAKER_00.
    payload = {"diarized_transcript": {"entries": [
        {"transcript": "ਜੀ", "start_time_seconds": 0.0,
         "end_time_seconds": 1.0, "speaker_id": 0},
    ]}}
    assert asr.parse_diarized(payload)[0].speaker_id == "0"


def test_parse_diarized_without_diarization_is_an_error():
    with pytest.raises(asr.AsrError, match="diarized_transcript"):
        asr.parse_diarized({"transcript": "x"})
```

- [ ] **Step 3: Run to confirm it fails**

Run: `pytest tests/test_asr.py -v`
Expected: FAIL — `ImportError: cannot import name 'asr'`

- [ ] **Step 4: Write the ASR module**

`src/katha/stages/asr.py`:

```python
import json
from pathlib import Path

from katha.config import Settings
from katha.stages.types import DiarizedEntry


class AsrError(RuntimeError):
    pass


def parse_diarized(payload: dict) -> list[DiarizedEntry]:
    entries = (payload.get("diarized_transcript") or {}).get("entries")
    if not entries:
        raise AsrError(
            "response has no diarized_transcript entries — "
            "was the job created with with_diarization=True?"
        )
    parsed = [
        DiarizedEntry(
            speaker_id=str(e.get("speaker_id", "0")),
            text=(e.get("transcript") or "").strip(),
            start_s=float(e["start_time_seconds"]),
            end_s=float(e["end_time_seconds"]),
        )
        for e in entries
    ]
    return [e for e in parsed if e.text]


class SarvamAsr:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        if client is None:
            from sarvamai import SarvamAI

            if not settings.sarvam_api_key:
                raise AsrError("KATHA_SARVAM_API_KEY is not set")
            client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        self.client = client

    def transcribe(self, wav: Path, out_dir: Path) -> tuple[list[DiarizedEntry], str]:
        s = self.settings
        out_dir.mkdir(parents=True, exist_ok=True)

        job = self.client.speech_to_text_job.create_job(
            model=s.sarvam_model,          # MUST be explicit: SDK defaults to saarika:v2.5
            mode=s.sarvam_mode,
            language_code=s.sarvam_language,
            with_timestamps=True,          # docs' examples omit this; it defaults to False
            with_diarization=True,         # Batch API only
            num_speakers=s.num_speakers,
        )
        job.upload_files(file_paths=[str(wav)])
        job.start()

        # Default timeout is 600s and WILL blow up on a 90-minute clip.
        job.wait_until_complete(
            poll_interval=s.sarvam_poll_seconds, timeout=s.sarvam_timeout_seconds
        )

        failed = (job.get_file_results() or {}).get("failed") or []
        if failed:
            raise AsrError(f"Sarvam rejected the file: {failed}")

        job.download_outputs(output_dir=str(out_dir))
        outputs = sorted(out_dir.glob("*.json"))
        if not outputs:
            raise AsrError(f"Sarvam job completed but wrote no JSON to {out_dir}")

        payload = json.loads(outputs[0].read_text(encoding="utf-8"))
        return parse_diarized(payload), str(getattr(job, "job_id", ""))
```

- [ ] **Step 5: Run the parser tests**

Run: `pytest tests/test_asr.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Add a test that the client sends the verified parameters**

The single most likely production failure here is silently transcribing with the legacy model. Pin it with a fake client.

Append to `tests/test_asr.py`:

```python
from pathlib import Path


class FakeJob:
    job_id = "job-123"

    def __init__(self, out_payload):
        self._payload = out_payload
        self.started = False

    def upload_files(self, file_paths):
        self.uploaded = file_paths

    def start(self):
        self.started = True

    def wait_until_complete(self, poll_interval, timeout):
        self.waited = (poll_interval, timeout)

    def get_file_results(self):
        return {"successful": ["0.json"], "failed": []}

    def download_outputs(self, output_dir):
        import json
        (Path(output_dir) / "0.json").write_text(
            json.dumps(self._payload), encoding="utf-8"
        )


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.speech_to_text_job = self

    def create_job(self, **kwargs):
        self.create_kwargs = kwargs
        return FakeJob(self.payload)


def test_transcribe_pins_saaras_v3_and_a_long_timeout(tmp_path, settings):
    client = FakeClient(PAYLOAD)
    entries, job_id = asr.SarvamAsr(settings, client=client).transcribe(
        Path("/audio/asr.wav"), tmp_path
    )

    assert client.create_kwargs["model"] == "saaras:v3"
    assert client.create_kwargs["language_code"] == "pa-IN"
    assert client.create_kwargs["with_diarization"] is True
    assert client.create_kwargs["with_timestamps"] is True
    assert client.create_kwargs["num_speakers"] == 2
    assert len(entries) == 3
    assert job_id == "job-123"
```

- [ ] **Step 7: Run the full ASR suite**

Run: `pytest tests/test_asr.py -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Smoke-test against the real API with a short clip**

Cut 60 seconds from a real katha file and run it through for real. This costs about ₹0.75 and is the only way to confirm the API contract.

```bash
ffmpeg -y -i /srv/media/katha/incoming/sample.mp3 -t 60 -ac 1 -ar 16000 /tmp/smoke.wav
python - <<'PY'
from pathlib import Path
from katha.config import get_settings
from katha.stages.asr import SarvamAsr

entries, job_id = SarvamAsr(get_settings()).transcribe(Path("/tmp/smoke.wav"), Path("/tmp/sarvam"))
print("job:", job_id, "entries:", len(entries))
for e in entries[:5]:
    print(f"  spk{e.speaker_id} [{e.start_s:.2f}-{e.end_s:.2f}] {e.text}")
PY
```

Expected: Gurmukhi text, two distinct `speaker_id` values, plausible timings. **If `speaker_id` comes back as `SPEAKER_00` rather than `"0"`, nothing breaks** — the parser coerces to `str`. If diarization returns only one speaker on a clip you know has two, that is Sarvam's beta diarization underperforming; record it and continue — Task 6's role inference degrades to `unknown` and the admin sets roles by hand.

- [ ] **Step 9: Commit**

```bash
git add src/katha/stages/types.py src/katha/stages/asr.py tests/test_asr.py
git commit -m "feat: Sarvam Batch ASR client with diarization"
```

---

## Task 5: Forced alignment

**Files:**
- Create: `src/katha/stages/align.py`
- Create: `tests/test_align.py`

**Interfaces:**
- Consumes: `katha.config.Settings`, `katha.stages.types.AlignedWord`.
- Produces:
  - `katha.stages.align.AlignError` (exception).
  - `katha.stages.align.to_aligned_words(results: list[dict]) -> list[AlignedWord]` — pure.
  - `katha.stages.align.Aligner(settings)` with `.align(wav: Path, text: str) -> list[AlignedWord]`. Lazily imports torch so the rest of the suite runs without it.

**Read the Global Constraints before writing this.** `romanize=True` and `language="pan"` are both non-negotiable and both fail *silently* (garbage timings, not an exception) if wrong.

- [ ] **Step 1: Write the failing conversion test**

`tests/test_align.py`:

```python
import pytest

from katha.stages import align

# postprocess_results() returns dicts of exactly this shape.
RESULTS = [
    {"start": 2.29, "end": 2.41, "text": "ਤੋ", "score": 0.93},
    {"start": 2.42, "end": 2.55, "text": "ਸੋ", "score": 0.88},
    {"start": 2.56, "end": 2.80, "text": "ਨਹੀ", "score": 0.91},
]


def test_to_aligned_words_preserves_gurmukhi_and_order():
    words = align.to_aligned_words(RESULTS)
    assert [w.text for w in words] == ["ਤੋ", "ਸੋ", "ਨਹੀ"]
    assert words[0].start_s == 2.29
    assert words[2].end_s == 2.80


def test_to_aligned_words_defaults_a_missing_score():
    words = align.to_aligned_words([{"start": 0.0, "end": 1.0, "text": "ਜੀ"}])
    assert words[0].score == 0.0


def test_to_aligned_words_rejects_empty_alignment():
    with pytest.raises(align.AlignError, match="no words"):
        align.to_aligned_words([])
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_align.py -v`
Expected: FAIL — `ImportError: cannot import name 'align'`

- [ ] **Step 3: Write the alignment module**

`src/katha/stages/align.py`:

```python
from pathlib import Path

from katha.config import Settings
from katha.stages.types import AlignedWord

MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"


class AlignError(RuntimeError):
    pass


def to_aligned_words(results: list[dict]) -> list[AlignedWord]:
    if not results:
        raise AlignError("aligner returned no words")
    return [
        AlignedWord(
            text=r["text"],
            start_s=float(r["start"]),
            end_s=float(r["end"]),
            score=float(r.get("score", 0.0)),
        )
        for r in results
    ]


class Aligner:
    """Wraps ctc-forced-aligner. Loads the model once; reuse across clips."""

    def __init__(self, settings: Settings):
        import torch
        from ctc_forced_aligner import load_alignment_model

        self.settings = settings
        self.device = settings.device
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model, self.tokenizer = load_alignment_model(
            self.device, model_path=MODEL, dtype=dtype
        )

    def align(self, wav: Path, text: str) -> list[AlignedWord]:
        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            load_audio,
            postprocess_results,
            preprocess_text,
        )

        if not text.strip():
            raise AlignError("nothing to align: transcript is empty")

        # load_audio shells out to ffmpeg and resamples to 16 kHz mono for us.
        waveform = load_audio(str(wav), self.model.dtype, self.model.device)

        emissions, stride = generate_emissions(
            self.model,
            waveform,
            window_length=30,
            context_length=2,
            batch_size=self.settings.align_batch_size,
        )

        # romanize=True is MANDATORY: the model vocab is 26 Latin letters and has
        # zero Gurmukhi codepoints. Romanization is internal — the words that come
        # back out are the original Gurmukhi. language is ISO-639-3 ('pan'), not 'pa'.
        tokens_starred, text_starred = preprocess_text(
            text,
            romanize=True,
            language=self.settings.align_language,
            split_size="word",
            star_frequency="edges",  # the library default is 'segment'; the CLI's is 'edges'
        )

        segments, scores, blank = get_alignments(emissions, tokens_starred, self.tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        results = postprocess_results(text_starred, spans, stride, scores, merge_threshold=0.0)
        return to_aligned_words(results)
```

- [ ] **Step 4: Run the conversion tests**

Run: `pytest tests/test_align.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Add a real alignment test behind the `heavy` marker**

Append to `tests/test_align.py`:

```python
from pathlib import Path

import pytest


@pytest.mark.heavy
def test_align_returns_monotonic_gurmukhi_words(settings, tmp_path):
    """Requires torch + the MMS model (~1.2GB on first run). Uses the smoke wav
    produced in Task 4 Step 8."""
    wav = Path("/tmp/smoke.wav")
    if not wav.exists():
        pytest.skip("run Task 4 Step 8 first to create /tmp/smoke.wav")

    words = align.Aligner(settings).align(wav, "ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਿਹ")

    assert len(words) == 8
    assert words[0].text == "ਵਾਹਿਗੁਰੂ"          # Gurmukhi out, not romanized
    starts = [w.start_s for w in words]
    assert starts == sorted(starts)              # monotonic
```

- [ ] **Step 6: Install the aligner and run the heavy test**

```bash
pip install -e ".[align]"
pytest tests/test_align.py -v -m heavy
```

Expected: PASS. First run downloads the MMS model. If `words[0].text` comes back as `vaahiguroo` rather than `ਵਾਹਿਗੁਰੂ`, the `romanize` plumbing has leaked into the output — stop and re-read the Global Constraints. If word count is wrong or timings are non-monotonic, check that `language` is `pan` and not `pa`.

- [ ] **Step 7: Commit**

```bash
git add src/katha/stages/align.py tests/test_align.py
git commit -m "feat: Punjabi forced alignment via ctc-forced-aligner (MMS)"
```

---

## Task 6: Segment assembly and speaker-role inference

**Files:**
- Create: `src/katha/stages/assemble.py`
- Modify: `src/katha/repo.py` (add `save_segments`, `set_clip_meta`, `get_clip`)
- Create: `tests/test_assemble.py`

**Interfaces:**
- Consumes: `DiarizedEntry`, `AlignedWord`, `AssembledSegment` from `katha.stages.types`.
- Produces:
  - `katha.stages.assemble.alignment_text(entries: list[DiarizedEntry]) -> str` — the exact text handed to the aligner.
  - `katha.stages.assemble.assemble_segments(entries, words) -> list[AssembledSegment]`
  - `katha.stages.assemble.infer_reciter_speaker_id(segments) -> str | None`
  - `katha.stages.assemble.role_for(speaker_id: str, reciter_id: str | None) -> str`
  - `katha.repo.save_segments(conn, clip_id: int, segments: list[AssembledSegment], reciter_id: str | None) -> None` — replaces all segments for the clip, transactionally.
  - `katha.repo.set_clip_meta(conn, clip_id, *, duration_seconds=None, reciter_speaker_id=None) -> None`
  - `katha.repo.get_clip(conn, clip_id) -> dict | None`

**The mapping problem.** Sarvam gives sentence chunks with rough spans; the aligner gives a flat list of word timings. To get tight per-sentence spans we must map words back to chunks. Because the alignment text is *built from* the chunks in order, a sequential walk works: chunk *i* consumes its own whitespace-token count from the word list. If the totals disagree — the aligner dropped a punctuation-only token, say — the walk is unsafe, so we fall back wholesale to Sarvam's own chunk timings and mark `timing_source='asr'` so the UI can show that those spans are loose.

- [ ] **Step 1: Write the failing assembly tests**

`tests/test_assemble.py`:

```python
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

# Speaker "1" recites briefly; speaker "0" explains at length. The airtime gap is
# what infer_reciter_speaker_id keys on, so keep it unambiguous — a tie would make
# the test pass on dict ordering rather than on the heuristic.
ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ ਨਹੀ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਐ ਪਾਤਸ਼ਾਹ", start_s=3.5, end_s=5.5),
]
WORDS = [
    AlignedWord(text="ਤੋ", start_s=2.10, end_s=2.30, score=0.90),
    AlignedWord(text="ਸੋ", start_s=2.31, end_s=2.50, score=0.80),
    AlignedWord(text="ਨਹੀ", start_s=2.51, end_s=2.90, score=0.70),   # -> 800 ms of reciter
    AlignedWord(text="ਐ", start_s=3.60, end_s=3.80, score=0.60),
    AlignedWord(text="ਪਾਤਸ਼ਾਹ", start_s=3.81, end_s=5.40, score=0.50),  # -> 1800 ms of kathakar
]


def test_alignment_text_joins_entries_in_order():
    assert assemble.alignment_text(ENTRIES) == "ਤੋ ਸੋ ਨਹੀ ਐ ਪਾਤਸ਼ਾਹ"


def test_assemble_takes_spans_from_the_aligned_words_not_the_asr():
    segments = assemble.assemble_segments(ENTRIES, WORDS)

    assert segments[0].start_ms == 2100   # first word's start, not the ASR's 2.0
    assert segments[0].end_ms == 2900     # last word's end, not the ASR's 3.0
    assert segments[0].timing_source == "aligned"
    assert segments[0].align_score == 0.80  # mean of 0.90, 0.80, 0.70
    assert [w.text for w in segments[0].words] == ["ਤੋ", "ਸੋ", "ਨਹੀ"]
    assert segments[1].start_ms == 3600


def test_assemble_falls_back_to_asr_timings_when_word_counts_disagree():
    segments = assemble.assemble_segments(ENTRIES, WORDS[:-1])  # one word short

    assert [s.timing_source for s in segments] == ["asr", "asr"]
    assert segments[0].start_ms == 2000   # straight from the ASR chunk
    assert segments[0].end_ms == 3000
    assert segments[0].align_score is None
    assert segments[0].words == []


def test_infer_reciter_is_the_speaker_with_less_airtime():
    # The kathakar explains at length; the reciter only recites. Least airtime wins.
    segments = assemble.assemble_segments(ENTRIES, WORDS)
    assert assemble.infer_reciter_speaker_id(segments) == "1"


def test_infer_reciter_is_none_when_diarization_found_one_speaker():
    solo = [DiarizedEntry(speaker_id="0", text="ਜੀ", start_s=0.0, end_s=1.0)]
    segments = assemble.assemble_segments(solo, [
        AlignedWord(text="ਜੀ", start_s=0.0, end_s=1.0, score=0.9)
    ])
    assert assemble.infer_reciter_speaker_id(segments) is None


def test_role_for_maps_ids_to_roles():
    assert assemble.role_for("1", "1") == "reciter"
    assert assemble.role_for("0", "1") == "kathakar"
    assert assemble.role_for("0", None) == "unknown"
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_assemble.py -v`
Expected: FAIL — `ImportError: cannot import name 'assemble'`

- [ ] **Step 3: Write the assembly module**

`src/katha/stages/assemble.py`:

```python
from katha.stages.types import AlignedWord, AssembledSegment, DiarizedEntry


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def alignment_text(entries: list[DiarizedEntry]) -> str:
    """The exact text handed to the aligner. Building it FROM the entries, in order,
    is what makes the word-to-entry walk in assemble_segments() sound."""
    return " ".join(e.text.strip() for e in entries if e.text.strip())


def _from_asr(idx: int, entry: DiarizedEntry) -> AssembledSegment:
    return AssembledSegment(
        idx=idx,
        speaker_id=entry.speaker_id,
        text_pa=entry.text.strip(),
        start_ms=_ms(entry.start_s),
        end_ms=_ms(entry.end_s),
        align_score=None,
        timing_source="asr",
        words=[],
    )


def assemble_segments(
    entries: list[DiarizedEntry], words: list[AlignedWord]
) -> list[AssembledSegment]:
    entries = [e for e in entries if e.text.strip()]
    if not entries:
        return []

    expected = sum(len(e.text.split()) for e in entries)
    if len(words) != expected:
        # The sequential walk is only safe if every token got a word. It didn't —
        # so don't guess. Fall back to the ASR's own (looser) chunk timings and
        # let the UI flag them.
        return [_from_asr(i, e) for i, e in enumerate(entries)]

    segments: list[AssembledSegment] = []
    cursor = 0
    for idx, entry in enumerate(entries):
        n = len(entry.text.split())
        span = words[cursor : cursor + n]
        cursor += n
        segments.append(
            AssembledSegment(
                idx=idx,
                speaker_id=entry.speaker_id,
                text_pa=entry.text.strip(),
                start_ms=_ms(span[0].start_s),
                end_ms=_ms(span[-1].end_s),
                align_score=sum(w.score for w in span) / n,
                timing_source="aligned",
                words=span,
            )
        )
    return segments


def infer_reciter_speaker_id(segments: list[AssembledSegment]) -> str | None:
    """Heuristic only: the kathakar talks far more than the reciter recites, so the
    speaker with the least total airtime is the reciter. Admin can override per clip —
    see the Global Constraints on why diarization cannot classify verses on its own."""
    totals: dict[str, int] = {}
    for s in segments:
        totals[s.speaker_id] = totals.get(s.speaker_id, 0) + (s.end_ms - s.start_ms)
    if len(totals) < 2:
        return None
    return min(totals, key=lambda k: totals[k])


def role_for(speaker_id: str, reciter_id: str | None) -> str:
    if reciter_id is None:
        return "unknown"
    return "reciter" if speaker_id == reciter_id else "kathakar"
```

- [ ] **Step 4: Run the assembly tests**

Run: `pytest tests/test_assemble.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Write the failing persistence test**

Append to `tests/test_assemble.py`:

```python
import pytest
from pathlib import Path

from katha import repo


@pytest.mark.db
def test_save_segments_persists_roles_and_words(db):
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    segments = assemble.assemble_segments(ENTRIES, WORDS)
    repo.save_segments(db, clip_id, segments, reciter_id="1")

    with db.cursor() as cur:
        cur.execute(
            "SELECT idx, speaker_role, start_ms, text_pa FROM segments"
            " WHERE clip_id = %s ORDER BY idx",
            (clip_id,),
        )
        rows = cur.fetchall()
        cur.execute("SELECT count(*) AS n FROM words")
        word_count = cur.fetchone()["n"]

    assert [r["speaker_role"] for r in rows] == ["reciter", "kathakar"]
    assert rows[0]["start_ms"] == 2100
    assert word_count == 5
    assert repo.get_clip(db, clip_id)["reciter_speaker_id"] == "1"


@pytest.mark.db
def test_save_segments_replaces_a_previous_run(db):
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    segments = assemble.assemble_segments(ENTRIES, WORDS)
    repo.save_segments(db, clip_id, segments, reciter_id="1")
    repo.save_segments(db, clip_id, segments, reciter_id="1")  # a retry

    with db.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM segments WHERE clip_id = %s", (clip_id,))
        assert cur.fetchone()["n"] == 2
```

- [ ] **Step 6: Run to confirm it fails**

Run: `pytest tests/test_assemble.py -v -k save_segments`
Expected: FAIL — `AttributeError: module 'katha.repo' has no attribute 'save_segments'`

- [ ] **Step 7: Add the persistence functions**

Append to `src/katha/repo.py`:

```python
from katha.stages.assemble import role_for
from katha.stages.types import AssembledSegment


def get_clip(conn: psycopg.Connection, clip_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM clips WHERE id = %s", (clip_id,))
        return cur.fetchone()


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


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))
```

- [ ] **Step 8: Run the full assembly suite**

Run: `pytest tests/test_assemble.py -v`
Expected: PASS (8 tests)

- [ ] **Step 9: Commit**

```bash
git add src/katha/stages/assemble.py src/katha/repo.py tests/test_assemble.py
git commit -m "feat: segment assembly, speaker-role inference, and persistence"
```

---

## Task 7: Worker — the stage machine

**Files:**
- Create: `src/katha/worker.py`
- Create: `tests/test_worker.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6.
- Produces:
  - `katha.worker.Pipeline(settings, conn, aligner=None, asr=None)` — dependency-injected so tests need neither GPU nor API key.
  - `katha.worker.Pipeline.run_stage(job: dict) -> str` — runs exactly one stage, returns the next stage name.
  - `katha.worker.Pipeline.run_once() -> bool` — claim one job, run one stage, record the outcome. Returns `False` when there is nothing to do.
  - `katha.worker.main()` — CLI entry point: watch, then drain the queue.

**Design:** one stage per `run_once()` call, not one clip. A crash mid-clip loses at most one stage, and `sarvam_job_id` is persisted so a restart during `asr` never re-pays for transcription. The daily throttle is enforced here (at job *start*), not at registration.

- [ ] **Step 1: Write the failing worker tests**

`tests/test_worker.py`:

```python
from pathlib import Path

import pytest

from katha import repo, worker
from katha.stages.prepare import PrepareResult
from katha.stages.types import AlignedWord, DiarizedEntry

# As in test_assemble: speaker "0" must clearly out-talk speaker "1" so that the
# reciter heuristic has an unambiguous answer.
ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਐ ਪਾਤਸ਼ਾਹ", start_s=3.5, end_s=5.5),
]
WORDS = [
    AlignedWord(text="ਤੋ", start_s=2.1, end_s=2.3, score=0.9),
    AlignedWord(text="ਸੋ", start_s=2.4, end_s=2.9, score=0.9),      # -> 800 ms of reciter
    AlignedWord(text="ਐ", start_s=3.6, end_s=3.8, score=0.6),
    AlignedWord(text="ਪਾਤਸ਼ਾਹ", start_s=3.9, end_s=5.4, score=0.6),  # -> 1800 ms of kathakar
]


class FakeAsr:
    def __init__(self):
        self.calls = 0

    def transcribe(self, wav, out_dir):
        self.calls += 1
        return ENTRIES, "sarvam-job-1"


class FakeAligner:
    def __init__(self):
        self.text = None

    def align(self, wav, text):
        self.text = text
        return WORDS


@pytest.fixture
def pipeline(db, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "work_dir", tmp_path / "work")
    monkeypatch.setattr(
        worker, "prepare_clip",
        lambda source, work_dir, s: PrepareResult(
            native=work_dir / "native.wav",
            denoised=None,
            asr_ready=work_dir / "asr.wav",
            duration_seconds=90.0,
            native_sample_rate=48000,
        ),
    )
    return worker.Pipeline(settings, db, aligner=FakeAligner(), asr=FakeAsr())


@pytest.mark.db
def test_run_once_walks_a_clip_through_every_stage(db, pipeline):
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    stages = []
    while pipeline.run_once():
        with db.cursor() as cur:
            cur.execute("SELECT stage, status FROM jobs")
            stages.append(tuple(cur.fetchone().values()))

    assert stages == [
        ("asr", "queued"),
        ("align", "queued"),
        ("assemble", "queued"),
        ("complete", "done"),
    ]


@pytest.mark.db
def test_a_completed_clip_has_aligned_segments(db, pipeline):
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    while pipeline.run_once():
        pass

    with db.cursor() as cur:
        cur.execute(
            "SELECT speaker_role, start_ms, timing_source FROM segments"
            " WHERE clip_id = %s ORDER BY idx",
            (clip_id,),
        )
        rows = cur.fetchall()

    assert [r["speaker_role"] for r in rows] == ["reciter", "kathakar"]
    assert rows[0]["start_ms"] == 2100
    assert rows[0]["timing_source"] == "aligned"


@pytest.mark.db
def test_the_aligner_receives_text_built_from_the_diarized_entries(db, pipeline):
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    while pipeline.run_once():
        pass
    assert pipeline.aligner.text == "ਤੋ ਸੋ ਐ ਪਾਤਸ਼ਾਹ"


@pytest.mark.db
def test_a_failing_stage_is_recorded_and_retried_up_to_max_attempts(db, settings, pipeline, monkeypatch):
    monkeypatch.setattr(settings, "max_attempts", 2)
    monkeypatch.setattr(
        worker, "prepare_clip",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg exploded")),
    )
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    assert pipeline.run_once() is True   # attempt 1 -> requeued
    assert pipeline.run_once() is True   # attempt 2 -> exhausted
    assert pipeline.run_once() is False  # nothing claimable

    with db.cursor() as cur:
        cur.execute("SELECT status, attempts, last_error FROM jobs")
        job = cur.fetchone()
    assert job["status"] == "failed"
    assert job["attempts"] == 2
    assert "ffmpeg exploded" in job["last_error"]


@pytest.mark.db
def test_the_daily_throttle_stops_new_work(db, settings, pipeline):
    with db.cursor() as cur:
        cur.execute("UPDATE settings SET value = '0' WHERE key = 'files_per_day'")
    db.commit()
    repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")

    assert pipeline.run_once() is False
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_worker.py -v`
Expected: FAIL — `ImportError: cannot import name 'worker' from 'katha'`

- [ ] **Step 3: Write the worker**

`src/katha/worker.py`:

```python
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg

from katha import repo
from katha.config import Settings, get_settings
from katha.db import connect
from katha.ingest.watcher import scan_once
from katha.stages.asr import SarvamAsr
from katha.stages.assemble import (
    alignment_text,
    assemble_segments,
    infer_reciter_speaker_id,
)
from katha.stages.prepare import prepare_clip

log = logging.getLogger("katha.worker")

NEXT_STAGE = {
    "prepare": "asr",
    "asr": "align",
    "align": "assemble",
    "assemble": "complete",
}


class Pipeline:
    def __init__(self, settings: Settings, conn: psycopg.Connection, aligner=None, asr=None):
        self.settings = settings
        self.conn = conn
        self._aligner = aligner
        self._asr = asr
        # Stage outputs that live only in memory between stages of the same clip.
        # Everything durable goes to the DB; this is just a cache so `align` doesn't
        # have to re-download the ASR result.
        self._entries: dict[int, list] = {}
        self._words: dict[int, list] = {}

    @property
    def aligner(self):
        if self._aligner is None:
            from katha.stages.align import Aligner

            self._aligner = Aligner(self.settings)
        return self._aligner

    @property
    def asr(self):
        if self._asr is None:
            self._asr = SarvamAsr(self.settings)
        return self._asr

    def _work_dir(self, clip_id: int) -> Path:
        return self.settings.work_dir / str(clip_id)

    def _quota_left(self) -> int:
        limit = repo.get_setting_int(self.conn, "files_per_day", 3)
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        return limit - repo.count_jobs_started_since(self.conn, since)

    # --- stages -----------------------------------------------------------

    def _prepare(self, job: dict) -> None:
        clip = repo.get_clip(self.conn, job["clip_id"])
        result = prepare_clip(
            Path(clip["source_path"]), self._work_dir(job["clip_id"]), self.settings
        )
        repo.add_audio_file(
            self.conn, job["clip_id"], "original", Path(clip["source_path"]), None
        )
        repo.add_audio_file(
            self.conn, job["clip_id"], "native", result.native, result.native_sample_rate
        )
        if result.denoised:
            repo.add_audio_file(
                self.conn, job["clip_id"], "denoised", result.denoised,
                result.native_sample_rate,
            )
        repo.add_audio_file(self.conn, job["clip_id"], "asr_ready", result.asr_ready, 16000)
        repo.set_clip_meta(
            self.conn, job["clip_id"], duration_seconds=result.duration_seconds
        )

    def _asr_stage(self, job: dict) -> None:
        clip_id = job["clip_id"]
        work = self._work_dir(clip_id)
        entries, sarvam_job_id = self.asr.transcribe(work / "asr.wav", work / "sarvam")
        repo.set_job(self.conn, job["id"], sarvam_job_id=sarvam_job_id)
        self._entries[clip_id] = entries

    def _align(self, job: dict) -> None:
        clip_id = job["clip_id"]
        entries = self._entries[clip_id]
        self._words[clip_id] = self.aligner.align(
            self._work_dir(clip_id) / "asr.wav", alignment_text(entries)
        )

    def _assemble(self, job: dict) -> None:
        clip_id = job["clip_id"]
        segments = assemble_segments(self._entries[clip_id], self._words[clip_id])
        reciter_id = infer_reciter_speaker_id(segments)
        repo.save_segments(self.conn, clip_id, segments, reciter_id)
        self._entries.pop(clip_id, None)
        self._words.pop(clip_id, None)

    # --- driver -----------------------------------------------------------

    def run_stage(self, job: dict) -> str:
        {
            "prepare": self._prepare,
            "asr": self._asr_stage,
            "align": self._align,
            "assemble": self._assemble,
        }[job["stage"]](job)
        return NEXT_STAGE[job["stage"]]

    def run_once(self) -> bool:
        if self._quota_left() <= 0:
            log.info("daily throttle reached; idling")
            return False

        job = repo.claim_next_job(self.conn)
        if job is None:
            return False

        try:
            next_stage = self.run_stage(job)
        except Exception as exc:  # noqa: BLE001 — the stage boundary is the retry boundary
            exhausted = job["attempts"] >= self.settings.max_attempts
            log.exception("clip %s failed in %s", job["clip_id"], job["stage"])
            repo.set_job(
                self.conn,
                job["id"],
                status="failed" if exhausted else "queued",
                last_error=f"{job['stage']}: {exc}",
            )
            return True

        repo.set_job(
            self.conn,
            job["id"],
            stage=next_stage,
            status="done" if next_stage == "complete" else "queued",
        )
        return True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    with connect(settings) as conn:
        new = scan_once(conn, settings)
        log.info("registered %d new clip(s)", len(new))
        pipeline = Pipeline(settings, conn)
        while pipeline.run_once():
            pass
    log.info("queue drained")


if __name__ == "__main__":
    main()
```

Note the in-memory `_entries` / `_words` cache: a clip's stages run back-to-back in one process. If the worker dies between `asr` and `align`, the restarted worker's cache is empty and `_align` raises `KeyError` — which the retry path turns into a normal failure, and the recorded `sarvam_job_id` means a human can re-download that transcript rather than re-pay for it. Making the ASR result durable across restarts is a follow-on improvement, not a slice requirement; it is called out in the Deferred list below.

- [ ] **Step 4: Run the worker tests**

Run: `pytest tests/test_worker.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: PASS. Everything except `-m heavy` runs without torch or an API key.

- [ ] **Step 6: Drive one real clip end to end**

```bash
cp /path/to/one-real-katha.mp3 /srv/media/katha/incoming/
python -m katha.worker
```

Expected log: `registered 1 new clip(s)`, then four stages, then `queue drained`. Then confirm the result is real:

```bash
docker compose exec db psql -U katha -d katha -c \
  "SELECT idx, speaker_role, start_ms, end_ms, timing_source, left(text_pa, 40)
   FROM segments ORDER BY idx LIMIT 15"
```

Expected: Gurmukhi text, alternating roles, `timing_source = 'aligned'`. **If `timing_source` is `asr` for every row**, the word-count walk failed — check whether Sarvam's `transcript` normalisation is inserting or dropping tokens relative to the diarized entries, and compare `sum(len(e.text.split()))` against `len(words)` by hand. That is the one integration risk this plan cannot fully de-risk in unit tests.

- [ ] **Step 7: Schedule the worker so the pipeline is actually hands-off**

`worker.main()` drains the queue and exits. Without a scheduler nothing ever calls it, and the "drop a file in and walk away" goal is not met. A systemd timer is the right shape here — the worker is a batch drain, not a daemon, so a timer beats a `while True` loop with a `sleep` in it: crashes get restarted by systemd, and the daily throttle already lives in the database.

`/etc/systemd/system/katha-worker.service`:

```ini
[Unit]
Description=Katha pipeline worker (drain the queue once)
After=network-online.target docker.service

[Service]
Type=oneshot
User=katha
WorkingDirectory=/opt/katha
EnvironmentFile=/opt/katha/.env
ExecStart=/opt/katha/.venv/bin/python -m katha.worker
```

`/etc/systemd/system/katha-worker.timer`:

```ini
[Unit]
Description=Run the katha worker every 15 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=15min
Unit=katha-worker.service

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now katha-worker.timer
systemctl list-timers katha-worker.timer
```

Expected: the timer is listed with a next-run time. `journalctl -u katha-worker -f` shows a drain cycle every 15 minutes. Overlap is not a concern — `claim_next_job` uses `FOR UPDATE SKIP LOCKED`, so a slow run and the next tick cannot claim the same job.

Verify the whole loop for real: drop a second clip into the watch folder, wait, and confirm it reaches `complete` with no manual command.

- [ ] **Step 8: Commit**

```bash
git add src/katha/worker.py tests/test_worker.py deploy/
git commit -m "feat: stage-machine worker with retries, throttle, and systemd timer"
```

Keep copies of both unit files under `deploy/` in the repo so the host config is version-controlled rather than living only in `/etc`.

---

## Task 8: API — clips, segments, and audio range requests

**Files:**
- Create: `src/katha/api/__init__.py`, `src/katha/api/ranges.py`, `src/katha/api/main.py`
- Modify: `src/katha/repo.py` (add `list_clips`, `list_segments`, `get_audio_file`, `update_segment_text`, `requeue_job`)
- Create: `tests/test_ranges.py`, `tests/test_api.py`

**Interfaces:**
- Produces:
  - `katha.api.ranges.parse_range(header: str | None, size: int) -> tuple[int, int] | None` — pure; raises `ValueError` when the range is unsatisfiable.
  - FastAPI app at `katha.api.main:app` with:
    - `GET /api/clips`
    - `GET /api/clips/{clip_id}` → clip + segments
    - `PATCH /api/clips/{clip_id}` → `{"reciter_speaker_id": "1"}`, recomputes every segment's role
    - `PATCH /api/segments/{segment_id}` → `{"text_pa": "..."}`, sets `edited = true`
    - `POST /api/clips/{clip_id}/retry`
    - `GET /api/audio/{clip_id}` → range-capable audio stream (serves the `native` file — denoised audio is for the machine, the admin should hear what was actually said)

**Why range requests matter:** spec §8 requires seeking to an offset without downloading a 90-minute file. A plain `FileResponse` makes the browser fetch the whole thing before it can seek.

- [ ] **Step 1: Write the failing range tests**

`tests/test_ranges.py`:

```python
import pytest

from katha.api.ranges import parse_range


def test_no_header_means_the_whole_file():
    assert parse_range(None, 1000) is None


def test_open_ended_range_runs_to_the_last_byte():
    assert parse_range("bytes=500-", 1000) == (500, 999)


def test_closed_range_is_inclusive():
    assert parse_range("bytes=0-499", 1000) == (0, 499)


def test_range_past_the_end_is_clamped():
    assert parse_range("bytes=900-5000", 1000) == (900, 999)


def test_suffix_range_counts_back_from_the_end():
    assert parse_range("bytes=-200", 1000) == (800, 999)


def test_a_start_past_the_end_is_unsatisfiable():
    with pytest.raises(ValueError):
        parse_range("bytes=2000-", 1000)


def test_a_malformed_header_is_ignored_rather_than_fatal():
    assert parse_range("furlongs=0-10", 1000) is None
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest tests/test_ranges.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'katha.api'`

- [ ] **Step 3: Write the range parser**

`src/katha/api/__init__.py`: (empty file)

`src/katha/api/ranges.py`:

```python
import re

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """RFC 7233 single-range parsing. Returns inclusive (start, end), or None for
    'send the whole file'. Raises ValueError when the range cannot be satisfied."""
    if not header:
        return None
    match = _RANGE.fullmatch(header.strip())
    if not match:
        return None

    raw_start, raw_end = match.group(1), match.group(2)
    if not raw_start and not raw_end:
        return None

    if not raw_start:  # suffix range: last N bytes
        length = int(raw_end)
        if length <= 0:
            raise ValueError("zero-length suffix range")
        return max(0, size - length), size - 1

    start = int(raw_start)
    end = int(raw_end) if raw_end else size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        raise ValueError(f"range {start}-{end} is outside a {size}-byte file")
    return start, end
```

- [ ] **Step 4: Run the range tests**

Run: `pytest tests/test_ranges.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Add the read/write repo functions**

Append to `src/katha/repo.py`:

```python
def list_clips(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id, c.title, c.duration_seconds, c.reciter_speaker_id,
                   j.stage, j.status, j.attempts, j.last_error,
                   (SELECT count(*) FROM segments s WHERE s.clip_id = c.id) AS segment_count,
                   (SELECT count(*) FROM segments s
                     WHERE s.clip_id = c.id AND s.timing_source = 'asr') AS loose_count
            FROM clips c JOIN jobs j ON j.clip_id = c.id
            ORDER BY c.created_at DESC
        """)
        return cur.fetchall()


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
                   ) AS words
            FROM segments s WHERE s.clip_id = %s ORDER BY s.idx
        """, (clip_id,))
        return cur.fetchall()


def get_audio_file(conn: psycopg.Connection, clip_id: int, kind: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM audio_files WHERE clip_id = %s AND kind = %s", (clip_id, kind)
        )
        return cur.fetchone()


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
                WHEN %s IS NULL          THEN 'unknown'
                WHEN speaker_id = %s     THEN 'reciter'
                ELSE 'kathakar'
            END
            WHERE clip_id = %s
        """, (reciter_id, reciter_id, clip_id))
    conn.commit()


def requeue_job(conn: psycopg.Connection, clip_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = 'queued', attempts = 0, last_error = NULL,"
            " updated_at = now() WHERE clip_id = %s AND status = 'failed'",
            (clip_id,),
        )
    conn.commit()
```

- [ ] **Step 6: Write the failing API tests**

`tests/test_api.py`:

```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from katha import repo
from katha.api.main import app, get_conn, get_app_settings
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਐ ਪਾਤਸ਼ਾਹ", start_s=3.5, end_s=4.5),
]
WORDS = [
    AlignedWord(text="ਤੋ", start_s=2.1, end_s=2.3, score=0.9),
    AlignedWord(text="ਸੋ", start_s=2.4, end_s=2.9, score=0.9),
    AlignedWord(text="ਐ", start_s=3.6, end_s=3.8, score=0.6),
    AlignedWord(text="ਪਾਤਸ਼ਾਹ", start_s=3.9, end_s=4.4, score=0.6),
]


@pytest.fixture
def client(db, settings, tmp_path):
    audio = tmp_path / "native.wav"
    audio.write_bytes(bytes(range(256)) * 4)  # 1024 deterministic bytes

    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_app_settings] = lambda: settings
    yield TestClient(app), audio
    app.dependency_overrides.clear()


@pytest.fixture
def clip_id(db, tmp_path):
    cid = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    repo.save_segments(db, cid, assemble.assemble_segments(ENTRIES, WORDS), reciter_id="1")
    repo.add_audio_file(db, cid, "native", tmp_path / "native.wav", 48000)
    return cid


@pytest.mark.db
def test_get_clip_returns_segments_with_words(client, clip_id):
    api, _ = client
    body = api.get(f"/api/clips/{clip_id}").json()
    assert body["clip"]["reciter_speaker_id"] == "1"
    assert len(body["segments"]) == 2
    assert body["segments"][0]["speaker_role"] == "reciter"
    assert [w["text"] for w in body["segments"][0]["words"]] == ["ਤੋ", "ਸੋ"]


@pytest.mark.db
def test_patching_the_reciter_reassigns_every_role(client, clip_id):
    api, _ = client
    api.patch(f"/api/clips/{clip_id}", json={"reciter_speaker_id": "0"})
    segments = api.get(f"/api/clips/{clip_id}").json()["segments"]
    assert [s["speaker_role"] for s in segments] == ["kathakar", "reciter"]


@pytest.mark.db
def test_editing_a_segment_marks_it_edited(client, clip_id):
    api, _ = client
    segment_id = api.get(f"/api/clips/{clip_id}").json()["segments"][0]["id"]
    api.patch(f"/api/segments/{segment_id}", json={"text_pa": "ਤੋ ਸੋ ਨਹੀ"})
    segments = api.get(f"/api/clips/{clip_id}").json()["segments"]
    assert segments[0]["text_pa"] == "ਤੋ ਸੋ ਨਹੀ"
    assert segments[0]["edited"] is True


@pytest.mark.db
def test_audio_without_a_range_header_returns_the_whole_file(client, clip_id):
    api, _ = client
    r = api.get(f"/api/audio/{clip_id}")
    assert r.status_code == 200
    assert r.headers["accept-ranges"] == "bytes"
    assert len(r.content) == 1024


@pytest.mark.db
def test_audio_with_a_range_header_returns_206_and_only_that_slice(client, clip_id):
    api, audio = client
    r = api.get(f"/api/audio/{clip_id}", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206
    assert r.headers["content-range"] == "bytes 100-199/1024"
    assert r.content == audio.read_bytes()[100:200]


@pytest.mark.db
def test_an_unsatisfiable_range_returns_416(client, clip_id):
    api, _ = client
    r = api.get(f"/api/audio/{clip_id}", headers={"Range": "bytes=99999-"})
    assert r.status_code == 416
```

- [ ] **Step 7: Run to confirm it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'katha.api.main'`

- [ ] **Step 8: Write the API**

`src/katha/api/main.py`:

```python
from pathlib import Path
from typing import Iterator

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from katha import repo
from katha.api.ranges import parse_range
from katha.config import Settings, get_settings
from katha.db import connect

app = FastAPI(title="Katha Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK = 64 * 1024


def get_app_settings() -> Settings:
    return get_settings()


def get_conn(settings: Settings = Depends(get_app_settings)) -> Iterator[psycopg.Connection]:
    conn = connect(settings)
    try:
        yield conn
    finally:
        conn.close()


class ReciterPatch(BaseModel):
    reciter_speaker_id: str | None


class SegmentPatch(BaseModel):
    text_pa: str


@app.get("/api/clips")
def list_clips(conn=Depends(get_conn)):
    return {"clips": repo.list_clips(conn)}


@app.get("/api/clips/{clip_id}")
def get_clip(clip_id: int, conn=Depends(get_conn)):
    clip = repo.get_clip(conn, clip_id)
    if clip is None:
        raise HTTPException(404, "no such clip")
    return {"clip": clip, "segments": repo.list_segments(conn, clip_id)}


@app.patch("/api/clips/{clip_id}")
def patch_clip(clip_id: int, patch: ReciterPatch, conn=Depends(get_conn)):
    if repo.get_clip(conn, clip_id) is None:
        raise HTTPException(404, "no such clip")
    repo.set_reciter(conn, clip_id, patch.reciter_speaker_id)
    return {"ok": True}


@app.patch("/api/segments/{segment_id}")
def patch_segment(segment_id: int, patch: SegmentPatch, conn=Depends(get_conn)):
    if not repo.update_segment_text(conn, segment_id, patch.text_pa):
        raise HTTPException(404, "no such segment")
    return {"ok": True}


@app.post("/api/clips/{clip_id}/retry")
def retry_clip(clip_id: int, conn=Depends(get_conn)):
    repo.requeue_job(conn, clip_id)
    return {"ok": True}


@app.get("/api/audio/{clip_id}")
def get_audio(clip_id: int, range: str | None = Header(default=None), conn=Depends(get_conn)):
    # Serve the NATIVE file, not the denoised one: the admin is checking what was
    # actually said, and the denoiser is a machine-listening aid, not ground truth.
    row = repo.get_audio_file(conn, clip_id, "native")
    if row is None:
        raise HTTPException(404, "clip has no prepared audio yet")

    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(404, f"audio missing from disk: {path}")

    size = path.stat().st_size
    try:
        span = parse_range(range, size)
    except ValueError:
        return Response(
            status_code=416, headers={"Content-Range": f"bytes */{size}"}
        )

    headers = {"Accept-Ranges": "bytes"}
    if span is None:
        start, end, status = 0, size - 1, 200
    else:
        start, end = span
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)

    def stream() -> Iterator[bytes]:
        remaining = end - start + 1
        with path.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        stream(), status_code=status, headers=headers, media_type="audio/wav"
    )
```

- [ ] **Step 9: Run the API tests**

Run: `pytest tests/test_api.py -v`
Expected: PASS (6 tests)

- [ ] **Step 10: Start the API and check it against the real clip from Task 7**

```bash
uvicorn katha.api.main:app --host 0.0.0.0 --port 8000 &
curl -s localhost:8000/api/clips | head -c 400
curl -s -D- -o /dev/null -H "Range: bytes=0-1023" localhost:8000/api/audio/1
```

Expected: the clip list shows your real clip; the audio request returns `HTTP/1.1 206 Partial Content` with a `Content-Range` header.

- [ ] **Step 11: Commit**

```bash
git add src/katha/api src/katha/repo.py tests/test_ranges.py tests/test_api.py
git commit -m "feat: API with clips, segments, and range-capable audio streaming"
```

---

## Task 9: Admin UI — job dashboard

**Files:**
- Create: `web/package.json`, `web/next.config.mjs`, `web/tsconfig.json`, `web/.env.local.example`
- Create: `web/app/layout.tsx`, `web/app/globals.css`, `web/app/page.tsx`
- Create: `web/lib/api.ts`, `web/lib/types.ts`

**Interfaces:**
- Consumes: `GET /api/clips`, `POST /api/clips/{id}/retry` from Task 8.
- Produces:
  - `web/lib/types.ts` — `Clip`, `Segment`, `Word` TypeScript types mirroring the API.
  - `web/lib/api.ts` — `fetchClips()`, `fetchClip(id)`, `patchReciter(id, speakerId)`, `patchSegment(id, textPa)`, `retryClip(id)`, `audioUrl(id)`.

- [ ] **Step 1: Scaffold the Next.js app**

```bash
cd web
npm init -y
npm install next@latest react@latest react-dom@latest gurmukhi-utils@3.2.2
npm install -D typescript @types/react @types/node
```

`web/package.json` scripts block:

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start -p 3000"
  }
}
```

`web/next.config.mjs`:

```js
export default { reactStrictMode: true }
```

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "ES2022"],
    "jsx": "preserve",
    "module": "esnext",
    "moduleResolution": "bundler",
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`web/.env.local.example`:

```bash
NEXT_PUBLIC_API=http://localhost:8000
```

**`gurmukhi-utils` is pinned to 3.2.2 deliberately.** The monorepo's `4.0.0-alpha.0` is unpublished, and the scoped `@shabados/gurmukhi-utils` is a stale mirror. 3.2.2 is the one that ships `index.d.ts` with `toEnglish`/`toHindi`/`isGurmukhi`.

- [ ] **Step 2: Write the shared types**

`web/lib/types.ts`:

```ts
export type Word = { text: string; start_ms: number; end_ms: number }

export type Segment = {
  id: number
  idx: number
  speaker_id: string
  speaker_role: 'reciter' | 'kathakar' | 'unknown'
  start_ms: number
  end_ms: number
  text_pa: string
  align_score: number | null
  timing_source: 'aligned' | 'asr'
  edited: boolean
  words: Word[]
}

export type Clip = {
  id: number
  title: string
  duration_seconds: number | null
  reciter_speaker_id: string | null
  stage: 'prepare' | 'asr' | 'align' | 'assemble' | 'complete'
  status: 'queued' | 'running' | 'failed' | 'done'
  attempts: number
  last_error: string | null
  segment_count: number
  loose_count: number
}
```

- [ ] **Step 3: Write the API client**

`web/lib/api.ts`:

```ts
import type { Clip, Segment } from './types'

const API = process.env.NEXT_PUBLIC_API ?? 'http://localhost:8000'

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) throw new Error(`${init?.method ?? 'GET'} ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

export const fetchClips = () => json<{ clips: Clip[] }>('/api/clips').then((r) => r.clips)

export const fetchClip = (id: number) =>
  json<{ clip: Clip; segments: Segment[] }>(`/api/clips/${id}`)

export const patchReciter = (id: number, reciter_speaker_id: string | null) =>
  json(`/api/clips/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ reciter_speaker_id }),
  })

export const patchSegment = (id: number, text_pa: string) =>
  json(`/api/segments/${id}`, { method: 'PATCH', body: JSON.stringify({ text_pa }) })

export const retryClip = (id: number) =>
  json(`/api/clips/${id}/retry`, { method: 'POST' })

export const audioUrl = (clipId: number) => `${API}/api/audio/${clipId}`
```

- [ ] **Step 4: Write the layout and styles**

`web/app/layout.tsx`:

```tsx
import type { ReactNode } from 'react'
import './globals.css'

export const metadata = { title: 'Katha Pipeline' }

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header>
          <a href="/">Katha Pipeline</a>
        </header>
        <main>{children}</main>
      </body>
    </html>
  )
}
```

`web/app/globals.css`:

```css
:root {
  --bg: #12100e;
  --panel: #1c1917;
  --line: #33302c;
  --ink: #ece7e1;
  --muted: #9a938b;
  --reciter: #d4a24c;
  --kathakar: #7fa8c9;
  --warn: #c96f5c;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.6 system-ui, sans-serif;
}

header {
  padding: 14px 24px;
  border-bottom: 1px solid var(--line);
  font-weight: 600;
}
header a { color: var(--ink); text-decoration: none; }

main { padding: 24px; max-width: 1100px; margin: 0 auto; }

/* Gurmukhi needs more line height and a font that actually has the glyphs. */
.gurmukhi {
  font-family: 'Noto Sans Gurmukhi', 'Nirmala UI', sans-serif;
  font-size: 19px;
  line-height: 2;
}

table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }
th { color: var(--muted); font-weight: 500; font-size: 13px; }

.pill {
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 12px; border: 1px solid var(--line);
}
.pill.failed { color: var(--warn); border-color: var(--warn); }
.pill.done { color: var(--reciter); border-color: var(--reciter); }

button {
  background: var(--panel); color: var(--ink); border: 1px solid var(--line);
  border-radius: 6px; padding: 6px 12px; cursor: pointer; font: inherit;
}
button:hover { border-color: var(--muted); }
```

- [ ] **Step 5: Write the dashboard**

`web/app/page.tsx`:

```tsx
'use client'

import { useEffect, useState } from 'react'
import { fetchClips, retryClip } from '@/lib/api'
import type { Clip } from '@/lib/types'

function duration(seconds: number | null): string {
  if (!seconds) return '—'
  const m = Math.floor(seconds / 60)
  return `${m}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

export default function Dashboard() {
  const [clips, setClips] = useState<Clip[]>([])
  const [error, setError] = useState<string | null>(null)

  const load = () => fetchClips().then(setClips).catch((e) => setError(String(e)))

  useEffect(() => {
    load()
    const timer = setInterval(load, 5000) // stages are minutes long; 5s is plenty
    return () => clearInterval(timer)
  }, [])

  if (error) return <p className="pill failed">{error}</p>

  return (
    <table>
      <thead>
        <tr>
          <th>Clip</th><th>Length</th><th>Stage</th>
          <th>Segments</th><th>Loose timings</th><th></th>
        </tr>
      </thead>
      <tbody>
        {clips.map((c) => (
          <tr key={c.id}>
            <td>
              {c.status === 'done'
                ? <a href={`/clips/${c.id}`}>{c.title}</a>
                : c.title}
              {c.last_error && <div className="pill failed">{c.last_error}</div>}
            </td>
            <td>{duration(c.duration_seconds)}</td>
            <td><span className={`pill ${c.status}`}>{c.stage} · {c.status}</span></td>
            <td>{c.segment_count}</td>
            <td>{c.loose_count > 0 ? <span className="pill failed">{c.loose_count}</span> : '—'}</td>
            <td>
              {c.status === 'failed' && (
                <button onClick={() => retryClip(c.id).then(load)}>Retry</button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
```

The **Loose timings** column surfaces the `timing_source = 'asr'` fallback from Task 6. A non-zero count means alignment did not map cleanly onto that clip and its spans are only as tight as Sarvam's sentence chunks — exactly the thing you need to see, not have buried.

- [ ] **Step 6: Run it against a real clip**

```bash
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`. Expected: the clip you processed in Task 7, showing `complete · done`, its segment count, and a clickable title.

- [ ] **Step 7: Commit**

```bash
cd .. && git add web/
git commit -m "feat: admin dashboard with job status and retry"
```

---

## Task 10: Admin UI — review view with synced audio

**Files:**
- Create: `web/app/clips/[id]/page.tsx`
- Create: `web/components/SegmentList.tsx`, `web/components/SpeakerRoles.tsx`
- Create: `web/lib/roman.ts`

**Interfaces:**
- Consumes: `fetchClip`, `patchReciter`, `patchSegment`, `audioUrl` from Task 9.
- Produces: the review screen. This is the deliverable the whole slice exists for.

**Behaviour required by spec §7:** selecting a segment plays *only* that span (using the stored `start_ms`/`end_ms`), the transcript is shown in Gurmukhi with a romanized reading aid, and the admin can correct text inline. Plus the role override this plan added, because diarization cannot be trusted to identify the reciter.

- [ ] **Step 1: Write the romanization helper**

`web/lib/roman.ts`:

```ts
import pkg from 'gurmukhi-utils'

const { toEnglish } = pkg as { toEnglish: (t: string) => string }

/** Romanized Punjabi is the review aid, per spec §3.3: easier to scan and to type
 *  than Gurmukhi. Gurmukhi remains the canonical stored form — never write this back. */
export function romanize(gurmukhi: string): string {
  try {
    return toEnglish(gurmukhi)
  } catch {
    return ''
  }
}
```

- [ ] **Step 2: Write the speaker-role control**

`web/components/SpeakerRoles.tsx`:

```tsx
'use client'

import type { Clip, Segment } from '@/lib/types'

export function SpeakerRoles({
  clip,
  segments,
  onChange,
}: {
  clip: Clip
  segments: Segment[]
  onChange: (speakerId: string | null) => void
}) {
  const speakers = [...new Set(segments.map((s) => s.speaker_id))].sort()

  return (
    <div style={{ margin: '16px 0', padding: 12, border: '1px solid var(--line)', borderRadius: 8 }}>
      <div style={{ color: 'var(--muted)', fontSize: 13, marginBottom: 8 }}>
        Which speaker is the reciter? (Guessed from airtime — verify by ear.)
      </div>
      {speakers.map((id) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          style={{
            marginRight: 8,
            borderColor: clip.reciter_speaker_id === id ? 'var(--reciter)' : undefined,
            color: clip.reciter_speaker_id === id ? 'var(--reciter)' : undefined,
          }}
        >
          Speaker {id} {clip.reciter_speaker_id === id && '· reciter'}
        </button>
      ))}
      <button onClick={() => onChange(null)}>Neither / unclear</button>
    </div>
  )
}
```

- [ ] **Step 3: Write the segment list with span playback**

`web/components/SegmentList.tsx`:

```tsx
'use client'

import { useState } from 'react'
import { romanize } from '@/lib/roman'
import type { Segment } from '@/lib/types'

const clock = (ms: number) => {
  const total = Math.floor(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export function SegmentList({
  segments,
  activeId,
  onPlay,
  onSave,
}: {
  segments: Segment[]
  activeId: number | null
  onPlay: (segment: Segment) => void
  onSave: (id: number, text: string) => void
}) {
  const [editingId, setEditingId] = useState<number | null>(null)
  const [draft, setDraft] = useState('')

  return (
    <div>
      {segments.map((s) => {
        const colour = s.speaker_role === 'reciter' ? 'var(--reciter)' : 'var(--kathakar)'
        return (
          <div
            key={s.id}
            style={{
              padding: '12px 14px',
              margin: '8px 0',
              borderRadius: 8,
              borderLeft: `3px solid ${colour}`,
              background: activeId === s.id ? 'var(--panel)' : 'transparent',
            }}
          >
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', fontSize: 13 }}>
              <button onClick={() => onPlay(s)}>▶ {clock(s.start_ms)}</button>
              <span style={{ color: colour }}>{s.speaker_role}</span>
              {s.timing_source === 'asr' && (
                <span className="pill failed" title="Forced alignment did not map to this clip; span is only as tight as the ASR sentence chunk">
                  loose timing
                </span>
              )}
              {s.edited && <span className="pill">edited</span>}
            </div>

            {editingId === s.id ? (
              <>
                <textarea
                  className="gurmukhi"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  style={{
                    width: '100%', marginTop: 8, background: 'var(--bg)',
                    color: 'var(--ink)', border: '1px solid var(--line)',
                    borderRadius: 6, padding: 8,
                  }}
                  rows={2}
                />
                <button onClick={() => { onSave(s.id, draft); setEditingId(null) }}>Save</button>
                <button onClick={() => setEditingId(null)} style={{ marginLeft: 6 }}>Cancel</button>
              </>
            ) : (
              <div
                className="gurmukhi"
                onDoubleClick={() => { setEditingId(s.id); setDraft(s.text_pa) }}
                title="Double-click to correct"
                style={{ marginTop: 4 }}
              >
                {s.text_pa}
              </div>
            )}

            <div style={{ color: 'var(--muted)', fontSize: 13 }}>{romanize(s.text_pa)}</div>
          </div>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 4: Write the review page**

`web/app/clips/[id]/page.tsx`:

```tsx
'use client'

import { use, useCallback, useEffect, useRef, useState } from 'react'
import { SegmentList } from '@/components/SegmentList'
import { SpeakerRoles } from '@/components/SpeakerRoles'
import { audioUrl, fetchClip, patchReciter, patchSegment } from '@/lib/api'
import type { Clip, Segment } from '@/lib/types'

export default function Review({ params }: { params: Promise<{ id: string }> }) {
  const clipId = Number(use(params).id)
  const audio = useRef<HTMLAudioElement>(null)
  const stopAt = useRef<number | null>(null)

  const [clip, setClip] = useState<Clip | null>(null)
  const [segments, setSegments] = useState<Segment[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)

  const load = useCallback(
    () => fetchClip(clipId).then((r) => { setClip(r.clip); setSegments(r.segments) }),
    [clipId],
  )
  useEffect(() => { load() }, [load])

  // Play exactly the segment's span and stop at its end (spec §7: "auto-plays only that span").
  const play = (segment: Segment) => {
    const el = audio.current
    if (!el) return
    setActiveId(segment.id)
    stopAt.current = segment.end_ms / 1000
    el.currentTime = segment.start_ms / 1000
    void el.play()
  }

  const onTimeUpdate = () => {
    const el = audio.current
    if (el && stopAt.current !== null && el.currentTime >= stopAt.current) {
      el.pause()
      stopAt.current = null
    }
  }

  if (!clip) return <p>Loading…</p>

  return (
    <>
      <h2>{clip.title}</h2>

      <audio
        ref={audio}
        src={audioUrl(clipId)}
        onTimeUpdate={onTimeUpdate}
        controls
        preload="metadata"
        style={{ width: '100%', position: 'sticky', top: 0, zIndex: 1 }}
      />

      <SpeakerRoles
        clip={clip}
        segments={segments}
        onChange={(id) => patchReciter(clipId, id).then(load)}
      />

      <SegmentList
        segments={segments}
        activeId={activeId}
        onPlay={play}
        onSave={(id, text) => patchSegment(id, text).then(load)}
      />
    </>
  )
}
```

`preload="metadata"` plus the Task 8 range endpoint is what makes seeking into a 90-minute file instant — the browser fetches only the byte range it needs.

- [ ] **Step 5: Review a real clip end to end**

With `uvicorn` and `npm run dev` both running, open `http://localhost:3000`, click your clip, and check every one of these by hand:

1. The transcript renders in Gurmukhi with a romanized line under each segment.
2. Clicking ▶ on a segment plays **only** that span and stops at its end.
3. Reciter and kathakar segments are visually distinct.
4. Clicking the other speaker in the roles control flips every segment's role.
5. Double-clicking a segment lets you correct it; after saving, it shows `edited`.
6. Seeking deep into the file is instant (check the browser network tab for `206` responses).

This checklist *is* the acceptance test for the slice. If item 2 plays the wrong audio, the `start_ms`/`end_ms` from Task 6 are wrong — go back to the `timing_source` question in Task 7 Step 6 rather than patching the UI.

- [ ] **Step 6: Commit**

```bash
git add web/
git commit -m "feat: review view with span-synced audio playback and inline correction"
```

---

## Deferred, deliberately

Recorded here so the next plan starts from a known position rather than rediscovering these.

- **ASR results are not durable across a worker restart.** `Pipeline._entries` is in-memory; a crash between the `asr` and `align` stages loses the transcript, though `jobs.sarvam_job_id` means it can be re-downloaded rather than re-paid for. Persist the raw Sarvam JSON to `work_dir` and reload it in `_align` when the correction plan lands.
- **One worker process assumed.** `claim_next_job` already uses `FOR UPDATE SKIP LOCKED`, so a second worker is safe at the DB level, but the in-memory stage cache means a clip must finish all four stages in one process. Fix that (previous bullet) before scaling out.
- **`mode="verbatim"` untested.** Sarvam's `transcribe` mode normalises numbers and cleans disfluencies, which may quietly tidy liturgical phrasing. Spec §4 stage 2 already calls for A/B testing denoise; add a `verbatim`-vs-`transcribe` A/B on the same clip once there's a labelled sample to score against.
- **Diarization is beta and may collapse to one speaker.** Role inference degrades to `unknown` and the admin sets roles by hand, which is survivable but annoying. If it happens often, revisit.
- **No settings UI.** Spec §7 wants an admin panel for files-per-day, the denoise toggle, confidence thresholds, and watch folders. Only `files_per_day` is in the `settings` table and it is editable only by SQL; the rest are environment variables requiring a restart. Fold the settings panel into the correction plan, once the confidence thresholds it needs to expose actually exist.
- **Only the database is containerised.** Spec §13 sketches a service per component. This slice runs the worker and API directly on the host under systemd, against a Dockerised Postgres. That is the right amount of infrastructure for one clip at a time; revisit when the RAG bot and `llm-worker` need to scale independently.
- **Verse detection deliberately absent.** The sample transcript proves verses appear inside kathakar segments, so the Nanak Prakash and BaniDB lookups must run on *spans within* segments, not on whole reciter segments. Do not build the verse plan around `speaker_role = 'reciter'`.
