# Katha Pipeline — Correction Slice: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After `assemble`, every clip gets a Claude review pass that flags probable ASR errors with 1–5 Gurmukhi candidates, a correction ledger auto-applies known fixes when the surrounding context matches, and the admin resolves the rest in one click while teaching the ledger.

**Architecture:** A new `correct` stage in the existing stage machine calls an `LLMProvider` (`claude -p` on the Max plan, Anthropic API as fallback) once per window of segments, caches each validated window on disk, persists `flags`, then runs a pure ledger-matching pass (rapidfuzz surface match + cosine on a hashing context embedding) that either auto-applies into `segments.text_pa` with a reversible `corrections` row or attaches a ranked suggestion. FastAPI gains resolve / revert / ledger / settings endpoints; the Next.js review view gains a flag panel, plus `/ledger` and `/settings` pages.

**Tech Stack:** Python 3.13, psycopg 3, Alembic, FastAPI, `rapidfuzz` 3.14, `gurmukhiutils` 0.3, `anthropic` 1.4 SDK, Claude Code CLI 2.1 (`claude -p`), Postgres 17 + pgvector 0.8.5, Next.js 16 + `gurmukhi-utils@3.2.2`.

**Spec:** `docs/superpowers/specs/2026-09-05-katha-correction-slice-design.md`

## Global Constraints

Copied from the spec's verified facts. Do not substitute from memory.

- **`claude -p` structured output:** flags `--output-format json --json-schema <schema>`; stdout is one JSON object; the parsed object is in `structured_output`; `is_error` is `true` on failure with the message in `result`.
- **Never pass `--bare`** — it disables OAuth ("Anthropic auth is strictly ANTHROPIC_API_KEY"). Use `--safe-mode` (disables CLAUDE.md, hooks, plugins, MCP; auth works normally).
- **`--tools ""`** disables every built-in tool. **`--system-prompt-file <path>`** replaces the default system prompt. User prompt goes on **stdin** as UTF-8 bytes (Windows command lines cap at 32 KB; Gurmukhi round-trips byte-exact via `subprocess.run(input=...)`, verified).
- **`--model claude-sonnet-5`** (full id) works on the CLI; the same string is used for the API backend. Default model is `claude-sonnet-5` per spec §10 ("a cheaper, faster model for routine correction"); it is a DB setting, not code.
- **Anthropic SDK structured output:** `client.messages.create(..., output_config={"format": {"type": "json_schema", "schema": {...}}})`; the first `text` block is valid JSON. Schemas carry `"additionalProperties": false`.
- **`gurmukhiutils`:** `from gurmukhiutils.convert import convert; convert(text, "guru_latn_pa")` romanizes; `from gurmukhiutils.unicode import unicode_normalize`. **`rapidfuzz`:** `from rapidfuzz import fuzz; fuzz.ratio(a, b)` is 0–100.
- **pgvector:** the ledger's `context_embedding` is an untyped `vector` column, written as `%s::vector` from the text form `[0.1,0.2,…]` and read back with `::text`. No `register_vector`, no numpy in app code.
- **Gurmukhi is canonical storage.** `romanized` is derived, never typed in.
- **Verses live inside kathakar segments.** Nothing keys on `speaker_role`.
- **All timestamps are integer ms** (unchanged; nothing here touches timing).
- **Every repo function opens and closes its own transaction** (`conn.commit()` on write, `conn.rollback()` after a read). Keep the contract.
- **Windows dev host:** tests must pass on Windows; stub executables are `.bat` on Windows and `#!/bin/sh` elsewhere (the denoiser test's pattern). Console output is cp1252 — never `print` Gurmukhi in tests; assert on values.
- **Env `PYTHONIOENCODING=utf-8`** for any script that prints Gurmukhi during development.

### File Structure

```
src/katha/
  appsettings.py          NEW  typed DB-setting specs + defaults (shared by migration seed, repo, API)
  text.py                 NEW  normalize(), romanize()  (gurmukhiutils)
  embed.py                NEW  Embedder protocol, HashingEmbedder, cosine()
  llm.py                  NEW  LLMProvider, LLMError, ClaudeCliProvider, AnthropicApiProvider, make_provider
  fsutil.py               NEW  write_text_atomic()
  stages/correct.py       NEW  windows, prompts, FLAG_SCHEMA, validate_flags, ProposedFlag (pure)
  stages/ledger.py        NEW  LedgerEntry, Match, surface_score, context_text, match, apply_span, merge_embeddings (pure)
  review.py               NEW  accept_flag / reject_flag / revert_correction (service layer used by the API)
  repo.py                 MOD  settings, flags, corrections, ledger, segment-text accessors; counts in list_clips/list_segments
  worker.py               MOD  `correct` stage, ledger pass, NEXT_STAGE
  config.py               MOD  llm_backend, claude_bin, llm_timeout_seconds, anthropic_api_key, llm_max_tokens
  api/main.py             MOD  new endpoints
migrations/versions/0002_corrections.py   NEW
tests/
  conftest.py             MOD  TABLES, settings reset
  test_appsettings.py test_text.py test_embed.py test_llm.py test_correct.py test_ledger.py
  test_repo_corrections.py test_worker_correct.py test_review.py test_api_corrections.py   NEW
  test_worker.py test_api.py test_schema.py                                                MOD
web/
  lib/types.ts lib/api.ts                MOD
  lib/highlight.tsx                      NEW
  components/FlagPanel.tsx               NEW
  components/SegmentList.tsx             MOD
  app/page.tsx app/layout.tsx            MOD
  app/ledger/page.tsx app/settings/page.tsx  NEW
SETUP.md .env.example pyproject.toml     MOD
```

---

## Task 1: Dependencies, typed settings, schema migration

**Files:**
- Modify: `pyproject.toml`
- Create: `src/katha/appsettings.py`
- Create: `migrations/versions/0002_corrections.py`
- Modify: `src/katha/config.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_appsettings.py`, `tests/test_schema.py`

**Interfaces:**
- Produces: `appsettings.SPECS: dict[str, SettingSpec]`, `appsettings.DEFAULTS: dict[str, str]`, `appsettings.coerce(key, raw) -> int|float|bool|str`, `appsettings.validate(key, value) -> str` (raises `KeyError` on unknown key, `ValueError` on bad value).
- Produces: tables `correction_ledger`, `flags`, `corrections`; `jobs.stage` accepts `'correct'`; `Settings.llm_backend`, `.claude_bin`, `.llm_timeout_seconds`, `.anthropic_api_key`, `.llm_max_tokens`.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml` `dependencies`, add:

```toml
    "rapidfuzz>=3.0",
    "gurmukhiutils>=0.3",
    "anthropic>=1.0",
```

Run: `uv pip install --python .venv/Scripts/python.exe -e ".[dev]"` (already installed in this session; this records them).

- [ ] **Step 2: Write the failing settings-spec test**

`tests/test_appsettings.py`:

```python
import pytest

from katha import appsettings


def test_every_default_round_trips_through_validate():
    for key, default in appsettings.DEFAULTS.items():
        assert appsettings.validate(key, default) == default


def test_coerce_returns_typed_values():
    assert appsettings.coerce("files_per_day", "3") == 3
    assert appsettings.coerce("ledger_context_threshold", "0.8") == 0.8
    assert appsettings.coerce("auto_apply_enabled", "true") is True
    assert appsettings.coerce("auto_apply_enabled", "false") is False
    assert appsettings.coerce("llm_model", "claude-sonnet-5") == "claude-sonnet-5"


def test_validate_rejects_out_of_range_and_wrong_type():
    with pytest.raises(ValueError):
        appsettings.validate("files_per_day", "-1")
    with pytest.raises(ValueError):
        appsettings.validate("ledger_context_threshold", "1.5")
    with pytest.raises(ValueError):
        appsettings.validate("ledger_surface_threshold", "abc")
    with pytest.raises(ValueError):
        appsettings.validate("auto_apply_enabled", "maybe")
    with pytest.raises(ValueError):
        appsettings.validate("llm_model", "   ")


def test_validate_canonicalises_bools_and_rejects_unknown_keys():
    assert appsettings.validate("auto_apply_enabled", "YES") == "true"
    assert appsettings.validate("auto_apply_enabled", "0") == "false"
    with pytest.raises(KeyError):
        appsettings.validate("no_such_key", "1")
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_appsettings.py -q`
Expected: FAIL — `ModuleNotFoundError: katha.appsettings`.

- [ ] **Step 4: Implement `appsettings.py`**

```python
"""Typed specs for the admin-editable rows in the `settings` table.

The migration seeds these rows (values duplicated there on purpose — a
migration must never import application code, or it drifts); a test asserts
the seed equals DEFAULTS so the two can't silently diverge.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SettingSpec:
    key: str
    kind: str  # 'int' | 'float' | 'bool' | 'str'
    default: str
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""


SPECS: dict[str, SettingSpec] = {
    s.key: s
    for s in (
        SettingSpec("files_per_day", "int", "3", 0, 1000,
                    "New clips the worker may start per rolling 24 h."),
        SettingSpec("llm_model", "str", "claude-sonnet-5", help=
                    "Model id for the correction pass (full id, e.g. claude-sonnet-5)."),
        SettingSpec("correct_window_segments", "int", "40", 1, 500,
                    "Segments sent to the LLM per call."),
        SettingSpec("correct_context_segments", "int", "3", 0, 50,
                    "Neighbouring segments shown as context on each side of a window."),
        SettingSpec("ledger_surface_threshold", "float", "88", 0, 100,
                    "Minimum rapidfuzz ratio (0-100) between a flagged span and a ledger surface form."),
        SettingSpec("ledger_context_threshold", "float", "0.8", 0, 1,
                    "Default cosine threshold stored on new ledger entries for auto-apply."),
        SettingSpec("auto_apply_enabled", "bool", "true", help=
                    "When off, ledger matches are only suggested, never applied."),
    )
}
DEFAULTS: dict[str, str] = {k: s.default for k, s in SPECS.items()}

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def coerce(key: str, raw: str) -> int | float | bool | str:
    spec = SPECS[key]
    raw = raw.strip()
    if spec.kind == "int":
        return int(raw)
    if spec.kind == "float":
        return float(raw)
    if spec.kind == "bool":
        low = raw.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"{key}: expected a boolean, got {raw!r}")
    if not raw:
        raise ValueError(f"{key}: must not be empty")
    return raw


def validate(key: str, value: str) -> str:
    """Return the canonical string to store, or raise KeyError / ValueError."""
    spec = SPECS[key]  # KeyError for unknown keys is intentional
    try:
        typed = coerce(key, value)
    except ValueError as exc:
        raise ValueError(f"{key}: {exc}") from None
    if spec.kind in ("int", "float"):
        if spec.minimum is not None and typed < spec.minimum:
            raise ValueError(f"{key}: must be >= {spec.minimum}")
        if spec.maximum is not None and typed > spec.maximum:
            raise ValueError(f"{key}: must be <= {spec.maximum}")
        return str(typed) if spec.kind == "int" else repr(float(typed))
    if spec.kind == "bool":
        return "true" if typed else "false"
    return typed
```

- [ ] **Step 5: Run the settings test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_appsettings.py -q`
Expected: 4 passed. (Note `repr(float("88"))` is `'88.0'` — the round-trip test compares against `DEFAULTS`, so store `ledger_surface_threshold` default as `"88.0"` and `ledger_context_threshold` as `"0.8"`. Fix the default strings, not the validator.)

- [ ] **Step 6: Write the failing schema test**

Append to `tests/test_schema.py`:

```python
@pytest.mark.db
def test_jobs_stage_accepts_correct(db):
    with db.cursor() as cur:
        cur.execute("INSERT INTO clips (source_path, title, sha256) VALUES ('/x', 'x', 'h') RETURNING id")
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
```

- [ ] **Step 7: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schema.py -q`
Expected: FAIL — `CheckViolation` on stage `'correct'`, and `to_regclass` returns None.

- [ ] **Step 8: Write the migration**

`migrations/versions/0002_corrections.py`:

```python
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
```

- [ ] **Step 9: Update `conftest.py`**

Replace the `TABLES` tuple and the settings reset:

```python
TABLES = ("corrections", "flags", "correction_ledger", "words", "segments", "jobs", "audio_files", "clips")
```

and in the `db` fixture teardown, replace the two-line settings reset with:

```python
        cur.execute("DELETE FROM settings")
        cur.executemany(
            "INSERT INTO settings (key, value) VALUES (%s, %s)",
            list(DEFAULTS.items()),
        )
```

with `from katha.appsettings import DEFAULTS` at the top.

- [ ] **Step 10: Extend `config.py`**

Add after `cors_origin`:

```python
    # --- LLM (correction pass) -------------------------------------------
    # 'claude_cli' runs `claude -p` on the Max subscription (the host user must
    # have run `claude login`). 'anthropic_api' bills pay-as-you-go via the SDK.
    llm_backend: str = "claude_cli"
    claude_bin: str = "claude"
    # Per LLM call. A 40-segment window normally returns in well under a minute.
    llm_timeout_seconds: int = 900
    # Only for llm_backend=anthropic_api. Empty = let the SDK read ANTHROPIC_API_KEY.
    anthropic_api_key: str = ""
    llm_max_tokens: int = 16000
```

- [ ] **Step 11: Run the schema test and the whole fast suite**

Run: `.venv/Scripts/python.exe -m pytest -q -m "not heavy"`
Expected: all green (the `_migrate` fixture applies 0002 automatically; previous 101 + 6 new).

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml src/katha/appsettings.py src/katha/config.py migrations/versions/0002_corrections.py tests/conftest.py tests/test_appsettings.py tests/test_schema.py
git commit -m "feat: correction-slice schema, typed settings, and LLM config"
```

---

## Task 2: Text utilities and the hashing embedder

**Files:**
- Create: `src/katha/text.py`, `src/katha/embed.py`
- Test: `tests/test_text.py`, `tests/test_embed.py`

**Interfaces:**
- Produces: `text.normalize(s: str) -> str`, `text.romanize(s: str) -> str`.
- Produces: `embed.Embedder` (Protocol: `name: str`, `dim: int`, `embed(text) -> list[float]`), `embed.HashingEmbedder()`, `embed.cosine(a, b) -> float`, `embed.vector_to_text(v) -> str`, `embed.vector_from_text(s) -> list[float]`.

- [ ] **Step 1: Failing tests**

`tests/test_text.py`:

```python
from katha.text import normalize, romanize


def test_normalize_strips_dandas_punctuation_and_collapses_whitespace():
    assert normalize("ਤੋ ਸੋ  ਨਹੀ ਦਾਤਾ ਕੋਇ ॥") == "ਤੋ ਸੋ ਨਹੀ ਦਾਤਾ ਕੋਇ"
    assert normalize("  ਭਾਈ, ਸੰਤੋਖ ਸਿੰਘ।") == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"


def test_normalize_casefolds_latin_leakage():
    assert normalize("Guru ਗੱਦੀ") == "guru ਗੱਦੀ"


def test_romanize_produces_latin():
    out = romanize("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ")
    assert out.startswith("bh")
    assert "ਭ" not in out


def test_romanize_never_raises_on_odd_input():
    assert romanize("") == ""
    assert isinstance(romanize("Guru ਗੱਦੀ ॥"), str)
```

`tests/test_embed.py`:

```python
import math

import pytest

from katha.embed import HashingEmbedder, cosine, vector_from_text, vector_to_text


def test_embedding_is_unit_length_and_deterministic():
    e = HashingEmbedder()
    v1 = e.embed("ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦੀ ਕਥਾ")
    v2 = e.embed("ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦੀ ਕਥਾ")
    assert v1 == v2
    assert len(v1) == e.dim == 256
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)


def test_similar_contexts_score_higher_than_unrelated_ones():
    e = HashingEmbedder()
    a = e.embed("ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦਾ ਜਨਮ ਤਲਵੰਡੀ ਵਿੱਚ ਹੋਇਆ")
    b = e.embed("ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਜੀ ਦਾ ਜਨਮ ਤਲਵੰਡੀ ਵਿਖੇ ਹੋਇਆ")
    c = e.embed("ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਗਰਮ ਹੈ")
    assert cosine(a, b) > 0.6
    assert cosine(a, c) < 0.3


def test_empty_text_embeds_to_zero_vector_and_cosine_is_zero():
    e = HashingEmbedder()
    z = e.embed("   ")
    assert all(x == 0.0 for x in z)
    assert cosine(z, e.embed("ਕੁਝ")) == 0.0


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])


def test_vector_text_round_trip():
    v = [0.5, -0.25, 1e-7]
    assert vector_from_text(vector_to_text(v)) == pytest.approx(v)
    assert vector_to_text(v).startswith("[") and vector_to_text(v).endswith("]")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_text.py tests/test_embed.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `text.py`**

```python
"""Gurmukhi text helpers shared by the ledger matcher, the embedder and the API.

gurmukhiutils is the Python port of the Shabad OS library the web UI already
uses (gurmukhi-utils@3.2.2), so server- and client-side romanization agree.
"""
import re
import unicodedata

from gurmukhiutils.convert import convert
from gurmukhiutils.unicode import unicode_normalize

# Dandas, Latin punctuation, and the dash family. Gurmukhi letters and matras
# are untouched. Note '॥' (U+0965) and '।' (U+0964) are Devanagari-block
# punctuation that Gurmukhi text uses too.
_PUNCT = re.compile(r"[॥।‖,.;:!?\"'()\[\]{}\-–—]+")


def normalize(s: str) -> str:
    try:
        s = unicode_normalize(s)
    except Exception:  # noqa: BLE001 — a normaliser bug must never break matching
        pass
    s = unicodedata.normalize("NFC", s)
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split()).casefold()


def romanize(s: str) -> str:
    """Display aid only (spec §3.3). Never write the result back as canonical text."""
    if not s:
        return ""
    try:
        return convert(s, "guru_latn_pa")
    except Exception:  # noqa: BLE001
        return s
```

- [ ] **Step 4: Implement `embed.py`**

```python
"""Context embeddings for the correction ledger.

The Embedder protocol is the seam the retrieval slice will plug a real
multilingual model into. HashingEmbedder is the dependency-free default: a
signed character-n-gram hash, L2-normalised, so cosine similarity measures
lexical overlap of the eight-word context around a flagged span — which is
the right fidelity for "is this the same kind of sentence the correction was
learned in". Its `name` is persisted on every ledger row; rows embedded by a
different model can only ever *suggest*, never auto-apply (see stages/ledger).
"""
import hashlib
import math
from typing import Protocol

from katha.text import normalize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    name = "hash-cng-256"
    dim = 256

    def __init__(self, ngrams: tuple[int, ...] = (2, 3, 4)):
        self.ngrams = ngrams

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        t = f" {normalize(text)} "
        if not t.strip():
            return vec
        for n in self.ngrams:
            for i in range(len(t) - n + 1):
                h = hashlib.blake2b(t[i : i + n].encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                vec[idx] += 1.0 if h[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def vector_to_text(v: list[float]) -> str:
    """pgvector's text input form. Written with `%s::vector`."""
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def vector_from_text(s: str) -> list[float]:
    body = s.strip()[1:-1].strip()
    return [float(x) for x in body.split(",")] if body else []
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_text.py tests/test_embed.py -q`
Expected: 9 passed. If `test_similar_contexts_score_higher_than_unrelated_ones` thresholds miss, print the actual cosines once (ASCII only) and set the assertions to a margin that holds, keeping `similar > unrelated` by at least 0.3.

- [ ] **Step 6: Commit**

```bash
git add src/katha/text.py src/katha/embed.py tests/test_text.py tests/test_embed.py
git commit -m "feat: gurmukhi text normalisation/romanization and hashing context embedder"
```

---

## Task 3: `LLMProvider` — `claude -p` and Anthropic API backends

**Files:**
- Create: `src/katha/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `LLMError(RuntimeError)`; `LLMProvider` Protocol with `name: str` and `complete_json(*, system: str, user: str, schema: dict) -> dict`; `ClaudeCliProvider(settings, model, bin_path=None)`; `AnthropicApiProvider(settings, model, client=None)`; `parse_cli_result(payload) -> dict`; `make_provider(settings, model) -> LLMProvider`.

- [ ] **Step 1: Failing tests**

`tests/test_llm.py`:

```python
import json
import os
import sys
from pathlib import Path

import pytest

from katha.config import Settings
from katha.llm import (
    AnthropicApiProvider,
    ClaudeCliProvider,
    LLMError,
    make_provider,
    parse_cli_result,
)

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}

STUB_PY = r'''
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
mode = open(os.path.join(here, "mode.txt"), encoding="utf-8").read().strip()
stdin = sys.stdin.buffer.read().decode("utf-8")
with open(os.path.join(here, "captured.json"), "w", encoding="utf-8") as f:
    json.dump({"argv": sys.argv[1:], "stdin": stdin, "cwd": os.getcwd()}, f, ensure_ascii=False)
if mode == "sleep":
    time.sleep(3); sys.exit(0)
if mode == "exit2":
    sys.stderr.write("boom"); sys.exit(2)
if mode == "garbage":
    sys.stdout.buffer.write(b"not json"); sys.exit(0)
with open(os.path.join(here, "response.json"), encoding="utf-8") as f:
    sys.stdout.buffer.write(f.read().encode("utf-8"))
'''


class CliStub:
    """A fake `claude` binary: a .bat/.sh launcher around a Python script so
    the stub's stdout bytes are exact UTF-8 regardless of the console code page."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        (root / "stub.py").write_text(STUB_PY, encoding="utf-8")
        if os.name == "nt":
            self.bin = root / "claude-stub.bat"
            self.bin.write_text(f'@echo off\r\n"{sys.executable}" "{root / "stub.py"}" %*\r\n')
        else:
            self.bin = root / "claude-stub"
            self.bin.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{root / "stub.py"}" "$@"\n')
            self.bin.chmod(0o755)
        self.mode("ok")
        self.respond({"type": "result", "is_error": False,
                      "structured_output": {"ok": True}, "total_cost_usd": 0.01,
                      "modelUsage": {"claude-sonnet-5": {}}})

    def mode(self, m: str) -> None:
        (self.root / "mode.txt").write_text(m, encoding="utf-8")

    def respond(self, payload: dict) -> None:
        (self.root / "response.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def captured(self) -> dict:
        return json.loads((self.root / "captured.json").read_text(encoding="utf-8"))


@pytest.fixture
def stub(tmp_path):
    return CliStub(tmp_path / "stub")


@pytest.fixture
def cli_settings(tmp_path):
    return Settings(_env_file=None, work_dir=tmp_path / "work", llm_timeout_seconds=1)


def test_cli_provider_returns_structured_output_and_pipes_prompt_on_stdin(stub, cli_settings):
    p = ClaudeCliProvider(cli_settings, model="claude-sonnet-5", bin_path=str(stub.bin))
    out = p.complete_json(system="SYS ਸਿਸਟਮ", user="USER ਵਾਹਿਗੁਰੂ", schema=SCHEMA)
    assert out == {"ok": True}
    cap = stub.captured()
    assert cap["stdin"] == "USER ਵਾਹਿਗੁਰੂ"
    argv = cap["argv"]
    assert argv[:2] == ["-p", "--safe-mode"]
    assert "--bare" not in argv
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5"
    assert argv[argv.index("--tools") + 1] == ""
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    sys_file = Path(argv[argv.index("--system-prompt-file") + 1])
    assert not sys_file.exists()  # cleaned up after the call
    assert Path(cap["cwd"]).resolve() == cli_settings.work_dir.resolve()


def test_cli_provider_system_prompt_file_contained_the_prompt(stub, cli_settings, monkeypatch):
    import katha.llm as llm_mod
    seen = {}
    real_run = llm_mod.subprocess.run

    def spy(cmd, **kw):
        seen["system"] = Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        return real_run(cmd, **kw)

    monkeypatch.setattr(llm_mod.subprocess, "run", spy)
    ClaudeCliProvider(cli_settings, model="m", bin_path=str(stub.bin)).complete_json(
        system="SYS ਸਿਸਟਮ", user="u", schema=SCHEMA)
    assert seen["system"] == "SYS ਸਿਸਟਮ"


@pytest.mark.parametrize("mode, needle", [
    ("exit2", "exited 2"),
    ("garbage", "non-JSON"),
    ("sleep", "timed out"),
])
def test_cli_provider_failure_modes_raise_llm_error(stub, cli_settings, mode, needle):
    stub.mode(mode)
    p = ClaudeCliProvider(cli_settings, model="m", bin_path=str(stub.bin))
    with pytest.raises(LLMError, match=needle):
        p.complete_json(system="s", user="u", schema=SCHEMA)


def test_cli_provider_surfaces_the_clis_own_error_text(stub, cli_settings):
    stub.respond({"type": "result", "is_error": True, "result": "Not logged in · Please run /login"})
    p = ClaudeCliProvider(cli_settings, model="m", bin_path=str(stub.bin))
    with pytest.raises(LLMError, match="Not logged in"):
        p.complete_json(system="s", user="u", schema=SCHEMA)


def test_cli_provider_missing_binary_fails_at_construction(cli_settings, tmp_path):
    with pytest.raises(LLMError, match="claude"):
        ClaudeCliProvider(cli_settings, model="m", bin_path=str(tmp_path / "nope"))


def test_parse_cli_result_requires_structured_output():
    with pytest.raises(LLMError, match="structured_output"):
        parse_cli_result({"type": "result", "is_error": False, "result": "{}"})
    with pytest.raises(LLMError):
        parse_cli_result({"type": "assistant"})
    assert parse_cli_result({"type": "result", "is_error": False,
                             "structured_output": {"a": 1}}) == {"a": 1}


class _Block:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block("text", text)]
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


class FakeClient:
    def __init__(self, resp):
        self.messages = FakeMessages(resp)


def test_api_provider_uses_structured_output_format_and_parses_text(cli_settings):
    client = FakeClient(_Resp('{"ok": true}'))
    p = AnthropicApiProvider(cli_settings, model="claude-sonnet-5", client=client)
    assert p.complete_json(system="S", user="U", schema=SCHEMA) == {"ok": True}
    kw = client.messages.calls[0]
    assert kw["model"] == "claude-sonnet-5"
    assert kw["system"] == "S"
    assert kw["messages"] == [{"role": "user", "content": "U"}]
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert kw["max_tokens"] == cli_settings.llm_max_tokens


@pytest.mark.parametrize("resp, needle", [
    (_Resp("not json"), "non-JSON"),
    (_Resp("{}", stop_reason="max_tokens"), "max_tokens"),
    (_Resp("{}", stop_reason="refusal"), "refus"),
])
def test_api_provider_failure_modes(cli_settings, resp, needle):
    p = AnthropicApiProvider(cli_settings, model="m", client=FakeClient(resp))
    with pytest.raises(LLMError, match=needle):
        p.complete_json(system="S", user="U", schema=SCHEMA)


def test_api_provider_wraps_sdk_errors(cli_settings):
    import anthropic
    import httpx2
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    p = AnthropicApiProvider(cli_settings, model="m", client=FakeClient(err))
    with pytest.raises(LLMError, match="Anthropic API"):
        p.complete_json(system="S", user="U", schema=SCHEMA)


def test_make_provider_selects_backend(cli_settings, stub):
    api = make_provider(Settings(_env_file=None, llm_backend="anthropic_api",
                                 anthropic_api_key="k"), model="m")
    assert api.name == "anthropic_api"
    cli = make_provider(Settings(_env_file=None, claude_bin=str(stub.bin),
                                 work_dir=cli_settings.work_dir), model="m")
    assert cli.name == "claude_cli"
    with pytest.raises(LLMError, match="llm_backend"):
        make_provider(Settings(_env_file=None, llm_backend="gpt"), model="m")
```

- [ ] **Step 2: Run to verify failure** — `.venv/Scripts/python.exe -m pytest tests/test_llm.py -q` → `ModuleNotFoundError: katha.llm`.

- [ ] **Step 3: Implement `llm.py`**

```python
"""The one seam every LLM call goes through (locked decision from the
transcript-slice plan). Two backends, chosen by Settings.llm_backend:

  claude_cli    `claude -p` on the Max subscription. The host user must have
                run `claude login`. Verified flags (2026-09-05):
                  --safe-mode         no CLAUDE.md / hooks / plugins / MCP; OAuth still works
                  --output-format json --json-schema S   -> stdout JSON with `structured_output`
                  --system-prompt-file F                  replaces the default system prompt
                  --tools ""                               no built-in tools
                  --no-session-persistence
                NEVER --bare: it disables OAuth ("auth is strictly ANTHROPIC_API_KEY").
                The user prompt is piped on stdin as UTF-8 (Windows caps argv at 32 KB).
  anthropic_api The official SDK; pay-as-you-go. Structured output via
                output_config.format = json_schema (SDK 1.x).

Nothing outside this module may learn which backend is live.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Protocol

from katha.config import Settings

log = logging.getLogger("katha.llm")


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict: ...


def parse_cli_result(payload: object) -> dict:
    if not isinstance(payload, dict) or payload.get("type") != "result":
        raise LLMError(f"claude -p wrote an unexpected top-level object: {str(payload)[:200]!r}")
    if payload.get("is_error"):
        raise LLMError(f"claude -p reported an error: {payload.get('result')}")
    out = payload.get("structured_output")
    if not isinstance(out, dict):
        raise LLMError(
            "claude -p returned no structured_output — was --json-schema passed and "
            f"did the model comply? result={str(payload.get('result'))[:200]!r}"
        )
    log.info(
        "claude -p ok: cost_usd=%s models=%s turns=%s",
        payload.get("total_cost_usd"), list((payload.get("modelUsage") or {}).keys()),
        payload.get("num_turns"),
    )
    return out


class ClaudeCliProvider:
    name = "claude_cli"

    def __init__(self, settings: Settings, model: str, bin_path: str | None = None):
        wanted = bin_path or settings.claude_bin
        found = shutil.which(wanted)
        if found is None:
            raise LLMError(
                f"claude CLI not found at {wanted!r} (KATHA_CLAUDE_BIN); install Claude "
                "Code and run `claude login` as the worker's user"
            )
        self.bin = found
        self.model = model
        self.settings = settings

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict:
        cwd = self.settings.work_dir
        cwd.mkdir(parents=True, exist_ok=True)
        fd, sys_path = tempfile.mkstemp(dir=cwd, prefix=".sysprompt.", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(system)
            cmd = [
                self.bin, "-p", "--safe-mode",
                "--output-format", "json",
                "--json-schema", json.dumps(schema, ensure_ascii=False),
                "--system-prompt-file", sys_path,
                "--tools", "",
                "--no-session-persistence",
                "--model", self.model,
            ]
            timeout = self.settings.llm_timeout_seconds
            try:
                proc = subprocess.run(
                    cmd, input=user.encode("utf-8"), capture_output=True,
                    timeout=timeout, cwd=cwd,
                )
            except subprocess.TimeoutExpired:
                raise LLMError(f"claude -p timed out after {timeout}s") from None
            except OSError as exc:
                raise LLMError(f"could not run claude -p: {exc}") from exc
        finally:
            try:
                os.unlink(sys_path)
            except OSError:
                pass

        stdout = proc.stdout.decode("utf-8", "replace")
        stderr = proc.stderr.decode("utf-8", "replace")
        if proc.returncode != 0:
            raise LLMError(
                f"claude -p exited {proc.returncode}: {(stderr or stdout).strip()[:500]}"
            )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            raise LLMError(f"claude -p wrote non-JSON output: {stdout[:300]!r}") from None
        return parse_cli_result(payload)


class AnthropicApiProvider:
    name = "anthropic_api"

    def __init__(self, settings: Settings, model: str, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key or None,
                timeout=float(settings.llm_timeout_seconds),
            )
        self.client = client
        self.model = model
        self.max_tokens = settings.llm_max_tokens

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict:
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Anthropic API unreachable: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Anthropic API call failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Anthropic API refused the request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise LLMError(
                f"Anthropic API output truncated at max_tokens={self.max_tokens}; "
                "raise KATHA_LLM_MAX_TOKENS or lower correct_window_segments"
            )
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("Anthropic API returned no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise LLMError(f"Anthropic API returned non-JSON text: {text[:300]!r}") from None


def make_provider(settings: Settings, model: str) -> LLMProvider:
    if settings.llm_backend == "claude_cli":
        return ClaudeCliProvider(settings, model)
    if settings.llm_backend == "anthropic_api":
        return AnthropicApiProvider(settings, model)
    raise LLMError(
        f"unknown llm_backend {settings.llm_backend!r}; expected claude_cli or anthropic_api"
    )
```

- [ ] **Step 4: Run tests** — `.venv/Scripts/python.exe -m pytest tests/test_llm.py -q`, expected all pass (the `sleep` case takes ~3 s on Windows: the orphaned Python child holds the pipe until it exits).

- [ ] **Step 5: Commit**

```bash
git add src/katha/llm.py tests/test_llm.py
git commit -m "feat: LLMProvider with claude -p (Max) and Anthropic API backends"
```

---

## Task 4: `correct` stage — windows, prompts, schema, validation (pure)

**Files:**
- Create: `src/katha/stages/correct.py`
- Test: `tests/test_correct.py`

**Interfaces:**
- Produces: `KINDS`, `FLAG_SCHEMA`, dataclasses `SegmentText(idx, role, text)`, `Window(index, target_idx: tuple[int,...], segments: list[SegmentText])`, `ProposedFlag(segment_idx, span_text, kind, reason, candidates: tuple[str,...])`; functions `make_windows(segments, size, context) -> list[Window]`, `build_system_prompt(gazetteer: list[tuple[str, str]]) -> str`, `build_user_prompt(window) -> str`, `validate_flags(payload, window) -> tuple[list[ProposedFlag], int]`, `flags_to_json(flags) -> str`, `flags_from_json(text) -> list[ProposedFlag]`.

- [ ] **Step 1: Failing tests**

`tests/test_correct.py`:

```python
import json

import pytest

from katha.stages.correct import (
    FLAG_SCHEMA,
    ProposedFlag,
    SegmentText,
    build_system_prompt,
    build_user_prompt,
    flags_from_json,
    flags_to_json,
    make_windows,
    validate_flags,
)


def segs(n):
    return [SegmentText(idx=i, role="kathakar" if i % 2 else "reciter", text=f"ਸ਼ਬਦ{i} ਹੋਰ{i}") for i in range(n)]


def test_windows_cover_every_segment_exactly_once_as_a_target():
    ws = make_windows(segs(10), size=4, context=1)
    assert [w.target_idx for w in ws] == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]
    assert [w.index for w in ws] == [0, 1, 2]
    assert [s.idx for s in ws[0].segments] == [0, 1, 2, 3, 4]
    assert [s.idx for s in ws[1].segments] == [3, 4, 5, 6, 7, 8]
    assert [s.idx for s in ws[2].segments] == [7, 8, 9]


def test_windows_of_empty_input_is_empty():
    assert make_windows([], size=4, context=1) == []


def test_user_prompt_marks_targets_and_context_as_json_lines():
    w = make_windows(segs(3), size=2, context=1)[0]
    lines = [json.loads(l) for l in build_user_prompt(w).splitlines() if l.startswith("{")]
    assert [(l["idx"], l["target"]) for l in lines] == [(0, True), (1, True), (2, False)]
    assert lines[0]["text"] == "ਸ਼ਬਦ0 ਹੋਰ0"
    assert lines[0]["role"] == "reciter"


def test_system_prompt_includes_gazetteer_and_rules():
    p = build_system_prompt([("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", "person"), ("ਤਲਵੰਡੀ", "place")])
    assert "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ" in p and "(person)" in p
    assert "verbatim" in p.lower()
    assert "Gurmukhi" in p
    assert build_system_prompt([]).strip()


def test_schema_is_strict_and_bounded():
    item = FLAG_SCHEMA["properties"]["flags"]["items"]
    assert FLAG_SCHEMA["additionalProperties"] is False
    assert item["additionalProperties"] is False
    assert item["properties"]["candidates"]["maxItems"] == 5
    assert item["properties"]["candidates"]["minItems"] == 1
    assert set(item["required"]) == {"segment_idx", "span_text", "kind", "reason", "candidates"}


def _window():
    return make_windows(segs(3), size=2, context=1)[0]  # targets 0,1; context 2


def test_validate_keeps_good_flags_and_normalises_candidates():
    payload = {"flags": [{
        "segment_idx": 1, "span_text": " ਹੋਰ1 ", "kind": "proper_noun", "reason": "r",
        "candidates": [" ਹੋਰ੧ ", "ਹੋਰ੧", "ਹੋਰ1", "", "ਹੋਰ੨"],
    }]}
    flags, dropped = validate_flags(payload, _window())
    assert dropped == 0
    assert flags == [ProposedFlag(1, "ਹੋਰ1", "proper_noun", "r", ("ਹੋਰ੧", "ਹੋਰ੨"))]


@pytest.mark.parametrize("bad", [
    {"segment_idx": 2, "span_text": "ਹੋਰ2", "kind": "other", "reason": "context-only", "candidates": ["x"]},
    {"segment_idx": 9, "span_text": "ਹੋਰ1", "kind": "other", "reason": "unknown idx", "candidates": ["x"]},
    {"segment_idx": 1, "span_text": "ਨਹੀਂ", "kind": "other", "reason": "not verbatim", "candidates": ["x"]},
    {"segment_idx": 1, "span_text": "", "kind": "other", "reason": "empty span", "candidates": ["x"]},
    {"segment_idx": 1, "span_text": "ਹੋਰ1", "kind": "other", "reason": "no candidate", "candidates": ["ਹੋਰ1", " "]},
    {"segment_idx": "1", "span_text": "ਹੋਰ1", "kind": "other", "reason": "idx not int", "candidates": ["x"]},
    {"segment_idx": True, "span_text": "ਹੋਰ1", "kind": "other", "reason": "bool idx", "candidates": ["x"]},
    "not an object",
])
def test_validate_drops_invalid_items(bad):
    flags, dropped = validate_flags({"flags": [bad]}, _window())
    assert flags == [] and dropped == 1


def test_validate_truncates_to_five_candidates_and_defaults_unknown_kind():
    payload = {"flags": [{"segment_idx": 0, "span_text": "ਸ਼ਬਦ0", "kind": "weird", "reason": "",
                          "candidates": [f"c{i}" for i in range(8)]}]}
    flags, _ = validate_flags(payload, _window())
    assert flags[0].kind == "other"
    assert flags[0].candidates == ("c0", "c1", "c2", "c3", "c4")


def test_validate_tolerates_missing_or_malformed_flags_key():
    assert validate_flags({}, _window()) == ([], 0)
    assert validate_flags({"flags": "nope"}, _window()) == ([], 0)
    assert validate_flags("junk", _window()) == ([], 0)


def test_json_round_trip():
    flags = [ProposedFlag(1, "ਹੋਰ1", "proper_noun", "r", ("ਹੋਰ੧",))]
    assert flags_from_json(flags_to_json(flags)) == flags
```

- [ ] **Step 2: Run to verify failure** — `.venv/Scripts/python.exe -m pytest tests/test_correct.py -q` → module missing.

- [ ] **Step 3: Implement `stages/correct.py`**

```python
"""Pure half of the `correct` stage: how a clip is cut into LLM windows, what
the LLM is asked, and how its answer is validated into ProposedFlags. No I/O,
no DB — the worker owns those (worker._correct)."""
import json
from dataclasses import dataclass

KINDS = ("proper_noun", "verse", "date", "number", "script", "garbled", "other")
MAX_CANDIDATES = 5

FLAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["flags"],
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["segment_idx", "span_text", "kind", "reason", "candidates"],
                "properties": {
                    "segment_idx": {"type": "integer"},
                    "span_text": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "reason": {"type": "string"},
                    "candidates": {
                        "type": "array", "minItems": 1, "maxItems": MAX_CANDIDATES,
                        "items": {"type": "string"},
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True)
class SegmentText:
    idx: int
    role: str
    text: str


@dataclass(frozen=True)
class Window:
    index: int
    target_idx: tuple[int, ...]
    segments: list[SegmentText]  # targets plus context, in transcript order


@dataclass(frozen=True)
class ProposedFlag:
    segment_idx: int
    span_text: str
    kind: str
    reason: str
    candidates: tuple[str, ...]


def make_windows(segments: list[SegmentText], size: int, context: int) -> list[Window]:
    if not segments:
        return []
    size = max(1, size)
    context = max(0, context)
    windows: list[Window] = []
    for w, start in enumerate(range(0, len(segments), size)):
        stop = min(start + size, len(segments))
        lo = max(0, start - context)
        hi = min(len(segments), stop + context)
        windows.append(Window(
            index=w,
            target_idx=tuple(s.idx for s in segments[start:stop]),
            segments=list(segments[lo:hi]),
        ))
    return windows


SYSTEM_PROMPT = """You are proofreading an automatic speech-recognition (ASR) transcript of Sikh katha in Punjabi, written in Gurmukhi script. The discourse expounds Bhai Santokh Singh's Sri Nanak Prakash (the life of Guru Nanak Sahib, in Braj verse written in Gurmukhi). One speaker recites verses (role "reciter"); another explains them in spoken Punjabi (role "kathakar"). Verse lines also get re-quoted inside the commentary.

Your job is to find spans the ASR probably got WRONG and propose fixes. Flag only when you would bet the transcript differs from what was said. Typical errors:
- proper_noun: garbled names of Gurus, Sikhs, places, rulers, books (a Gurmukhi spelling that is not a real name but sounds like one).
- verse: a broken or nonsensical line of Braj verse.
- date: an implausible year, Bikrami/Nanakshahi date, or age.
- number: a number that contradicts the surrounding text.
- script: Latin-script words that should be Gurmukhi (e.g. "Guru" for ਗੁਰੂ).
- garbled: a nonsense word or phrase that is clearly a mishearing.
- other: anything else you are confident is a recognition error.

Do NOT flag: dialect, colloquial grammar, disfluencies, honorific style, spelling variants that are both acceptable, or anything you merely find unusual. Never rewrite whole sentences. Prefer the smallest span that contains the error.

Rules for each flag:
- span_text must be copied VERBATIM from the segment text (exact substring, same codepoints, no added or removed characters).
- Only flag segments marked "target": true. Segments with "target": false are context only.
- candidates: 1 to 5 corrected Gurmukhi spellings of the span, most likely first. Gurmukhi only (except when the correct form genuinely is a Latin word). Never repeat the original span as a candidate.
- reason: one short sentence, in English.

Respond with JSON only, matching the schema you were given: {"flags": [...]}. If nothing needs flagging, respond {"flags": []}.
"""


def build_system_prompt(gazetteer: list[tuple[str, str]]) -> str:
    if not gazetteer:
        return SYSTEM_PROMPT
    lines = "\n".join(f"- {canonical} ({entity_type})" for canonical, entity_type in gazetteer)
    return (
        SYSTEM_PROMPT
        + "\nKnown correct spellings (from the correction ledger). Prefer these forms "
        "when a span looks like one of them:\n" + lines + "\n"
    )


def build_user_prompt(window: Window) -> str:
    targets = set(window.target_idx)
    head = (
        "Transcript segments, one JSON object per line, in order. "
        f"Flag only segments with \"target\": true (indices {window.target_idx[0]}"
        f"-{window.target_idx[-1]}).\n\n"
    )
    body = "\n".join(
        json.dumps(
            {"idx": s.idx, "role": s.role, "target": s.idx in targets, "text": s.text},
            ensure_ascii=False,
        )
        for s in window.segments
    )
    return head + body + "\n"


def validate_flags(payload: object, window: Window) -> tuple[list[ProposedFlag], int]:
    """Keep only flags the worker can act on. Returns (flags, dropped_count)."""
    raw = payload.get("flags") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return [], 0
    text_by_idx = {s.idx: s.text for s in window.segments}
    targets = set(window.target_idx)
    kept: list[ProposedFlag] = []
    dropped = 0
    for item in raw:
        flag = _validate_one(item, targets, text_by_idx)
        if flag is None:
            dropped += 1
        else:
            kept.append(flag)
    return kept, dropped


def _validate_one(item: object, targets: set[int], text_by_idx: dict[int, str]) -> ProposedFlag | None:
    if not isinstance(item, dict):
        return None
    idx = item.get("segment_idx")
    if not isinstance(idx, int) or isinstance(idx, bool) or idx not in targets:
        return None
    span = str(item.get("span_text") or "").strip()
    if not span or span not in text_by_idx[idx]:
        return None
    seen: set[str] = set()
    candidates: list[str] = []
    raw_candidates = item.get("candidates")
    for c in raw_candidates if isinstance(raw_candidates, list) else []:
        c = str(c).strip()
        if c and c != span and c not in seen:
            seen.add(c)
            candidates.append(c)
    if not candidates:
        return None
    kind = item.get("kind")
    if kind not in KINDS:
        kind = "other"
    return ProposedFlag(
        segment_idx=idx,
        span_text=span,
        kind=kind,
        reason=str(item.get("reason") or ""),
        candidates=tuple(candidates[:MAX_CANDIDATES]),
    )


def flags_to_json(flags: list[ProposedFlag]) -> str:
    return json.dumps(
        [{"segment_idx": f.segment_idx, "span_text": f.span_text, "kind": f.kind,
          "reason": f.reason, "candidates": list(f.candidates)} for f in flags],
        ensure_ascii=False,
    )


def flags_from_json(text: str) -> list[ProposedFlag]:
    return [
        ProposedFlag(d["segment_idx"], d["span_text"], d["kind"], d["reason"], tuple(d["candidates"]))
        for d in json.loads(text)
    ]
```

- [ ] **Step 4: Run tests** — `.venv/Scripts/python.exe -m pytest tests/test_correct.py -q`, expected all pass.

- [ ] **Step 5: Commit**

```bash
git add src/katha/stages/correct.py tests/test_correct.py
git commit -m "feat: correct-stage windows, prompts, schema, and flag validation"
```

---

## Task 5: Ledger matching (pure)

**Files:**
- Create: `src/katha/stages/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Produces: `LedgerEntry(id, canonical, surface_forms: tuple[str,...], embedding: tuple[float,...] | None, embedding_model: str | None, threshold: float)`, `Match(entry, surface_score, context_similarity: float | None, decision: 'auto'|'suggest')`, `surface_score(span, entry) -> float`, `context_text(text, span, words=8) -> str`, `match(span, context_vec, embedder_name, entries, surface_threshold) -> Match | None`, `apply_span(text, span, replacement) -> str | None`, `merge_embeddings(old, new) -> list[float]`.

- [ ] **Step 1: Failing tests**

`tests/test_ledger.py`:

```python
import math

import pytest

from katha.embed import HashingEmbedder
from katha.stages.ledger import (
    LedgerEntry,
    apply_span,
    context_text,
    match,
    merge_embeddings,
    surface_score,
)

E = HashingEmbedder()


def entry(canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", forms=("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ",), ctx=None, model=E.name, threshold=0.8, id=1):
    return LedgerEntry(id=id, canonical=canonical, surface_forms=tuple(forms),
                       embedding=tuple(E.embed(ctx)) if ctx else None,
                       embedding_model=model if ctx else None, threshold=threshold)


def test_surface_score_uses_best_of_canonical_and_surface_forms():
    e = entry()
    assert surface_score("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", e) == 100.0
    assert surface_score("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ।", e) == 100.0   # punctuation ignored
    assert 80 < surface_score("ਭਾਈ ਸੰਤੋਕ ਸਿੰਗ", e) < 100
    assert surface_score("ਤਲਵੰਡੀ", e) < 40


def test_context_text_returns_words_around_the_span_excluding_it():
    text = "ਕੱਲ ਆਪ ਜੀ ਨੇ ਸਰਵਣ ਕੀਤਾ ਸੀ ਮੰਗਲਾਚਰਨ ਚਰਨ ਨਾਮ ਗੁਰੂ ਦੇ ਚਰਨ"
    ctx = context_text(text, "ਸੀ ਮੰਗਲਾਚਰਨ", words=2)
    assert ctx == "ਸਰਵਣ ਕੀਤਾ ਚਰਨ ਨਾਮ"
    assert context_text(text, "ਨਹੀਂ ਮੌਜੂਦ") == ""
    assert context_text("ਸਿਰਫ਼ ਸਪੈਨ", "ਸਿਰਫ਼ ਸਪੈਨ") == ""


CTX = "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ"


def test_match_auto_when_surface_and_context_both_pass():
    e = entry(ctx="ਕਵੀ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ")
    m = match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", E.embed(context_text(CTX.replace("ਸਿੰਘ", "ਸਿੰਗ"), "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ")),
              E.name, [e], surface_threshold=88)
    assert m is not None and m.decision == "auto"
    assert m.entry.id == 1 and m.surface_score == 100.0
    assert m.context_similarity is not None and m.context_similarity >= 0.8


def test_match_suggests_when_context_is_dissimilar_or_missing():
    e = entry(ctx="ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਗਰਮ ਹੈ ਅਤੇ ਧੁੱਪ ਤੇਜ਼ ਹੈ")
    m = match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", E.embed("ਕਵੀ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ"), E.name, [e], 88)
    assert m is not None and m.decision == "suggest"
    m2 = match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", None, E.name, [entry()], 88)
    assert m2 is not None and m2.decision == "suggest" and m2.context_similarity is None


def test_match_never_autos_across_embedding_models():
    e = entry(ctx="ਕਵੀ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ", model="bge-m3")
    m = match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", E.embed("ਕਵੀ ਜੀ ਨੇ ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਲਿਖਿਆ"), E.name, [e], 88)
    assert m is not None and m.decision == "suggest" and m.context_similarity is None


def test_match_returns_none_below_surface_threshold_or_when_span_already_canonical():
    assert match("ਤਲਵੰਡੀ", None, E.name, [entry()], 88) is None
    assert match("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", None, E.name, [entry()], 88) is None
    assert match("x", None, E.name, [], 88) is None


def test_match_picks_the_best_surface_score():
    a = entry(canonical="ਸੰਤੋਖ", forms=("ਸੰਤੋਕ",), id=1)
    b = entry(canonical="ਸੰਤੋਸ਼", forms=("ਸੰਤੋਖ ਸਿੰਗ",), id=2)
    m = match("ਸੰਤੋਖ ਸਿੰਗ", None, E.name, [a, b], 88)
    assert m is not None and m.entry.id == 2


def test_apply_span_replaces_first_occurrence_only():
    assert apply_span("ਏ ਬੀ ਏ", "ਏ", "ਸੀ") == "ਸੀ ਬੀ ਏ"
    assert apply_span("ਏ ਬੀ", "ਡੀ", "ਸੀ") is None
    assert apply_span("ਏ ਬੀ", "", "ਸੀ") is None


def test_merge_embeddings_is_a_unit_mean():
    old = [1.0, 0.0]
    new = [0.0, 1.0]
    merged = merge_embeddings(old, new)
    assert merged == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert merge_embeddings(None, new) == new
```

- [ ] **Step 2: Run to verify failure** — module missing.

- [ ] **Step 3: Implement `stages/ledger.py`**

```python
"""Pure ledger matching (spec §6). Two independent gates:

  1. surface:  rapidfuzz ratio between the flagged span and the entry's
               canonical form or any recorded misrecognition >= threshold
  2. context:  cosine between the embedding of the words around the span and
               the entry's stored context embedding >= the entry's own threshold

Both pass -> 'auto' (the worker applies it and logs a reversible correction).
Only (1) -> 'suggest' (the entry's canonical form is ranked first in the UI).
Entries embedded by a different model than the one running can only suggest:
a model swap must never produce silent bad auto-applies.
"""
import math
from dataclasses import dataclass

from rapidfuzz import fuzz

from katha.embed import cosine
from katha.text import normalize


@dataclass(frozen=True)
class LedgerEntry:
    id: int
    canonical: str
    surface_forms: tuple[str, ...]
    embedding: tuple[float, ...] | None
    embedding_model: str | None
    threshold: float


@dataclass(frozen=True)
class Match:
    entry: LedgerEntry
    surface_score: float
    context_similarity: float | None
    decision: str  # 'auto' | 'suggest'


def surface_score(span: str, entry: LedgerEntry) -> float:
    s = normalize(span)
    if not s:
        return 0.0
    forms = (entry.canonical, *entry.surface_forms)
    return max((fuzz.ratio(s, normalize(f)) for f in forms if normalize(f)), default=0.0)


def context_text(text: str, span: str, words: int = 8) -> str:
    """The `words` whitespace tokens on each side of the span's first
    occurrence, span excluded. Empty when the span is absent or alone."""
    pos = text.find(span) if span else -1
    if pos < 0:
        return ""
    before = text[:pos].split()[-words:]
    after = text[pos + len(span):].split()[:words]
    return " ".join(before + after)


def match(
    span: str,
    context_vec: list[float] | None,
    embedder_name: str,
    entries: list[LedgerEntry],
    surface_threshold: float,
) -> Match | None:
    span_norm = normalize(span)
    best: tuple[float, LedgerEntry] | None = None
    for e in entries:
        if normalize(e.canonical) == span_norm:
            continue  # nothing to correct
        score = surface_score(span, e)
        if score >= surface_threshold and (best is None or score > best[0]):
            best = (score, e)
    if best is None:
        return None
    score, e = best
    similarity: float | None = None
    if (
        context_vec is not None
        and e.embedding is not None
        and e.embedding_model == embedder_name
        and len(e.embedding) == len(context_vec)
    ):
        similarity = cosine(list(e.embedding), context_vec)
    decision = "auto" if similarity is not None and similarity >= e.threshold else "suggest"
    return Match(entry=e, surface_score=float(score), context_similarity=similarity, decision=decision)


def apply_span(text: str, span: str, replacement: str) -> str | None:
    if not span or span not in text:
        return None
    return text.replace(span, replacement, 1)


def merge_embeddings(old: list[float] | None, new: list[float]) -> list[float]:
    if not old or len(old) != len(new):
        return list(new)
    summed = [a + b for a, b in zip(old, new)]
    norm = math.sqrt(sum(v * v for v in summed))
    return [v / norm for v in summed] if norm else list(new)
```

- [ ] **Step 4: Run tests** — `.venv/Scripts/python.exe -m pytest tests/test_ledger.py -q`. If `test_match_auto_when_surface_and_context_both_pass` fails on the 0.8 threshold, the two contexts differ only by the span's absence, so the cosine should be well above 0.8; investigate `context_text` before touching the threshold.

- [ ] **Step 5: Commit**

```bash
git add src/katha/stages/ledger.py tests/test_ledger.py
git commit -m "feat: pure ledger matching — surface fuzz, context cosine, apply/merge helpers"
```

---

## Task 6: Repository — settings, flags, corrections, ledger

**Files:**
- Modify: `src/katha/repo.py`
- Create: `src/katha/fsutil.py`
- Test: `tests/test_repo_corrections.py`

**Interfaces:**
- Produces (settings): `get_setting(conn, key, default: str) -> str`, `get_setting_float`, `get_setting_bool`, `get_all_settings(conn) -> dict[str,str]`, `set_setting(conn, key, value)`.
- Produces (segments): `list_segment_texts(conn, clip_id) -> list[dict(id, idx, speaker_role, text_pa)]`, `get_segment(conn, segment_id) -> dict | None`.
- Produces (flags): `FlagRow = tuple[int, str, str, str, list[str]]` (segment_id, span_text, kind, reason, candidates); `save_flags(conn, clip_id, rows) -> int`; `list_open_flags(conn, clip_id) -> list[dict]` (joins `text_pa`, `segment_id`); `get_flag(conn, flag_id) -> dict | None` (joins `text_pa`, `clip_id`); `set_flag_suggestion(conn, flag_id, ledger_id, candidates)`; `resolve_flag(conn, flag_id, status, resolved_text=None, ledger_id=None)`; `reopen_flag(conn, flag_id)`.
- Produces (corrections): `add_correction(conn, *, segment_id, flag_id, ledger_id, source, before, after, similarity) -> int`; `get_correction(conn, id) -> dict | None`; `mark_reverted(conn, id)`; `apply_auto_correction(conn, *, flag_id, segment_id, ledger_id, new_text, before, after, similarity) -> int` (one transaction).
- Produces (ledger): `ledger_entries(conn) -> list[LedgerEntry]`; `list_ledger(conn, q: str | None, include_retired: bool) -> list[dict]`; `get_ledger(conn, id) -> dict | None`; `upsert_ledger(conn, *, canonical, romanized, surface_form, entity_type, english_gloss, embedding, embedding_model, threshold, created_by='admin') -> int`; `update_ledger(conn, id, **fields) -> bool`; `bump_ledger_use(conn, id)`; `top_gazetteer(conn, limit=200) -> list[tuple[str,str]]`.
- Modifies: `list_clips` adds `open_flags`, `auto_applied`; `list_segments` adds `flags` and `corrections` JSON arrays.
- Produces: `fsutil.write_text_atomic(path: Path, text: str)`.

- [ ] **Step 1: Failing tests**

`tests/test_repo_corrections.py`:

```python
from pathlib import Path

import pytest

from katha import repo
from katha.embed import HashingEmbedder
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ", start_s=3.5, end_s=5.5),
]
WORDS = [
    AlignedWord("ਤੋ", 2.1, 2.3, 0.9), AlignedWord("ਸੋ", 2.4, 2.9, 0.9),
    AlignedWord("ਭਾਈ", 3.6, 3.8, 0.6), AlignedWord("ਸੰਤੋਖ", 3.9, 4.4, 0.6),
    AlignedWord("ਸਿੰਗ", 4.5, 4.9, 0.6), AlignedWord("ਜੀ", 5.0, 5.4, 0.6),
]
E = HashingEmbedder()


@pytest.fixture
def clip(db):
    cid = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    repo.save_segments(db, cid, assemble.assemble_segments(ENTRIES, WORDS), reciter_id="1")
    return cid


def seg_ids(db, clip_id):
    return [r["id"] for r in repo.list_segment_texts(db, clip_id)]


@pytest.mark.db
def test_settings_round_trip_and_typed_getters(db):
    assert repo.get_setting(db, "llm_model", "x") == "claude-sonnet-5"
    assert repo.get_setting(db, "missing", "dflt") == "dflt"
    assert repo.get_setting_float(db, "ledger_context_threshold", 0.0) == 0.8
    assert repo.get_setting_bool(db, "auto_apply_enabled", False) is True
    repo.set_setting(db, "auto_apply_enabled", "false")
    assert repo.get_setting_bool(db, "auto_apply_enabled", True) is False
    assert repo.get_all_settings(db)["auto_apply_enabled"] == "false"


@pytest.mark.db
def test_save_flags_replaces_open_flags_but_keeps_resolved_ones(db, clip):
    s1, s2 = seg_ids(db, clip)
    n = repo.save_flags(db, clip, [(s2, "ਸਿੰਗ", "proper_noun", "r", ["ਸਿੰਘ"]),
                                   (s1, "ਸੋ", "other", "r2", ["ਸੌ"])])
    assert n == 2
    open_ = repo.list_open_flags(db, clip)
    assert [(f["segment_id"], f["span_text"], f["candidates"]) for f in open_] == [
        (s1, "ਸੋ", ["ਸੌ"]), (s2, "ਸਿੰਗ", ["ਸਿੰਘ"])]
    assert open_[1]["text_pa"] == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ"
    repo.resolve_flag(db, open_[1]["id"], "accepted", resolved_text="ਸਿੰਘ")
    # re-running the stage: the open flag is replaced; the same span on the same
    # segment is NOT re-inserted because a resolved flag for it already exists
    n = repo.save_flags(db, clip, [(s2, "ਸਿੰਗ", "proper_noun", "r", ["ਸਿੰਘ"]),
                                   (s1, "ਤੋ", "other", "r3", ["ਤੌ"])])
    assert n == 1
    assert [f["span_text"] for f in repo.list_open_flags(db, clip)] == ["ਤੋ"]
    flag = repo.get_flag(db, open_[1]["id"])
    assert flag["status"] == "accepted" and flag["clip_id"] == clip


@pytest.mark.db
def test_flag_suggestion_reopen_and_segment_accessors(db, clip):
    s1, s2 = seg_ids(db, clip)
    repo.save_flags(db, clip, [(s2, "ਸਿੰਗ", "proper_noun", "r", ["ਸਿੰਘ", "ਸਿੰਘ੨"])])
    fid = repo.list_open_flags(db, clip)[0]["id"]
    lid = repo.upsert_ledger(db, canonical="ਸਿੰਘ", romanized="siṅgh", surface_form="ਸਿੰਗ",
                             entity_type="term", english_gloss=None, embedding=None,
                             embedding_model=None, threshold=0.8)
    repo.set_flag_suggestion(db, fid, lid, ["ਸਿੰਘ", "ਸਿੰਘ੨"])
    f = repo.get_flag(db, fid)
    assert f["ledger_id"] == lid and f["candidates"] == ["ਸਿੰਘ", "ਸਿੰਘ੨"]
    repo.resolve_flag(db, fid, "rejected")
    assert repo.get_flag(db, fid)["status"] == "rejected"
    repo.reopen_flag(db, fid)
    f = repo.get_flag(db, fid)
    assert f["status"] == "open" and f["resolved_text"] is None and f["resolved_at"] is None
    assert repo.get_segment(db, s2)["text_pa"] == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ"
    assert repo.get_segment(db, 999999) is None


@pytest.mark.db
def test_apply_auto_correction_is_one_transaction_and_shows_up_in_reads(db, clip):
    s1, s2 = seg_ids(db, clip)
    repo.save_flags(db, clip, [(s2, "ਸਿੰਗ", "proper_noun", "r", ["ਸਿੰਘ"])])
    fid = repo.list_open_flags(db, clip)[0]["id"]
    lid = repo.upsert_ledger(db, canonical="ਸਿੰਘ", romanized="siṅgh", surface_form="ਸਿੰਗ",
                             entity_type="term", english_gloss=None,
                             embedding=E.embed("ਭਾਈ ਸੰਤੋਖ ਜੀ"), embedding_model=E.name, threshold=0.8)
    cid = repo.apply_auto_correction(db, flag_id=fid, segment_id=s2, ledger_id=lid,
                                     new_text="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ", before="ਸਿੰਗ", after="ਸਿੰਘ",
                                     similarity=0.93)
    seg = repo.get_segment(db, s2)
    assert seg["text_pa"] == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ" and seg["edited"] is True
    c = repo.get_correction(db, cid)
    assert (c["source"], c["before_text"], c["after_text"], c["flag_id"], c["ledger_id"]) == \
        ("auto", "ਸਿੰਗ", "ਸਿੰਘ", fid, lid)
    assert c["context_similarity"] == pytest.approx(0.93)
    assert repo.get_flag(db, fid)["status"] == "auto_applied"
    led = repo.get_ledger(db, lid)
    assert led["times_applied"] == 1 and led["last_used_at"] is not None
    segs = repo.list_segments(db, clip)
    assert segs[1]["flags"][0]["status"] == "auto_applied"
    assert segs[1]["corrections"][0]["after_text"] == "ਸਿੰਘ"
    clips = repo.list_clips(db)
    assert clips[0]["open_flags"] == 0 and clips[0]["auto_applied"] == 1
    repo.mark_reverted(db, cid)
    assert repo.get_correction(db, cid)["reverted_at"] is not None
    assert repo.list_clips(db)[0]["auto_applied"] == 0


@pytest.mark.db
def test_ledger_upsert_merges_surface_forms_and_embeddings(db):
    v1 = E.embed("ਕਵੀ ਜੀ ਨੇ ਲਿਖਿਆ")
    lid = repo.upsert_ledger(db, canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", romanized="bhai", surface_form="ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ",
                             entity_type="person", english_gloss="the poet", embedding=v1,
                             embedding_model=E.name, threshold=0.8)
    v2 = E.embed("ਸੂਰਜ ਪ੍ਰਕਾਸ਼ ਦੇ ਰਚੇਤਾ")
    lid2 = repo.upsert_ledger(db, canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", romanized="bhai", surface_form="ਭਾਈ ਸੰਤੋਕ ਸਿੰਘ",
                              entity_type="person", english_gloss=None, embedding=v2,
                              embedding_model=E.name, threshold=0.5)
    assert lid2 == lid
    entries = repo.ledger_entries(db)
    assert len(entries) == 1
    e = entries[0]
    assert set(e.surface_forms) == {"ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "ਭਾਈ ਸੰਤੋਕ ਸਿੰਘ"}
    assert e.threshold == pytest.approx(0.8)          # an existing threshold is kept
    assert e.embedding_model == E.name and len(e.embedding) == E.dim
    assert e.embedding != tuple(v1) and e.embedding != tuple(v2)   # merged
    row = repo.get_ledger(db, lid)
    assert row["english_gloss"] == "the poet"          # not overwritten by None
    assert repo.top_gazetteer(db) == [("ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", "person")]
    assert repo.list_ledger(db, q="santokh", include_retired=False) == []
    assert [r["id"] for r in repo.list_ledger(db, q="bhai", include_retired=False)] == [lid]
    assert [r["id"] for r in repo.list_ledger(db, q="ਸੰਤੋਕ", include_retired=False)] == [lid]
    assert repo.update_ledger(db, lid, retired=True, confidence_threshold=0.9) is True
    assert repo.ledger_entries(db) == []
    assert repo.top_gazetteer(db) == []
    assert repo.list_ledger(db, q=None, include_retired=False) == []
    assert repo.list_ledger(db, q=None, include_retired=True)[0]["confidence_threshold"] == pytest.approx(0.9)
    assert repo.update_ledger(db, 424242, retired=False) is False
    assert repo.update_ledger(db, lid) is False


@pytest.mark.db
def test_upsert_with_embedding_from_another_model_replaces_instead_of_merging(db):
    lid = repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="talvandi", surface_form="ਤਲਬੰਡੀ",
                             entity_type="place", english_gloss=None, embedding=[1.0, 0.0],
                             embedding_model="old-2", threshold=0.8)
    v = E.embed("ਜਨਮ ਅਸਥਾਨ")
    repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="talvandi", surface_form="ਤਲਵੰਢੀ",
                       entity_type="place", english_gloss=None, embedding=v,
                       embedding_model=E.name, threshold=0.8)
    e = repo.ledger_entries(db)[0]
    assert e.id == lid and e.embedding_model == E.name and e.embedding == pytest.approx(tuple(v))


def test_write_text_atomic(tmp_path):
    from katha.fsutil import write_text_atomic
    target = tmp_path / "x.json"
    write_text_atomic(target, "ਪਹਿਲਾ")
    write_text_atomic(target, "ਦੂਜਾ")
    assert target.read_text(encoding="utf-8") == "ਦੂਜਾ"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
```

- [ ] **Step 2: Run to verify failure** — attribute errors on `repo`.

- [ ] **Step 3: Implement `fsutil.py`**

```python
import os
import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a same-directory temp file + os.replace so a reader only ever
    sees the fully-old or fully-new file (same rationale as worker._save_words)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
```

- [ ] **Step 4: Add the repo functions**

Append to `src/katha/repo.py` (add `import json` and `from katha.embed import vector_from_text, vector_to_text`, `from katha.stages.ledger import LedgerEntry, merge_embeddings` at the top):

```python
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

def _insert_correction(cur, *, segment_id, flag_id, ledger_id, source, before, after, similarity) -> int:
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
            "UPDATE segments SET text_pa = %s, edited = true WHERE id = %s", (new_text, segment_id)
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
            id=r["id"], canonical=r["canonical_gurmukhi"], surface_forms=tuple(r["surface_forms"]),
            embedding=tuple(vector_from_text(r["context_embedding"])) if r["context_embedding"] else None,
            embedding_model=r["embedding_model"], threshold=float(r["confidence_threshold"]),
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
    embedding (same model) or replace it (different model)."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_LEDGER_COLS} FROM correction_ledger WHERE canonical_gurmukhi = %s FOR UPDATE",
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
            old_vec = vector_from_text(row["context_embedding"]) if row["context_embedding"] else None
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
```

Then modify `list_clips` to add two correlated counts after `loose_count`:

```sql
                   (SELECT count(*) FROM flags f JOIN segments s ON s.id = f.segment_id
                     WHERE s.clip_id = c.id AND f.status = 'open') AS open_flags,
                   (SELECT count(*) FROM corrections x JOIN segments s ON s.id = x.segment_id
                     WHERE s.clip_id = c.id AND x.source = 'auto' AND x.reverted_at IS NULL) AS auto_applied
```

and `list_segments` to add, after the `words` aggregate:

```sql
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
```

- [ ] **Step 5: Run** — `.venv/Scripts/python.exe -m pytest tests/test_repo_corrections.py tests/test_repo.py tests/test_api.py -q`. All green. If psycopg complains about the `text[]` parameter for `surface_forms`, pass a Python `list[str]` (psycopg 3 adapts lists to arrays natively).

- [ ] **Step 6: Commit**

```bash
git add src/katha/repo.py src/katha/fsutil.py tests/test_repo_corrections.py
git commit -m "feat: repo layer for settings, flags, corrections, and the correction ledger"
```

---

## Task 7: Worker — the `correct` stage and the ledger pass

**Files:**
- Modify: `src/katha/worker.py`
- Modify: `tests/test_worker.py` (stage sequence, fixture)
- Test: `tests/test_worker_correct.py`

**Interfaces:**
- Consumes: `make_windows`, `build_system_prompt`, `build_user_prompt`, `validate_flags`, `flags_to_json`, `flags_from_json`, `FLAG_SCHEMA`, `SegmentText` (Task 4); `match`, `context_text`, `apply_span` (Task 5); `make_provider` (Task 3); `HashingEmbedder` (Task 2); repo functions (Task 6); `write_text_atomic`.
- Produces: `Pipeline(settings, conn, aligner=None, asr=None, llm=None, embedder=None)`; `NEXT_STAGE["assemble"] == "correct"`, `NEXT_STAGE["correct"] == "complete"`; `Pipeline._correct(job)`, `Pipeline._apply_ledger(clip_id) -> tuple[int, int]` (auto_applied, suggested).

- [ ] **Step 1: Update the existing worker tests**

In `tests/test_worker.py`, add a `FakeLlm` and pass it in the fixture:

```python
class FakeLlm:
    name = "fake"

    def __init__(self, flags=None):
        self.flags = flags or []
        self.calls = 0

    def complete_json(self, *, system, user, schema):
        self.calls += 1
        return {"flags": self.flags}
```

In the `pipeline` fixture: `return worker.Pipeline(settings, db, aligner=FakeAligner(), asr=FakeAsr(), llm=FakeLlm())`.

Update `test_run_once_walks_a_clip_through_every_stage`'s expected list to:

```python
    assert stages == [
        ("asr", "queued"),
        ("align", "queued"),
        ("assemble", "queued"),
        ("correct", "queued"),
        ("complete", "done"),
    ]
```

Search the file for any other assertion that a clip ends at `("complete", "done")` after exactly four `run_once()` calls or that counts stages, and add one for `correct`. Run: `.venv/Scripts/python.exe -m pytest tests/test_worker.py -q` → the sequence test fails (still 4 stages).

- [ ] **Step 2: Write the new failing tests**

`tests/test_worker_correct.py`:

```python
import json
from pathlib import Path

import pytest

from katha import repo, worker
from katha.embed import HashingEmbedder
from katha.stages.prepare import PrepareResult
from katha.stages.types import AlignedWord, DiarizedEntry
from tests.test_worker import FakeAligner, FakeAsr, FakeLlm, _write_sarvam_json  # noqa: F401

ENTRIES = [
    DiarizedEntry(speaker_id="1", text="ਤੋ ਸੋ", start_s=2.0, end_s=3.0),
    DiarizedEntry(speaker_id="0", text="ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ", start_s=3.5, end_s=5.5),
    DiarizedEntry(speaker_id="0", text="ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦਾ ਜਨਮ ਤਲਬੰਡੀ ਵਿੱਚ ਹੋਇਆ", start_s=6.0, end_s=9.0),
]
WORDS = [AlignedWord(w, 2.0 + i * 0.3, 2.2 + i * 0.3, 0.9)
         for i, w in enumerate(" ".join(e.text for e in ENTRIES).split())]
E = HashingEmbedder()


class Asr(FakeAsr):
    def transcribe(self, wav, out_dir):
        self.calls += 1
        _write_sarvam_json(out_dir, ENTRIES)
        return ENTRIES, "sarvam-job-1"


class Aligner(FakeAligner):
    def align(self, wav, text):
        self.text = text
        return WORDS


def make_pipeline(db, settings, tmp_path, monkeypatch, llm):
    monkeypatch.setattr(settings, "work_dir", tmp_path / "work")
    monkeypatch.setattr(
        worker, "prepare_clip",
        lambda source, work_dir, s: PrepareResult(
            native=work_dir / "native.wav", denoised=None, asr_ready=work_dir / "asr.wav",
            duration_seconds=90.0, native_sample_rate=48000),
    )
    return worker.Pipeline(settings, db, aligner=Aligner(), asr=Asr(), llm=llm, embedder=E)


def run_to_end(db, pipeline):
    clip_id = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    while pipeline.run_once():
        pass
    return clip_id


FLAGS = [
    {"segment_idx": 1, "span_text": "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "kind": "proper_noun", "reason": "name",
     "candidates": ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"]},
    {"segment_idx": 2, "span_text": "ਤਲਬੰਡੀ", "kind": "place", "reason": "place",
     "candidates": ["ਤਲਵੰਡੀ"]},
    {"segment_idx": 2, "span_text": "ਨਹੀਂ ਹੈ", "kind": "garbled", "reason": "not verbatim -> dropped",
     "candidates": ["x"]},
]


@pytest.mark.db
def test_correct_stage_persists_validated_flags_and_completes(db, settings, tmp_path, monkeypatch):
    llm = FakeLlm(FLAGS)
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, llm)
    clip_id = run_to_end(db, pipeline)
    with db.cursor() as cur:
        cur.execute("SELECT stage, status, last_error FROM jobs WHERE clip_id = %s", (clip_id,))
        assert tuple(cur.fetchone().values()) == ("complete", "done", None)
    db.rollback()
    open_ = repo.list_open_flags(db, clip_id)
    assert [(f["span_text"], f["kind"], f["candidates"]) for f in open_] == [
        ("ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "proper_noun", ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"]),
        ("ਤਲਬੰਡੀ", "other", ["ਤਲਵੰਡੀ"]),     # 'place' is not a flag kind -> 'other'
    ]
    assert llm.calls == 1  # 3 segments, window of 40 -> one call
    cache = tmp_path / "work" / str(clip_id) / "flags" / "0.json"
    assert cache.exists() and len(json.loads(cache.read_text(encoding="utf-8"))) == 2


@pytest.mark.db
def test_correct_stage_windows_by_setting_and_reuses_cached_windows_on_retry(db, settings, tmp_path, monkeypatch):
    repo.set_setting(db, "correct_window_segments", "1")
    repo.set_setting(db, "correct_context_segments", "0")

    class Flaky(FakeLlm):
        def complete_json(self, *, system, user, schema):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transient LLM outage")
            return {"flags": []}

    llm = Flaky()
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, llm)
    monkeypatch.setattr(settings, "retry_backoff_seconds", 0)
    clip_id = run_to_end(db, pipeline)
    with db.cursor() as cur:
        cur.execute("SELECT stage, status FROM jobs WHERE clip_id = %s", (clip_id,))
        assert tuple(cur.fetchone().values()) == ("complete", "done")
    db.rollback()
    # 3 windows; call 2 failed; retry re-ran only windows 1 and 2 -> 5 calls, not 6
    assert llm.calls == 5
    assert sorted(p.name for p in (tmp_path / "work" / str(clip_id) / "flags").iterdir()) == \
        ["0.json", "1.json", "2.json"]


@pytest.mark.db
def test_ledger_pass_auto_applies_when_context_matches_and_suggests_otherwise(db, settings, tmp_path, monkeypatch):
    # Learned in a near-identical context -> auto. Learned elsewhere -> suggest only.
    repo.upsert_ledger(db, canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", romanized="bhai santokh singh",
                       surface_form="ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", entity_type="person", english_gloss=None,
                       embedding=E.embed("ਕਵੀ ਜੀ ਨੇ ਲਿਖਿਆ"), embedding_model=E.name, threshold=0.8)
    repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="talvandi", surface_form="ਤਲਬੰਡੀ",
                       entity_type="place", english_gloss=None,
                       embedding=E.embed("ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਗਰਮ ਹੈ ਅਤੇ ਧੁੱਪ ਤੇਜ਼ ਹੈ"),
                       embedding_model=E.name, threshold=0.8)
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, FakeLlm(FLAGS))
    clip_id = run_to_end(db, pipeline)

    segs = repo.list_segments(db, clip_id)
    assert segs[1]["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ ਨੇ ਲਿਖਿਆ"
    assert segs[1]["edited"] is True
    assert segs[1]["flags"][0]["status"] == "auto_applied"
    assert segs[1]["corrections"][0]["source"] == "auto"
    assert segs[1]["corrections"][0]["context_similarity"] >= 0.8

    assert segs[2]["text_pa"] == "ਗੁਰੂ ਨਾਨਕ ਸਾਹਿਬ ਦਾ ਜਨਮ ਤਲਬੰਡੀ ਵਿੱਚ ਹੋਇਆ"  # untouched
    f = segs[2]["flags"][0]
    assert f["status"] == "open" and f["candidates"][0] == "ਤਲਵੰਡੀ" and f["ledger_id"] is not None

    assert repo.list_clips(db)[0]["open_flags"] == 1
    assert repo.list_clips(db)[0]["auto_applied"] == 1
    assert repo.get_ledger(db, segs[1]["flags"][0]["ledger_id"])["times_applied"] == 1


@pytest.mark.db
def test_auto_apply_can_be_switched_off_but_suggestions_remain(db, settings, tmp_path, monkeypatch):
    repo.set_setting(db, "auto_apply_enabled", "false")
    repo.upsert_ledger(db, canonical="ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", romanized="b", surface_form="ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ",
                       entity_type="person", english_gloss=None,
                       embedding=E.embed("ਕਵੀ ਜੀ ਨੇ ਲਿਖਿਆ"), embedding_model=E.name, threshold=0.8)
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, FakeLlm(FLAGS[:1]))
    clip_id = run_to_end(db, pipeline)
    seg = repo.list_segments(db, clip_id)[1]
    assert seg["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ"
    assert seg["flags"][0]["status"] == "open" and seg["flags"][0]["ledger_id"] is not None


@pytest.mark.db
def test_llm_failure_is_a_normal_stage_failure_with_the_message_on_the_job(db, settings, tmp_path, monkeypatch):
    from katha.llm import LLMError

    class Dead(FakeLlm):
        def complete_json(self, *, system, user, schema):
            raise LLMError("claude -p reported an error: Not logged in · Please run /login")

    monkeypatch.setattr(settings, "max_attempts", 1)
    pipeline = make_pipeline(db, settings, tmp_path, monkeypatch, Dead())
    clip_id = run_to_end(db, pipeline)
    row = repo.list_clips(db)[0]
    assert (row["stage"], row["status"]) == ("correct", "failed")
    assert "Not logged in" in row["last_error"]
    assert row["segment_count"] == 3  # transcript still reviewable


@pytest.mark.db
def test_system_prompt_carries_the_gazetteer(db, settings, tmp_path, monkeypatch):
    repo.upsert_ledger(db, canonical="ਤਲਵੰਡੀ", romanized="t", surface_form=None, entity_type="place",
                       english_gloss=None, embedding=None, embedding_model=None, threshold=0.8)

    class Spy(FakeLlm):
        def complete_json(self, *, system, user, schema):
            self.system = system
            self.user = user
            return {"flags": []}

    llm = Spy()
    run_to_end(db, make_pipeline(db, settings, tmp_path, monkeypatch, llm))
    assert "ਤਲਵੰਡੀ (place)" in llm.system
    assert '"target": true' in llm.user
```

- [ ] **Step 3: Run to verify failure** — `Pipeline.__init__() got an unexpected keyword argument 'llm'`.

- [ ] **Step 4: Implement in `worker.py`**

Imports to add:

```python
from katha.embed import HashingEmbedder
from katha.fsutil import write_text_atomic
from katha.llm import make_provider
from katha.stages.correct import (
    FLAG_SCHEMA,
    SegmentText,
    build_system_prompt,
    build_user_prompt,
    flags_from_json,
    flags_to_json,
    make_windows,
    validate_flags,
)
from katha.stages.ledger import apply_span, context_text, match
```

Change `NEXT_STAGE`:

```python
NEXT_STAGE = {
    "prepare": "asr",
    "asr": "align",
    "align": "assemble",
    "assemble": "correct",
    "correct": "complete",
}
```

Change the constructor and add the stage:

```python
    def __init__(self, settings: Settings, conn: psycopg.Connection,
                 aligner=None, asr=None, llm=None, embedder=None):
        self.settings = settings
        self.conn = conn
        self._aligner = aligner
        self._asr = asr
        self._llm = llm
        self.embedder = embedder or HashingEmbedder()

    def _provider(self):
        if self._llm is not None:
            return self._llm
        model = repo.get_setting(self.conn, "llm_model", "claude-sonnet-5")
        return make_provider(self.settings, model)

    # --- correct ------------------------------------------------------------
    #
    #   correct  -> reads   segments (DB), writes work/<clip_id>/flags/<n>.json per
    #               window (validated LLM output, atomic), then flags (DB), then
    #               runs the ledger pass. A retry after a failed window reuses
    #               every window file that already exists.

    def _correct(self, job: dict) -> None:
        clip_id = job["clip_id"]
        rows = repo.list_segment_texts(self.conn, clip_id)
        if not rows:
            log.warning("clip %s: no segments to correct", clip_id)
            return
        segments = [
            SegmentText(idx=r["idx"], role=r["speaker_role"] or "unknown", text=r["text_pa"])
            for r in rows
        ]
        id_by_idx = {r["idx"]: r["id"] for r in rows}
        size = repo.get_setting_int(self.conn, "correct_window_segments", 40)
        context = repo.get_setting_int(self.conn, "correct_context_segments", 3)
        windows = make_windows(segments, size, context)
        system = build_system_prompt(repo.top_gazetteer(self.conn))
        cache_dir = self._work_dir(clip_id) / "flags"
        provider = None

        proposed = []
        for w in windows:
            path = cache_dir / f"{w.index}.json"
            if path.exists():
                flags = flags_from_json(path.read_text(encoding="utf-8"))
            else:
                if provider is None:
                    provider = self._provider()
                payload = provider.complete_json(
                    system=system, user=build_user_prompt(w), schema=FLAG_SCHEMA
                )
                flags, dropped = validate_flags(payload, w)
                log.info(
                    "clip %s window %d/%d: %d flag(s) kept, %d dropped",
                    clip_id, w.index + 1, len(windows), len(flags), dropped,
                )
                write_text_atomic(path, flags_to_json(flags))
            proposed.extend(flags)

        n = repo.save_flags(
            self.conn, clip_id,
            [(id_by_idx[f.segment_idx], f.span_text, f.kind, f.reason, list(f.candidates))
             for f in proposed],
        )
        log.info("clip %s: %d flag(s) persisted from %d window(s)", clip_id, n, len(windows))
        auto, suggested = self._apply_ledger(clip_id)
        log.info("clip %s: ledger auto-applied %d, suggested %d", clip_id, auto, suggested)

    def _apply_ledger(self, clip_id: int) -> tuple[int, int]:
        entries = repo.ledger_entries(self.conn)
        if not entries:
            return 0, 0
        surface_threshold = repo.get_setting_float(self.conn, "ledger_surface_threshold", 88.0)
        auto_enabled = repo.get_setting_bool(self.conn, "auto_apply_enabled", True)
        auto = suggested = 0
        for flag in repo.list_open_flags(self.conn, clip_id):
            ctx = context_text(flag["text_pa"], flag["span_text"])
            vec = self.embedder.embed(ctx) if ctx else None
            m = match(flag["span_text"], vec, self.embedder.name, entries, surface_threshold)
            if m is None:
                continue
            canonical = m.entry.canonical
            if m.decision == "auto" and auto_enabled:
                new_text = apply_span(flag["text_pa"], flag["span_text"], canonical)
                if new_text is not None:
                    repo.apply_auto_correction(
                        self.conn, flag_id=flag["id"], segment_id=flag["segment_id"],
                        ledger_id=m.entry.id, new_text=new_text, before=flag["span_text"],
                        after=canonical, similarity=m.context_similarity or 0.0,
                    )
                    auto += 1
                    continue
            candidates = [canonical] + [c for c in flag["candidates"] if c != canonical]
            repo.set_flag_suggestion(self.conn, flag["id"], m.entry.id, candidates[:6])
            suggested += 1
        return auto, suggested
```

Register the stage in `run_stage`'s dict: `"correct": self._correct,`.

- [ ] **Step 5: Run** — `.venv/Scripts/python.exe -m pytest tests/test_worker.py tests/test_worker_correct.py -q`. All green. If the cached-window retry test counts 6 calls, the failed window's file was written before the exception — it must not be (the write happens only after validation succeeds).

- [ ] **Step 6: Commit**

```bash
git add src/katha/worker.py tests/test_worker.py tests/test_worker_correct.py
git commit -m "feat: correct stage — windowed LLM flagging with on-disk cache and ledger auto-apply"
```

---

## Task 8: Review service and API endpoints

**Files:**
- Create: `src/katha/review.py`
- Modify: `src/katha/api/main.py`
- Test: `tests/test_review.py`, `tests/test_api_corrections.py`

**Interfaces:**
- Produces (`review.py`): `class ReviewError(Exception)` with `.status: int`; `accept_flag(conn, embedder, flag_id, text, remember: dict | None) -> dict` (returns `{"segment_id", "text_pa", "correction_id", "ledger_id"}`); `reject_flag(conn, flag_id) -> None`; `revert_correction(conn, correction_id) -> dict` (returns `{"segment_id", "text_pa"}`).
- Produces (API): `POST /api/flags/{id}/resolve`, `POST /api/corrections/{id}/revert`, `GET /api/ledger?q=&include_retired=`, `POST /api/ledger`, `PATCH /api/ledger/{id}`, `GET /api/settings`, `PUT /api/settings`.

- [ ] **Step 1: Failing service tests**

`tests/test_review.py`:

```python
from pathlib import Path

import pytest

from katha import repo, review
from katha.embed import HashingEmbedder
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

ENTRIES = [DiarizedEntry("0", "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ", 3.5, 5.5)]
WORDS = [AlignedWord(w, 3.5 + i * 0.2, 3.6 + i * 0.2, 0.9) for i, w in enumerate(ENTRIES[0].text.split())]
E = HashingEmbedder()


@pytest.fixture
def flag(db):
    cid = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    repo.save_segments(db, cid, assemble.assemble_segments(ENTRIES, WORDS), reciter_id=None)
    sid = repo.list_segment_texts(db, cid)[0]["id"]
    repo.save_flags(db, cid, [(sid, "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "proper_noun", "r", ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"])])
    return repo.list_open_flags(db, cid)[0]


@pytest.mark.db
def test_accept_rewrites_segment_logs_correction_and_teaches_ledger(db, flag):
    out = review.accept_flag(db, E, flag["id"], "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ",
                             remember={"entity_type": "person", "english_gloss": "the poet"})
    assert out["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ ਨੇ ਲਿਖਿਆ"
    seg = repo.get_segment(db, flag["segment_id"])
    assert seg["text_pa"] == out["text_pa"] and seg["edited"] is True
    f = repo.get_flag(db, flag["id"])
    assert f["status"] == "accepted" and f["resolved_text"] == "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"
    c = repo.get_correction(db, out["correction_id"])
    assert (c["source"], c["before_text"], c["after_text"], c["ledger_id"]) == \
        ("admin", "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", out["ledger_id"])
    led = repo.get_ledger(db, out["ledger_id"])
    assert led["surface_forms"] == ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ"] and led["entity_type"] == "person"
    assert led["romanized"].startswith("bh") and led["english_gloss"] == "the poet"
    assert led["embedding_model"] == E.name and led["confidence_threshold"] == pytest.approx(0.8)
    assert f["ledger_id"] == out["ledger_id"]


@pytest.mark.db
def test_accept_without_remember_leaves_ledger_alone(db, flag):
    out = review.accept_flag(db, E, flag["id"], "ਕੁਝ ਹੋਰ", remember=None)
    assert out["ledger_id"] is None and repo.list_ledger(db, None, True) == []
    assert repo.get_segment(db, flag["segment_id"])["text_pa"] == "ਕਵੀ ਕੁਝ ਹੋਰ ਜੀ ਨੇ ਲਿਖਿਆ"


@pytest.mark.db
def test_accept_rejects_blank_text_and_non_open_flags(db, flag):
    with pytest.raises(review.ReviewError) as exc:
        review.accept_flag(db, E, flag["id"], "   ", None)
    assert exc.value.status == 400
    review.reject_flag(db, flag["id"])
    with pytest.raises(review.ReviewError) as exc:
        review.accept_flag(db, E, flag["id"], "x", None)
    assert exc.value.status == 409
    with pytest.raises(review.ReviewError) as exc:
        review.accept_flag(db, E, 999999, "x", None)
    assert exc.value.status == 404


@pytest.mark.db
def test_accept_marks_flag_stale_when_span_was_edited_away(db, flag):
    repo.update_segment_text(db, flag["segment_id"], "ਬਿਲਕੁਲ ਵੱਖਰਾ ਪਾਠ")
    with pytest.raises(review.ReviewError) as exc:
        review.accept_flag(db, E, flag["id"], "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", None)
    assert exc.value.status == 409
    assert repo.get_flag(db, flag["id"])["status"] == "stale"


@pytest.mark.db
def test_revert_restores_text_reopens_flag_and_is_idempotent_safe(db, flag):
    out = review.accept_flag(db, E, flag["id"], "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", None)
    back = review.revert_correction(db, out["correction_id"])
    assert back["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ"
    assert repo.get_flag(db, flag["id"])["status"] == "open"
    assert repo.get_correction(db, out["correction_id"])["reverted_at"] is not None
    with pytest.raises(review.ReviewError) as exc:
        review.revert_correction(db, out["correction_id"])
    assert exc.value.status == 409


@pytest.mark.db
def test_revert_refuses_when_the_corrected_text_is_gone(db, flag):
    out = review.accept_flag(db, E, flag["id"], "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ", None)
    repo.update_segment_text(db, flag["segment_id"], "ਹੁਣ ਕੁਝ ਹੋਰ")
    with pytest.raises(review.ReviewError) as exc:
        review.revert_correction(db, out["correction_id"])
    assert exc.value.status == 409
    with pytest.raises(review.ReviewError) as exc:
        review.revert_correction(db, 999999)
    assert exc.value.status == 404
```

- [ ] **Step 2: Implement `review.py`**

```python
"""Admin actions on flags and corrections. Thin, transactional-per-call, and
HTTP-agnostic so the API layer only maps ReviewError.status to a response."""
import psycopg

from katha import repo
from katha.embed import Embedder
from katha.stages.ledger import apply_span, context_text
from katha.text import romanize


class ReviewError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def accept_flag(
    conn: psycopg.Connection, embedder: Embedder, flag_id: int, text: str, remember: dict | None
) -> dict:
    text = (text or "").strip()
    if not text:
        raise ReviewError(400, "replacement text must not be empty")
    flag = repo.get_flag(conn, flag_id)
    if flag is None:
        raise ReviewError(404, "no such flag")
    if flag["status"] != "open":
        raise ReviewError(409, f"flag is {flag['status']}, not open")
    original = flag["text_pa"]
    new_text = apply_span(original, flag["span_text"], text)
    if new_text is None:
        repo.resolve_flag(conn, flag_id, "stale")
        raise ReviewError(409, "the flagged span is no longer present in the segment (marked stale)")

    ledger_id = None
    if remember:
        ctx = context_text(original, flag["span_text"])
        vec = embedder.embed(ctx) if ctx else None
        ledger_id = repo.upsert_ledger(
            conn,
            canonical=text,
            romanized=romanize(text),
            surface_form=flag["span_text"],
            entity_type=remember.get("entity_type") or "other",
            english_gloss=remember.get("english_gloss") or None,
            embedding=vec,
            embedding_model=embedder.name if vec else None,
            threshold=repo.get_setting_float(conn, "ledger_context_threshold", 0.8),
        )

    repo.update_segment_text(conn, flag["segment_id"], new_text)
    correction_id = repo.add_correction(
        conn, segment_id=flag["segment_id"], flag_id=flag_id, ledger_id=ledger_id,
        source="admin", before=flag["span_text"], after=text, similarity=None,
    )
    repo.resolve_flag(conn, flag_id, "accepted", resolved_text=text, ledger_id=ledger_id)
    return {"segment_id": flag["segment_id"], "text_pa": new_text,
            "correction_id": correction_id, "ledger_id": ledger_id}


def reject_flag(conn: psycopg.Connection, flag_id: int) -> None:
    flag = repo.get_flag(conn, flag_id)
    if flag is None:
        raise ReviewError(404, "no such flag")
    if flag["status"] != "open":
        raise ReviewError(409, f"flag is {flag['status']}, not open")
    repo.resolve_flag(conn, flag_id, "rejected")


def revert_correction(conn: psycopg.Connection, correction_id: int) -> dict:
    c = repo.get_correction(conn, correction_id)
    if c is None:
        raise ReviewError(404, "no such correction")
    if c["reverted_at"] is not None:
        raise ReviewError(409, "already reverted")
    seg = repo.get_segment(conn, c["segment_id"])
    restored = apply_span(seg["text_pa"], c["after_text"], c["before_text"])
    if restored is None:
        raise ReviewError(409, "the corrected text is no longer present in the segment")
    repo.update_segment_text(conn, seg["id"], restored)
    repo.mark_reverted(conn, correction_id)
    if c["flag_id"] is not None:
        repo.reopen_flag(conn, c["flag_id"])
    return {"segment_id": seg["id"], "text_pa": restored}
```

- [ ] **Step 3: Run** — `.venv/Scripts/python.exe -m pytest tests/test_review.py -q` → green.

- [ ] **Step 4: Failing API tests**

`tests/test_api_corrections.py`:

```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from katha import repo
from katha.api.main import app, get_conn, get_app_settings
from katha.stages import assemble
from katha.stages.types import AlignedWord, DiarizedEntry

ENTRIES = [DiarizedEntry("0", "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ", 3.5, 5.5)]
WORDS = [AlignedWord(w, 3.5 + i * 0.2, 3.6 + i * 0.2, 0.9) for i, w in enumerate(ENTRIES[0].text.split())]


@pytest.fixture
def api(db, settings):
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_app_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def flagged(db):
    cid = repo.register_clip(db, Path("/a/k.mp3"), "k", "abc")
    repo.save_segments(db, cid, assemble.assemble_segments(ENTRIES, WORDS), reciter_id=None)
    sid = repo.list_segment_texts(db, cid)[0]["id"]
    repo.save_flags(db, cid, [(sid, "ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ", "proper_noun", "r", ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"])])
    return cid, repo.list_open_flags(db, cid)[0]["id"]


@pytest.mark.db
def test_clip_payload_includes_flags_and_list_counts(api, flagged):
    cid, fid = flagged
    body = api.get(f"/api/clips/{cid}").json()
    f = body["segments"][0]["flags"][0]
    assert f["id"] == fid and f["status"] == "open" and f["candidates"] == ["ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ"]
    assert body["segments"][0]["corrections"] == []
    row = api.get("/api/clips").json()["clips"][0]
    assert row["open_flags"] == 1 and row["auto_applied"] == 0


@pytest.mark.db
def test_resolve_accept_then_revert_round_trip(api, flagged):
    cid, fid = flagged
    r = api.post(f"/api/flags/{fid}/resolve", json={
        "action": "accept", "text": "ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ",
        "remember": {"entity_type": "person", "english_gloss": "poet"}})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਘ ਜੀ ਨੇ ਲਿਖਿਆ" and out["ledger_id"]
    seg = api.get(f"/api/clips/{cid}").json()["segments"][0]
    assert seg["flags"][0]["status"] == "accepted"
    assert seg["corrections"][0]["source"] == "admin" and seg["corrections"][0]["reverted"] is False
    r = api.post(f"/api/corrections/{out['correction_id']}/revert")
    assert r.status_code == 200 and r.json()["text_pa"] == "ਕਵੀ ਭਾਈ ਸੰਤੋਖ ਸਿੰਗ ਜੀ ਨੇ ਲਿਖਿਆ"
    seg = api.get(f"/api/clips/{cid}").json()["segments"][0]
    assert seg["flags"][0]["status"] == "open" and seg["corrections"][0]["reverted"] is True
    assert api.post(f"/api/corrections/{out['correction_id']}/revert").status_code == 409


@pytest.mark.db
def test_resolve_reject_and_error_statuses(api, flagged):
    cid, fid = flagged
    assert api.post(f"/api/flags/{fid}/resolve", json={"action": "accept"}).status_code == 400
    assert api.post(f"/api/flags/{fid}/resolve", json={"action": "nope"}).status_code == 422
    assert api.post(f"/api/flags/{fid}/resolve", json={"action": "reject"}).status_code == 200
    assert api.post(f"/api/flags/{fid}/resolve", json={"action": "reject"}).status_code == 409
    assert api.post("/api/flags/999999/resolve", json={"action": "reject"}).status_code == 404


@pytest.mark.db
def test_ledger_endpoints(api):
    r = api.post("/api/ledger", json={"canonical_gurmukhi": "ਤਲਵੰਡੀ", "surface_forms": ["ਤਲਬੰਡੀ"],
                                      "entity_type": "place"})
    assert r.status_code == 200, r.text
    lid = r.json()["id"]
    rows = api.get("/api/ledger", params={"q": "talv"}).json()["entries"]
    assert [e["id"] for e in rows] == [lid]
    assert rows[0]["romanized"].startswith("tal") and rows[0]["confidence_threshold"] == pytest.approx(0.8)
    r = api.patch(f"/api/ledger/{lid}", json={"surface_forms": ["ਤਲਬੰਡੀ", "ਤਲਵੰਢੀ"],
                                              "confidence_threshold": 0.9, "retired": True})
    assert r.status_code == 200
    assert api.get("/api/ledger").json()["entries"] == []
    row = api.get("/api/ledger", params={"include_retired": "true"}).json()["entries"][0]
    assert row["retired"] is True and row["surface_forms"] == ["ਤਲਬੰਡੀ", "ਤਲਵੰਢੀ"]
    assert api.patch("/api/ledger/999999", json={"retired": False}).status_code == 404
    assert api.post("/api/ledger", json={"canonical_gurmukhi": " ", "entity_type": "place"}).status_code == 400
    assert api.post("/api/ledger", json={"canonical_gurmukhi": "x", "entity_type": "alien"}).status_code == 422


@pytest.mark.db
def test_settings_endpoints_validate(api):
    body = api.get("/api/settings").json()
    assert body["values"]["files_per_day"] == "3"
    assert any(s["key"] == "llm_model" for s in body["specs"])
    r = api.put("/api/settings", json={"values": {"files_per_day": "5", "auto_apply_enabled": "no"}})
    assert r.status_code == 200
    assert api.get("/api/settings").json()["values"]["auto_apply_enabled"] == "false"
    assert api.get("/api/settings").json()["values"]["files_per_day"] == "5"
    assert api.put("/api/settings", json={"values": {"files_per_day": "-2"}}).status_code == 400
    assert api.put("/api/settings", json={"values": {"bogus": "1"}}).status_code == 400
    assert api.get("/api/settings").json()["values"]["files_per_day"] == "5"  # nothing partial
```

- [ ] **Step 5: Add the endpoints to `api/main.py`**

Imports to add: `from typing import Literal`, `from katha import appsettings, review`, `from katha.embed import HashingEmbedder`, `from katha.text import romanize`.

```python
ENTITY_TYPES = Literal["person", "place", "date", "book", "title", "term", "other"]
_embedder = HashingEmbedder()


class RememberSpec(BaseModel):
    entity_type: ENTITY_TYPES
    english_gloss: str | None = None


class ResolveBody(BaseModel):
    action: Literal["accept", "reject"]
    text: str | None = None
    remember: RememberSpec | None = None


class LedgerCreate(BaseModel):
    canonical_gurmukhi: str
    surface_forms: list[str] = []
    entity_type: ENTITY_TYPES
    english_gloss: str | None = None
    confidence_threshold: float | None = None


class LedgerPatch(BaseModel):
    canonical_gurmukhi: str | None = None
    surface_forms: list[str] | None = None
    entity_type: ENTITY_TYPES | None = None
    english_gloss: str | None = None
    confidence_threshold: float | None = None
    retired: bool | None = None


class SettingsBody(BaseModel):
    values: dict[str, str]


def _review(fn, *args):
    try:
        return fn(*args)
    except review.ReviewError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@app.post("/api/flags/{flag_id}/resolve")
def resolve_flag(flag_id: int, body: ResolveBody, conn=Depends(get_conn)):
    if body.action == "reject":
        _review(review.reject_flag, conn, flag_id)
        return {"ok": True, "status": "rejected"}
    if body.text is None or not body.text.strip():
        raise HTTPException(400, "accept requires non-empty text")
    remember = body.remember.model_dump() if body.remember else None
    out = _review(review.accept_flag, conn, _embedder, flag_id, body.text, remember)
    return {"ok": True, "status": "accepted", **out}


@app.post("/api/corrections/{correction_id}/revert")
def revert_correction(correction_id: int, conn=Depends(get_conn)):
    return {"ok": True, **_review(review.revert_correction, conn, correction_id)}


@app.get("/api/ledger")
def list_ledger(q: str | None = None, include_retired: bool = False, conn=Depends(get_conn)):
    return {"entries": repo.list_ledger(conn, q, include_retired)}


@app.post("/api/ledger")
def create_ledger(body: LedgerCreate, conn=Depends(get_conn)):
    canonical = body.canonical_gurmukhi.strip()
    if not canonical:
        raise HTTPException(400, "canonical_gurmukhi must not be empty")
    threshold = body.confidence_threshold
    if threshold is None:
        threshold = repo.get_setting_float(conn, "ledger_context_threshold", 0.8)
    ledger_id = repo.upsert_ledger(
        conn, canonical=canonical, romanized=romanize(canonical), surface_form=None,
        entity_type=body.entity_type, english_gloss=body.english_gloss, embedding=None,
        embedding_model=None, threshold=threshold,
    )
    forms = [f.strip() for f in body.surface_forms if f.strip() and f.strip() != canonical]
    if forms:
        existing = repo.get_ledger(conn, ledger_id)["surface_forms"]
        merged = list(dict.fromkeys([*existing, *forms]))
        repo.update_ledger(conn, ledger_id, surface_forms=merged)
    return {"ok": True, "id": ledger_id}


@app.patch("/api/ledger/{ledger_id}")
def patch_ledger(ledger_id: int, body: LedgerPatch, conn=Depends(get_conn)):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if "canonical_gurmukhi" in fields:
        fields["canonical_gurmukhi"] = fields["canonical_gurmukhi"].strip()
        if not fields["canonical_gurmukhi"]:
            raise HTTPException(400, "canonical_gurmukhi must not be empty")
        fields["romanized"] = romanize(fields["canonical_gurmukhi"])
    if "surface_forms" in fields:
        fields["surface_forms"] = [f.strip() for f in fields["surface_forms"] if f.strip()]
    if "confidence_threshold" in fields and not 0.0 <= fields["confidence_threshold"] <= 1.0:
        raise HTTPException(400, "confidence_threshold must be within 0..1")
    if not fields:
        raise HTTPException(400, "nothing to update")
    if repo.get_ledger(conn, ledger_id) is None:
        raise HTTPException(404, "no such ledger entry")
    repo.update_ledger(conn, ledger_id, **fields)
    return {"ok": True}


@app.get("/api/settings")
def get_settings_endpoint(conn=Depends(get_conn)):
    specs = [
        {"key": s.key, "kind": s.kind, "default": s.default, "minimum": s.minimum,
         "maximum": s.maximum, "help": s.help}
        for s in appsettings.SPECS.values()
    ]
    return {"values": repo.get_all_settings(conn), "specs": specs}


@app.put("/api/settings")
def put_settings(body: SettingsBody, conn=Depends(get_conn)):
    canonical: dict[str, str] = {}
    for key, value in body.values.items():          # validate everything first
        try:
            canonical[key] = appsettings.validate(key, value)
        except KeyError:
            raise HTTPException(400, f"unknown setting: {key}") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
    for key, value in canonical.items():
        repo.set_setting(conn, key, value)
    return {"ok": True, "values": repo.get_all_settings(conn)}
```

- [ ] **Step 6: Run** — `.venv/Scripts/python.exe -m pytest tests/test_api_corrections.py tests/test_api.py -q` → green. Then the whole fast suite.

- [ ] **Step 7: Commit**

```bash
git add src/katha/review.py src/katha/api/main.py tests/test_review.py tests/test_api_corrections.py
git commit -m "feat: flag resolution, correction revert, ledger and settings endpoints"
```

---

## Task 9: Admin UI — flags in the review view, ledger and settings pages

**Files:**
- Modify: `web/lib/types.ts`, `web/lib/api.ts`, `web/components/SegmentList.tsx`, `web/app/page.tsx`, `web/app/layout.tsx`, `web/app/globals.css`
- Create: `web/lib/highlight.tsx`, `web/components/FlagPanel.tsx`, `web/app/ledger/page.tsx`, `web/app/settings/page.tsx`

**Interfaces:**
- Consumes: the Task 8 endpoints and the `flags` / `corrections` arrays on each segment.

- [ ] **Step 1: Types and API client**

Append to `web/lib/types.ts`:

```ts
export type FlagStatus = 'open' | 'accepted' | 'rejected' | 'auto_applied' | 'stale'
export type EntityType = 'person' | 'place' | 'date' | 'book' | 'title' | 'term' | 'other'
export const ENTITY_TYPES: EntityType[] = ['person', 'place', 'date', 'book', 'title', 'term', 'other']

export type Flag = {
  id: number
  span_text: string
  kind: string
  reason: string
  candidates: string[]
  ledger_id: number | null
  status: FlagStatus
  resolved_text: string | null
}

export type Correction = {
  id: number
  flag_id: number | null
  ledger_id: number | null
  source: 'auto' | 'admin'
  before_text: string
  after_text: string
  context_similarity: number | null
  reverted: boolean
}

export type LedgerEntry = {
  id: number
  canonical_gurmukhi: string
  romanized: string
  surface_forms: string[]
  entity_type: EntityType
  english_gloss: string | null
  embedding_model: string | null
  confidence_threshold: number
  times_applied: number
  retired: boolean
  last_used_at: string | null
}

export type SettingSpec = {
  key: string
  kind: 'int' | 'float' | 'bool' | 'str'
  default: string
  minimum: number | null
  maximum: number | null
  help: string
}
```

Add `flags: Flag[]` and `corrections: Correction[]` to `Segment`, and `open_flags: number` and `auto_applied: number` to `ClipListItem`.

Append to `web/lib/api.ts`:

```ts
import type { EntityType, LedgerEntry, SettingSpec } from './types'

export type ResolveBody =
  | { action: 'reject' }
  | { action: 'accept'; text: string; remember?: { entity_type: EntityType; english_gloss?: string } }

export const resolveFlag = (id: number, body: ResolveBody) =>
  json<{ ok: boolean; status: string; text_pa?: string }>(`/api/flags/${id}/resolve`, {
    method: 'POST',
    body: JSON.stringify(body),
  })

export const revertCorrection = (id: number) =>
  json<{ ok: boolean; text_pa: string }>(`/api/corrections/${id}/revert`, { method: 'POST' })

export const fetchLedger = (q: string, includeRetired: boolean) =>
  json<{ entries: LedgerEntry[] }>(
    `/api/ledger?q=${encodeURIComponent(q)}&include_retired=${includeRetired}`,
  ).then((r) => r.entries)

export const createLedger = (body: {
  canonical_gurmukhi: string
  surface_forms: string[]
  entity_type: EntityType
  english_gloss?: string
}) => json<{ ok: boolean; id: number }>('/api/ledger', { method: 'POST', body: JSON.stringify(body) })

export const patchLedger = (id: number, body: Partial<Omit<LedgerEntry, 'id' | 'romanized' | 'times_applied' | 'embedding_model' | 'last_used_at'>>) =>
  json(`/api/ledger/${id}`, { method: 'PATCH', body: JSON.stringify(body) })

export const fetchSettings = () =>
  json<{ values: Record<string, string>; specs: SettingSpec[] }>('/api/settings')

export const putSettings = (values: Record<string, string>) =>
  json<{ ok: boolean; values: Record<string, string> }>('/api/settings', {
    method: 'PUT',
    body: JSON.stringify({ values }),
  })
```

Also make `json()` surface the API's `detail` message: after `if (!res.ok)`, read `const detail = await res.text().catch(() => '')` and throw `new Error(\`${init?.method ?? 'GET'} ${path} → ${res.status} ${detail}\`)`.

- [ ] **Step 2: Highlight helper** — `web/lib/highlight.tsx`:

```tsx
import type { ReactNode } from 'react'

/** Wrap the first occurrence of each span in <mark>. Overlaps are dropped. */
export function highlight(text: string, spans: string[]): ReactNode[] {
  const ranges: [number, number][] = []
  for (const s of spans) {
    if (!s) continue
    const at = text.indexOf(s)
    if (at < 0) continue
    const end = at + s.length
    if (ranges.some(([a, b]) => at < b && end > a)) continue
    ranges.push([at, end])
  }
  ranges.sort((x, y) => x[0] - y[0])
  const out: ReactNode[] = []
  let cursor = 0
  ranges.forEach(([a, b], i) => {
    if (a > cursor) out.push(text.slice(cursor, a))
    out.push(<mark key={i}>{text.slice(a, b)}</mark>)
    cursor = b
  })
  if (cursor < text.length) out.push(text.slice(cursor))
  return out
}
```

- [ ] **Step 3: `FlagPanel.tsx`**

```tsx
'use client'

import { useState } from 'react'
import { romanize } from '@/lib/roman'
import { ENTITY_TYPES, type EntityType, type Flag } from '@/lib/types'
import type { ResolveBody } from '@/lib/api'

export function FlagPanel({ flag, onResolve }: { flag: Flag; onResolve: (body: ResolveBody) => Promise<void> }) {
  const [custom, setCustom] = useState('')
  const [remember, setRemember] = useState(false)
  const [entityType, setEntityType] = useState<EntityType>(flag.kind === 'proper_noun' ? 'person' : 'other')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async (body: ResolveBody) => {
    setBusy(true)
    setError(null)
    try {
      await onResolve(body)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const accept = (text: string) =>
    run({ action: 'accept', text, remember: remember ? { entity_type: entityType } : undefined })

  return (
    <div className="flag">
      <div className="flag-head">
        <span className="pill">{flag.kind}</span>
        <span className="gurmukhi" style={{ fontSize: 16 }}>{flag.span_text}</span>
        <span style={{ color: 'var(--muted)' }}>{flag.reason}</span>
        {flag.ledger_id && <span className="pill done" title="Top candidate comes from the ledger">ledger</span>}
      </div>
      <div className="flag-candidates">
        {flag.candidates.map((c) => (
          <button key={c} disabled={busy} onClick={() => accept(c)} className="candidate">
            <span className="gurmukhi" style={{ fontSize: 17, lineHeight: 1.4 }}>{c}</span>
            <span style={{ color: 'var(--muted)', fontSize: 12 }}>{romanize(c)}</span>
          </button>
        ))}
      </div>
      <div className="flag-actions">
        <input
          className="gurmukhi"
          placeholder="custom Gurmukhi…"
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          style={{ fontSize: 16, flex: 1 }}
        />
        <button disabled={busy || !custom.trim()} onClick={() => accept(custom)}>Apply</button>
        <button disabled={busy} onClick={() => run({ action: 'reject' })}>Reject</button>
        <label style={{ fontSize: 13, display: 'flex', gap: 6, alignItems: 'center' }}>
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          Remember as
          <select value={entityType} onChange={(e) => setEntityType(e.target.value as EntityType)} disabled={!remember}>
            {ENTITY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
      </div>
      {error && <div style={{ color: 'var(--warn)', fontSize: 13 }}>{error}</div>}
    </div>
  )
}
```

- [ ] **Step 4: Wire flags into `SegmentList.tsx`**

Add props `onResolve: (flagId: number, body: ResolveBody) => Promise<void>` and `onRevert: (correctionId: number) => Promise<void>`. Import `highlight`, `FlagPanel`, `ResolveBody`. Replace the read-only Gurmukhi `<div>`'s child `{s.text_pa}` with `{highlight(s.text_pa, s.flags.filter((f) => f.status === 'open').map((f) => f.span_text))}`. After the romanized line, add:

```tsx
            {s.corrections.filter((c) => !c.reverted).map((c) => (
              <div key={c.id} className="correction">
                <span className={`pill ${c.source === 'auto' ? 'done' : ''}`}>
                  {c.source === 'auto' ? 'auto-fixed' : 'corrected'}
                </span>
                <span className="gurmukhi" style={{ fontSize: 15 }}>{c.before_text} → {c.after_text}</span>
                {c.context_similarity !== null && (
                  <span style={{ color: 'var(--muted)', fontSize: 12 }}>sim {c.context_similarity.toFixed(2)}</span>
                )}
                <button onClick={() => onRevert(c.id).catch((e) => setSaveError(String(e)))}>Revert</button>
              </div>
            ))}
            {s.flags.filter((f) => f.status === 'open').map((f) => (
              <FlagPanel key={f.id} flag={f} onResolve={(body) => onResolve(f.id, body)} />
            ))}
            {s.flags.some((f) => f.status !== 'open' && f.status !== 'auto_applied') && (
              <div style={{ color: 'var(--muted)', fontSize: 12 }}>
                {s.flags.filter((f) => f.status !== 'open' && f.status !== 'auto_applied').length} flag(s) resolved
              </div>
            )}
```

In `web/app/clips/[id]/page.tsx`, pass `onResolve={(id, body) => resolveFlag(id, body).then(() => load())}` and `onRevert={(id) => revertCorrection(id).then(() => load())}` (import both from `@/lib/api`). Note `load()` returns a promise that resolves after state updates, so a failed reload surfaces as the page error, not a panel error.

- [ ] **Step 5: Dashboard and nav**

`web/app/page.tsx`: link the title when `c.segment_count > 0` (not only when `done`); add columns `Flags` (`c.open_flags > 0 ? <span className="pill">{c.open_flags}</span> : '—'`) and `Auto-fixed` (`c.auto_applied || '—'`).

`web/app/layout.tsx` header:

```tsx
        <header>
          <a href="/">Katha Pipeline</a>
          <nav><a href="/ledger">Ledger</a><a href="/settings">Settings</a></nav>
        </header>
```

`globals.css` additions:

```css
header { display: flex; justify-content: space-between; align-items: center; }
nav a { color: var(--muted); margin-left: 18px; text-decoration: none; font-weight: 500; }
nav a:hover { color: var(--ink); }
mark { background: rgba(212,162,76,.25); color: inherit; border-bottom: 2px solid var(--reciter); padding: 0 2px; }
.flag { margin: 8px 0 4px 0; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
.flag-head { display: flex; gap: 10px; align-items: center; font-size: 13px; flex-wrap: wrap; }
.flag-candidates { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.candidate { display: flex; flex-direction: column; align-items: flex-start; padding: 6px 10px; }
.flag-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.flag-actions input[type=text], .flag-actions input:not([type]) { background: var(--bg); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
.correction { display: flex; gap: 10px; align-items: center; font-size: 13px; margin-top: 6px; }
input, select { font: inherit; }
```

- [ ] **Step 6: `/ledger` page** — `web/app/ledger/page.tsx`:

```tsx
'use client'

import { useCallback, useEffect, useState } from 'react'
import { createLedger, fetchLedger, patchLedger } from '@/lib/api'
import { romanize } from '@/lib/roman'
import { ENTITY_TYPES, type EntityType, type LedgerEntry } from '@/lib/types'

const field = { background: 'var(--bg)', color: 'var(--ink)', border: '1px solid var(--line)', borderRadius: 6, padding: '6px 8px' } as const

export default function Ledger() {
  const [q, setQ] = useState('')
  const [includeRetired, setIncludeRetired] = useState(false)
  const [rows, setRows] = useState<LedgerEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState({ canonical: '', forms: '', type: 'person' as EntityType, gloss: '' })
  const [editing, setEditing] = useState<LedgerEntry | null>(null)

  const load = useCallback(
    () => fetchLedger(q, includeRetired).then(setRows).catch((e) => setError(String(e))),
    [q, includeRetired],
  )
  useEffect(() => { load() }, [load])

  const create = () =>
    createLedger({
      canonical_gurmukhi: draft.canonical,
      surface_forms: draft.forms.split(',').map((s) => s.trim()).filter(Boolean),
      entity_type: draft.type,
      english_gloss: draft.gloss || undefined,
    })
      .then(() => { setDraft({ canonical: '', forms: '', type: 'person', gloss: '' }); return load() })
      .catch((e) => setError(String(e)))

  const save = () => {
    if (!editing) return
    patchLedger(editing.id, {
      canonical_gurmukhi: editing.canonical_gurmukhi,
      surface_forms: editing.surface_forms,
      entity_type: editing.entity_type,
      english_gloss: editing.english_gloss,
      confidence_threshold: editing.confidence_threshold,
    })
      .then(() => { setEditing(null); return load() })
      .catch((e) => setError(String(e)))
  }

  const toggleRetired = (r: LedgerEntry) =>
    patchLedger(r.id, { retired: !r.retired }).then(load).catch((e) => setError(String(e)))

  return (
    <>
      <h2>Correction ledger</h2>
      {error && <p className="pill failed">{error}</p>}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 12 }}>
        <input placeholder="search Gurmukhi, romanized, or a misrecognition" value={q} onChange={(e) => setQ(e.target.value)} style={{ ...field, flex: 1 }} />
        <label style={{ fontSize: 13 }}>
          <input type="checkbox" checked={includeRetired} onChange={(e) => setIncludeRetired(e.target.checked)} /> show retired
        </label>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <input className="gurmukhi" placeholder="canonical Gurmukhi" value={draft.canonical} onChange={(e) => setDraft({ ...draft, canonical: e.target.value })} style={{ ...field, fontSize: 16 }} />
        <input className="gurmukhi" placeholder="misrecognitions, comma-separated" value={draft.forms} onChange={(e) => setDraft({ ...draft, forms: e.target.value })} style={{ ...field, fontSize: 16, flex: 1 }} />
        <select value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value as EntityType })} style={field}>
          {ENTITY_TYPES.map((t) => <option key={t}>{t}</option>)}
        </select>
        <input placeholder="English gloss" value={draft.gloss} onChange={(e) => setDraft({ ...draft, gloss: e.target.value })} style={field} />
        <button disabled={!draft.canonical.trim()} onClick={create}>Add</button>
      </div>

      <table>
        <thead>
          <tr><th>Canonical</th><th>Misrecognitions</th><th>Type</th><th>Gloss</th><th>Threshold</th><th>Used</th><th></th></tr>
        </thead>
        <tbody>
          {rows.map((r) => editing?.id === r.id ? (
            <tr key={r.id}>
              <td><input className="gurmukhi" value={editing.canonical_gurmukhi} onChange={(e) => setEditing({ ...editing, canonical_gurmukhi: e.target.value })} style={{ ...field, fontSize: 16 }} /></td>
              <td><input className="gurmukhi" value={editing.surface_forms.join(', ')} onChange={(e) => setEditing({ ...editing, surface_forms: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })} style={{ ...field, fontSize: 16, width: '100%' }} /></td>
              <td><select value={editing.entity_type} onChange={(e) => setEditing({ ...editing, entity_type: e.target.value as EntityType })} style={field}>{ENTITY_TYPES.map((t) => <option key={t}>{t}</option>)}</select></td>
              <td><input value={editing.english_gloss ?? ''} onChange={(e) => setEditing({ ...editing, english_gloss: e.target.value || null })} style={field} /></td>
              <td><input type="number" step="0.05" min="0" max="1" value={editing.confidence_threshold} onChange={(e) => setEditing({ ...editing, confidence_threshold: Number(e.target.value) })} style={{ ...field, width: 80 }} /></td>
              <td>{r.times_applied}</td>
              <td><button onClick={save}>Save</button> <button onClick={() => setEditing(null)}>Cancel</button></td>
            </tr>
          ) : (
            <tr key={r.id} style={{ opacity: r.retired ? 0.5 : 1 }}>
              <td><div className="gurmukhi" style={{ fontSize: 17 }}>{r.canonical_gurmukhi}</div><div style={{ color: 'var(--muted)', fontSize: 12 }}>{r.romanized}</div></td>
              <td className="gurmukhi" style={{ fontSize: 15 }}>{r.surface_forms.map((f) => <div key={f}>{f} <span style={{ color: 'var(--muted)', fontSize: 12 }}>{romanize(f)}</span></div>)}</td>
              <td>{r.entity_type}</td>
              <td>{r.english_gloss ?? '—'}</td>
              <td>{r.confidence_threshold.toFixed(2)}{!r.embedding_model && <div style={{ color: 'var(--muted)', fontSize: 11 }}>no context yet</div>}</td>
              <td>{r.times_applied}</td>
              <td><button onClick={() => setEditing(r)}>Edit</button> <button onClick={() => toggleRetired(r)}>{r.retired ? 'Restore' : 'Retire'}</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}
```

- [ ] **Step 7: `/settings` page** — `web/app/settings/page.tsx`:

```tsx
'use client'

import { useEffect, useState } from 'react'
import { fetchSettings, putSettings } from '@/lib/api'
import type { SettingSpec } from '@/lib/types'

const field = { background: 'var(--bg)', color: 'var(--ink)', border: '1px solid var(--line)', borderRadius: 6, padding: '6px 8px' } as const

export default function SettingsPage() {
  const [specs, setSpecs] = useState<SettingSpec[]>([])
  const [values, setValues] = useState<Record<string, string>>({})
  const [saved, setSaved] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    fetchSettings().then((r) => { setSpecs(r.specs); setValues(r.values); setSaved(r.values) }).catch((e) => setError(String(e)))
  }, [])

  const dirty = Object.keys(values).filter((k) => values[k] !== saved[k])

  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await putSettings(Object.fromEntries(dirty.map((k) => [k, values[k]])))
      setValues(r.values)
      setSaved(r.values)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <h2>Settings</h2>
      <p style={{ color: 'var(--muted)', fontSize: 13 }}>
        These live in the database and take effect on the worker&apos;s next run. Paths, keys and the LLM backend are environment variables (see .env).
      </p>
      {error && <p className="pill failed">{error}</p>}
      <table>
        <tbody>
          {specs.map((s) => (
            <tr key={s.key}>
              <td style={{ width: 260 }}><code>{s.key}</code><div style={{ color: 'var(--muted)', fontSize: 12 }}>{s.help}</div></td>
              <td>
                {s.kind === 'bool' ? (
                  <select value={values[s.key] ?? s.default} onChange={(e) => setValues({ ...values, [s.key]: e.target.value })} style={field}>
                    <option value="true">true</option><option value="false">false</option>
                  </select>
                ) : (
                  <input
                    type={s.kind === 'str' ? 'text' : 'number'}
                    step={s.kind === 'float' ? 'any' : 1}
                    min={s.minimum ?? undefined}
                    max={s.maximum ?? undefined}
                    value={values[s.key] ?? s.default}
                    onChange={(e) => setValues({ ...values, [s.key]: e.target.value })}
                    style={{ ...field, width: s.kind === 'str' ? 320 : 120 }}
                  />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 12 }}>
        <button disabled={busy || dirty.length === 0} onClick={save}>{busy ? 'Saving…' : `Save ${dirty.length ? `(${dirty.length})` : ''}`}</button>
      </div>
    </>
  )
}
```

- [ ] **Step 8: Build**

Run: `cd web && npm run build`. Expected: 0 TypeScript errors, the three routes listed (`/`, `/clips/[id]`, `/ledger`, `/settings`). Fix type errors in place (do not loosen `tsconfig`).

- [ ] **Step 9: Commit**

```bash
git add web/
git commit -m "feat: flag review panel, auto-fix revert, ledger browser, and settings page"
```

---

## Task 10: Documentation and deployment notes

**Files:**
- Modify: `SETUP.md`, `.env.example`

- [ ] **Step 1: `.env.example`** — add before the API section:

```bash
# ─── LLM (correction pass) ───────────────────────────────────────────────────
# claude_cli  runs `claude -p` on your Max subscription. The user the worker
#             runs as MUST have run `claude login` once on this host (OAuth
#             expires periodically — re-run it when the dashboard shows
#             "Not logged in"). Uses --safe-mode, never --bare.
# anthropic_api  pay-as-you-go via the SDK; needs KATHA_ANTHROPIC_API_KEY.
KATHA_LLM_BACKEND=claude_cli
KATHA_CLAUDE_BIN=claude
KATHA_LLM_TIMEOUT_SECONDS=900
KATHA_LLM_MAX_TOKENS=16000
KATHA_ANTHROPIC_API_KEY=
# The model itself (claude-sonnet-5 by default) is a DATABASE setting: change it
# on the Settings page, not here.
```

- [ ] **Step 2: `SETUP.md`** — update the stage diagram to include `correct` (Claude, local CLI); in Part 1 add "6. Claude Code CLI, logged in" with `claude login`; in Part 3 add a section **The correction pass** describing: what a flag is, the review panel, Remember-as-ledger, auto-fix + Revert, the `/ledger` and `/settings` pages, and that the per-window cache lives in `work/<clip>/flags/`; in Part 4 add `sudo -u katha claude login` and the API fallback; in Part 6 add troubleshooting entries: "correct: claude -p reported an error: Not logged in" → re-login as the service user; "claude CLI not found" → set `KATHA_CLAUDE_BIN`; "output truncated at max_tokens" → raise `KATHA_LLM_MAX_TOKENS` or lower `correct_window_segments`. Update the expected test count in Part 2.

- [ ] **Step 3: Full verification**

```bash
.venv/Scripts/python.exe -m pytest -q -m "not heavy"
cd web && npm run build
```

Both green. Then commit:

```bash
git add SETUP.md .env.example
git commit -m "docs: correction pass setup, claude login, LLM env settings"
```

---

## Deferred, deliberately

- **Live `claude -p` run on a real clip** — needs a human on the host with a logged-in CLI and a real transcript. The provider was verified against the real CLI on 2026-09-05 for the JSON/stdin contract only.
- **Re-running `correct` on an already-complete clip** — no path exists; `requeue_job` only clears `failed`. Add a "Re-check" button when the ledger has grown enough to matter.
- **Ledger `romanized` and the UI's romanization come from two ports of the same library** (Python `gurmukhiutils`, JS `gurmukhi-utils`); they agree on letters but may differ on diacritics. Search uses ILIKE on both, so this is cosmetic.
- **Embedding upgrade** — swapping `HashingEmbedder` for a real model requires re-learning contexts (rows from the old model only suggest). A `katha.tools.reembed` script belongs in the retrieval slice.
- **`words` rows are not rewritten** when a correction changes `text_pa` (pre-existing caveat).
- **Verse detection, facts, translation, RAG, auth** — next slices, per the spec's phasing.
