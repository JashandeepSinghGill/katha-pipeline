# Katha Pipeline — Correction Slice: Design

Date: 2026-09-05
Branch: `feat/correction-slice` (built on top of `feat/transcript-slice`)
Spec sections covered: §4 stages 5–6 and part of 8, §6 (ledger), §7 (settings, ledger browser), §10 (Claude Max integration), §11 flag 1.

## Goal

After a clip finishes `assemble`, it gets one more automated pass: Claude reads the Punjabi transcript, flags spans that are probably ASR errors (garbled proper nouns, broken verse lines, impossible dates, Latin-script leakage), and proposes one to five Gurmukhi candidate fixes per flag. A **correction ledger** then auto-applies any flag whose span fuzzy-matches a known misrecognition *and* whose surrounding context matches the context the ledger entry was learned in. Everything else lands in the admin review UI as a one-click choice. Each admin acceptance can teach the ledger, so the flag rate falls over time (spec §6: "what makes accuracy compound").

All LLM calls go through one `LLMProvider` interface with two backends — `claude -p` on the Max subscription and the Anthropic API as fallback — as locked in by the previous plan.

## Non-goals (this slice)

Verse detection against BaniDB / Nanak Prakash (§3, §4.7), the facts table (§7), English translation (§4.9), retrieval embeddings and the RAG bot (§9, §4.11), API authentication. The `Embedder` interface built here is deliberately minimal and will be upgraded by the retrieval slice.

## What was already decided (do not re-litigate)

- Stage machine with durable on-disk handoff between stages; every stage re-derives its input from disk or the DB, never from process memory.
- `LLMProvider` interface, backend chosen by config, nothing else learns which backend is live.
- Gurmukhi is the canonical stored form. Romanization is a display aid only.
- Verses appear *inside* kathakar segments, so nothing here keys on `speaker_role`.
- The daily throttle (`files_per_day`) already bounds LLM usage: the `correct` stage runs once per clip.

## Verified vendor facts (spiked 2026-09-05, do not substitute from memory)

- `claude -p` with `--output-format json --json-schema <schema>` returns one JSON object on stdout whose `structured_output` field is the parsed, schema-validated object. `is_error` is `false` on success; on failure `is_error` is `true` and `result` holds the message (e.g. `"Not logged in · Please run /login"`).
- `--bare` **disables OAuth** ("Anthropic auth is strictly ANTHROPIC_API_KEY"). It must not be used for the subscription path. `--safe-mode` disables CLAUDE.md, hooks, plugins, MCP and skills while "auth, model selection ... work normally" — that is the flag for a daemon.
- `--tools ""` disables every built-in tool. `--system-prompt-file <path>` replaces the default system prompt. The user prompt can be piped on stdin, which sidesteps the Windows 32 KB command-line limit.
- `--model` accepts a full model id (`claude-sonnet-5`) or an alias (`sonnet`). The full id is stored so the same value works for the API backend.
- `gurmukhiutils` 0.3.1 (Python port of the Shabad OS library) provides `convert(text, "guru_latn_pa")` for romanization and `unicode_normalize`. `rapidfuzz` 3.14 provides `fuzz.ratio` on Gurmukhi strings directly (92 for ਪਾਤਸ਼ਾਹ vs ਪਾਤਸਾਹ).
- `pgvector` accepts an untyped `vector` column (no fixed dimension). Python-side `register_vector(conn)` adapts lists/arrays.

## Architecture

```
prepare → asr → align → assemble → correct → complete
                                      │
                       ┌──────────────┴──────────────┐
                       │ 1. window segments           │
                       │ 2. LLMProvider.complete_json │  (cached per window on disk)
                       │ 3. validate → flags          │
                       │ 4. ledger match → auto-apply │
                       └──────────────────────────────┘
                                      │
                    flags / corrections / correction_ledger (Postgres)
                                      │
                  FastAPI  ──  review UI (resolve flags, revert auto-fixes,
                               ledger browser, settings page)
```

### Components

| Unit | Purpose | Depends on |
|---|---|---|
| `katha/llm.py` | `LLMProvider` protocol, `LLMError`, `ClaudeCliProvider`, `AnthropicApiProvider`, `make_provider(settings)` | subprocess, `anthropic` SDK |
| `katha/embed.py` | `Embedder` protocol, `HashingEmbedder`, `cosine()` | stdlib only |
| `katha/stages/correct.py` | Pure: windowing, prompt + schema, validation of LLM output into `ProposedFlag`s | types |
| `katha/stages/ledger.py` | Pure: normalisation, surface fuzzy score, context extraction, match decision, `apply_span` | rapidfuzz, gurmukhiutils, embed |
| `katha/repo.py` (extended) | flags, corrections, ledger, settings CRUD | psycopg |
| `katha/worker.py` (`_correct`) | Orchestrates the stage with per-window disk cache | all above |
| `katha/api/main.py` (extended) | resolve / revert / ledger / settings endpoints, flag counts | repo |
| `web/` | flags in the review view, dashboard counts, `/ledger`, `/settings` | API |

## Data model (migration `0002_corrections`)

```sql
ALTER jobs.stage CHECK adds 'correct'   -- prepare, asr, align, assemble, correct, complete

correction_ledger (
  id                   bigserial PK,
  canonical_gurmukhi   text NOT NULL UNIQUE,
  romanized            text NOT NULL,             -- gurmukhiutils, computed server-side
  surface_forms        text[] NOT NULL DEFAULT '{}',
  entity_type          text NOT NULL CHECK (IN person|place|date|book|title|term|other),
  english_gloss        text,
  context_embedding    vector,                    -- untyped; dimension belongs to embedding_model
  embedding_model      text,
  confidence_threshold real NOT NULL,             -- seeded from settings.ledger_context_threshold
  times_applied        integer NOT NULL DEFAULT 0,
  retired              boolean NOT NULL DEFAULT false,
  created_by           text NOT NULL DEFAULT 'admin',
  created_at           timestamptz NOT NULL DEFAULT now(),
  last_used_at         timestamptz
)

flags (
  id            bigserial PK,
  segment_id    bigint NOT NULL REFERENCES segments ON DELETE CASCADE,
  span_text     text NOT NULL,                    -- verbatim substring of text_pa at flag time
  kind          text NOT NULL CHECK (IN proper_noun|verse|date|number|script|garbled|other),
  reason        text NOT NULL,
  candidates    jsonb NOT NULL,                   -- ["ਗੁਰਮੁਖੀ", …] 1–5, ledger suggestion first if any
  ledger_id     bigint REFERENCES correction_ledger ON DELETE SET NULL,  -- surface-matched entry
  status        text NOT NULL DEFAULT 'open' CHECK (IN open|accepted|rejected|auto_applied|stale),
  resolved_text text,
  resolved_at   timestamptz,
  created_at    timestamptz NOT NULL DEFAULT now()
)
INDEX flags (segment_id), INDEX flags (status)

corrections (                                     -- the reversible application log (spec §6)
  id                 bigserial PK,
  segment_id         bigint NOT NULL REFERENCES segments ON DELETE CASCADE,
  flag_id            bigint REFERENCES flags ON DELETE SET NULL,
  ledger_id          bigint REFERENCES correction_ledger ON DELETE SET NULL,
  source             text NOT NULL CHECK (IN auto|admin),
  before_text        text NOT NULL,               -- the span replaced
  after_text         text NOT NULL,
  context_similarity real,
  applied_at         timestamptz NOT NULL DEFAULT now(),
  reverted_at        timestamptz
)

settings seeds:
  llm_model                 = claude-sonnet-5
  correct_window_segments   = 40
  correct_context_segments  = 3
  ledger_surface_threshold  = 88      -- rapidfuzz ratio, 0–100
  ledger_context_threshold  = 0.80    -- cosine, default for new ledger rows
  auto_apply_enabled        = true
```

Spans are located by **substring search at resolution time**, not by stored character offsets: an admin may have edited the segment between flagging and resolution. If the span is no longer present, the flag becomes `stale` and the API answers 409.

The existing caveat stands: `words` rows are alignment timing and are not rewritten when `text_pa` changes. Nothing in this slice reads `words.text` for display.

## `LLMProvider`

```python
class LLMProvider(Protocol):
    name: str
    def complete_json(self, *, system: str, user: str, schema: dict) -> dict: ...
```

**`ClaudeCliProvider`** runs

```
claude -p --safe-mode --output-format json --json-schema <schema>
       --system-prompt-file <tmp> --tools "" --no-session-persistence
       --model <settings.llm_model>
```

with the user prompt on stdin, `cwd` = the clip's work dir, and `timeout = llm_timeout_seconds`. It parses stdout as JSON, requires `type == "result"` and `is_error is False`, and returns `structured_output`. Anything else — non-zero exit, timeout, unparsable stdout, `is_error`, a missing `structured_output` — raises `LLMError` with the CLI's `result` text so "Not logged in" reaches the dashboard verbatim. Logs `total_cost_usd` and the model actually used. The binary is resolved once with `shutil.which(settings.claude_bin)`; a missing binary is an `LLMError` at construction.

**`AnthropicApiProvider`** uses the `anthropic` SDK with the same system/user prompts and asks for structured JSON output (mechanism per the current SDK docs, checked at implementation time via the `claude-api` skill — not from memory). Selected when `KATHA_LLM_BACKEND=anthropic_api`; needs `KATHA_ANTHROPIC_API_KEY`.

Selection: `make_provider(settings)`; `llm_backend` is an environment setting (it implies credentials), `llm_model` is a DB setting (routing is an admin knob, spec §7).

## The `correct` stage

1. Load the clip's segments (`idx`, `speaker_role`, `text_pa`). No segments → stage succeeds with zero flags.
2. Split into windows of `correct_window_segments` target segments, each padded with `correct_context_segments` neighbours on both sides, marked context-only in the prompt.
3. For window *i*: if `work/<clip>/flags/<i>.json` exists, load it; otherwise call `complete_json`, validate, and write the **validated** result atomically (same `os.replace` pattern as `words.json`). A retry after a failure re-runs only the windows that never completed. No LLM output is ever the only copy of anything.
4. Validation (pure, `correct.py`): the `segment_idx` must be a target of that window; `span_text` must be non-empty and occur verbatim in that segment's `text_pa`; 1–5 candidates, each non-empty, de-duplicated, none equal to the span. Invalid items are dropped and counted in a per-window log line.
5. Persist: `repo.save_flags` deletes this clip's `open` flags and inserts the new set. Resolved flags are never touched (they only exist if the stage already succeeded once, which cannot recur).
6. Ledger pass (`ledger.py` + repo): for each new flag, `match()`; on `auto` → `apply_span` on the segment, insert a `corrections` row (`source='auto'`), flag → `auto_applied`, bump `times_applied`/`last_used_at`; on `suggest` → prepend the entry's canonical form to `candidates` and set `flag.ledger_id`. Skipped entirely when `auto_apply_enabled` is false (suggestions still attach).

**Prompt.** System prompt: the domain (Bhai Santokh Singh's Nanak Prakash; Braj verses in Gurmukhi script; Punjabi commentary; Sikh history names, places, dates), what to flag, what *not* to flag (dialect, disfluency, style), the verbatim-span rule, the 1–5 Gurmukhi candidates rule ordered most-likely-first, and a **gazetteer** of up to 200 non-retired ledger canonical forms with entity types ("known correct spellings"). User prompt: one JSON line per segment `{idx, role, text, target: bool}`. Schema:

```json
{"type":"object","required":["flags"],"properties":{"flags":{"type":"array","items":{
  "type":"object","required":["segment_idx","span_text","kind","reason","candidates"],
  "properties":{"segment_idx":{"type":"integer"},"span_text":{"type":"string"},
    "kind":{"type":"string","enum":["proper_noun","verse","date","number","script","garbled","other"]},
    "reason":{"type":"string"},
    "candidates":{"type":"array","minItems":1,"maxItems":5,"items":{"type":"string"}}}}}}}
```

## Ledger matching (`ledger.py`, pure)

- `normalize(s)`: `gurmukhiutils.unicode_normalize` → NFC → strip dandas/punctuation → collapse whitespace.
- `surface_score(span, entry)` = max `fuzz.ratio(normalize(span), normalize(f))` over the canonical form and every surface form.
- `context_text(text_pa, span_text, words=8)`: the eight words on each side of the span's first occurrence, span excluded.
- `match(span, context_vec, entries, surface_threshold) -> Match | None`: best entry by surface score at or above the threshold. Decision is `auto` when the entry has an embedding from the **same** `embedding_model` and `cosine ≥ entry.confidence_threshold`; otherwise `suggest`. Entries embedded by a different model can only ever suggest — a model swap can't silently produce bad auto-applies.
- `apply_span(text, span, replacement) -> str | None`: replaces the first occurrence; `None` if absent.

Learning: accepting a flag with "remember" upserts the ledger row by `canonical_gurmukhi` — appends the span to `surface_forms` (deduped), and sets `context_embedding` to the L2-normalised mean of the old and new context vectors (or the new one if the models differ). `romanized` is computed server-side.

## `Embedder`

```python
class Embedder(Protocol):
    name: str      # stored in ledger.embedding_model
    dim: int
    def embed(self, text: str) -> list[float]: ...
```

`HashingEmbedder` (`name="hash-cng-256"`): character 2–4-grams of the normalised text, each hashed to a bucket and a sign, summed into 256 dims, L2-normalised. Deterministic, dependency-free, and lexical — which is the right fidelity for an eight-word context window. The retrieval slice replaces it with a real multilingual model; rows embedded by the hash model then downgrade to suggestions until re-learned, by the rule above.

## API

| Endpoint | Behaviour |
|---|---|
| `GET /api/clips` | adds `open_flags`, `auto_applied` counts; UI links any clip with segments |
| `GET /api/clips/{id}` | each segment carries `flags: [...]` and `corrections: [...]` |
| `POST /api/flags/{id}/resolve` | `{action:"accept", text, remember?: {entity_type, english_gloss?}}` or `{action:"reject"}`. Accept replaces the span, marks `edited`, logs a `corrections` row (`admin`), status `accepted`; span absent → status `stale`, 409 |
| `POST /api/corrections/{id}/revert` | restores `before_text` if `after_text` is still present, sets `reverted_at`, reopens the flag; else 409 |
| `GET /api/ledger?q=` | search canonical, romanized, surface forms |
| `POST /api/ledger`, `PATCH /api/ledger/{id}` | create / edit (canonical, surface forms, entity type, gloss, threshold, retired). No delete — retire |
| `GET /api/settings`, `PUT /api/settings` | typed validation per key; unknown keys 400 |

## UI

- **Dashboard**: link when `segment_count > 0`; columns *Flags* (open) and *Auto-fixed*; nav links to Ledger and Settings.
- **Review view**: the flagged span is highlighted inside the Gurmukhi line; beneath it, the reason and candidate buttons (Gurmukhi + romanized via `gurmukhi-utils`), a custom-text input, Reject, and a "Remember" checkbox with an entity-type select. Auto-applied corrections show as a pill (`before → after · sim 0.91`) with **Revert**. Resolved flags collapse to a pill.
- **/ledger**: search box, table with inline edit of canonical / surface forms / type / threshold, Retire toggle.
- **/settings**: one form for the seeded keys plus `files_per_day`.

## Error handling

- Any LLM failure fails the stage; the existing retry/backoff/terminal-`failed` machinery applies, and the per-window cache makes retries cheap.
- Flags on context-only segments, non-verbatim spans, or empty candidates are dropped and logged, never persisted.
- Auto-apply never fires unless the span is found verbatim in the current text.
- A clip whose `correct` stage is terminally failed is still reviewable (its transcript exists); the dashboard shows the error and Retry.

## Testing

- Pure: windowing edges, validation drops, `apply_span`, `normalize`, `surface_score`, `match` (auto / suggest / none / model mismatch), embedder determinism and unit norm, prompt contains the gazetteer.
- Provider: `ClaudeCliProvider` against a stub executable (`.bat` on Windows, `#!/bin/sh` elsewhere — the denoiser test's pattern) for success, `is_error`, bad JSON, non-zero exit, timeout, missing binary. `AnthropicApiProvider` against a fake client.
- DB: repo round-trips; `save_flags` idempotence; ledger upsert merges and re-embeds; revert.
- Worker: with a `FakeProvider` — stage order now ends `correct → complete`; flags persisted; window cache honoured on retry (provider call count); auto-apply end to end.
- API: resolve accept / reject / stale, revert 409, ledger CRUD, settings validation, counts.
- **Deferred to a human**: a live `claude -p` run on a real clip (needs `claude login` on the host, as the service user).

## Settings

Environment (`KATHA_` prefix): `LLM_BACKEND=claude_cli|anthropic_api`, `CLAUDE_BIN=claude`, `LLM_TIMEOUT_SECONDS=900`, `ANTHROPIC_API_KEY=`, `LLM_MAX_TOKENS=8192`.
Database (`settings` table, editable in the UI): the six seeds above plus `files_per_day`.

## Deployment notes

The worker calls `claude` as the `katha` service user, so that user must have run `claude login` once on Zira (OAuth, expires and needs periodic re-login — spec §10). `--safe-mode` keeps the user's `~/.claude` customisations out of the prompt. The API fallback is the mitigation for an expired login: set `KATHA_LLM_BACKEND=anthropic_api`.
