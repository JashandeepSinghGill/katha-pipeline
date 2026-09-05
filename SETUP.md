# Katha Pipeline — Setup Guide

Drop a recording of Nanak Prakash katha into a folder. Get back a speaker-labelled Punjabi transcript with word-accurate timestamps, reviewable in a browser with the exact audio span playing back.

This guide covers local development first (so you can see it work), then the real deployment on Zira.

---

## How it fits together

```
  watch folder
       │
       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  worker  (systemd timer, every 15 min)                       │
  │                                                              │
  │  prepare ──▶ asr ──▶ align ──▶ assemble ──▶ correct          │
  │    ffmpeg    Sarvam   MMS       merge        Claude (-p)     │
  │  DeepFilter  (cloud)  (local)   + roles      + ledger        │
  └──────────────────────────────────────────────────────────────┘
       │                              │
       ▼                              ▼
  work dir (audio)              Postgres (segments, words, jobs)
       │                              │
       └──────────┬───────────────────┘
                  ▼
            FastAPI  :8000   ◀── serves audio via HTTP range requests
                  │
                  ▼
            Next.js  :3000   ◀── the admin review UI
```

**One thing worth understanding before you start.** Sarvam's ASR returns *sentence-level* timestamps only — despite its JSON key being confusingly named `words`, those elements are whole sentences. So **all word-level timing comes from the local forced aligner**, not from ASR. When alignment can't map cleanly onto a clip, the pipeline falls back wholesale to Sarvam's looser sentence spans and flags every segment as **"loose timing"** in the UI. If you see that badge, that clip's audio sync is only as tight as a sentence.

---

## Part 1 — Things you must obtain first

These block everything else.

### 1. A Sarvam API key

Sign up at [sarvam.ai](https://sarvam.ai) and create a key. New accounts get ₹100 of free credit — roughly two hours of diarized audio.

Cost is **₹45 per hour** of audio with diarization (₹30 without). At the default 3 hours/day that's about ₹135/day (~US$1.60).

### 2. `ffmpeg` and `ffprobe`

Both are hard requirements — the aligner shells out to `ffmpeg` internally.

```bash
sudo apt install ffmpeg          # Debian/Ubuntu
winget install Gyan.FFmpeg       # Windows
```

Verify: `ffmpeg -version` and `ffprobe -version` both print something.

### 3. Python 3.11+ and Docker

Python 3.13 is what this was built and tested on. Docker runs Postgres.

### 4. Node 20+ (for the admin UI)

### 5. A GPU — optional but wanted

Alignment of a 90-minute clip runs comfortably on GPU and slowly on CPU. Everything works on CPU; it's just slower. Set `KATHA_DEVICE` accordingly.

> **Licence note.** The forced aligner (`ctc-forced-aligner`) and its MMS model are **CC-BY-NC 4.0 — non-commercial use only**. That's fine for a personal or parchar project. If this ever becomes commercial, you must swap it for NeMo Forced Aligner + `ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large`. That change touches one file (`src/katha/stages/align.py`).

### 6. Claude Code CLI, logged in

The correction pass runs `claude -p` on your Max subscription. Install Claude Code, then log in **as the user that will run the worker**:

```bash
claude login
```

The login is OAuth and expires after some hours to days. When it has lapsed, the dashboard shows `correct: claude -p reported an error: Not logged in` — run `claude login` again and press **Retry**. If you would rather pay per token, set `KATHA_LLM_BACKEND=anthropic_api` and `KATHA_ANTHROPIC_API_KEY` instead.

---

## Part 2 — Local setup

### Clone and configure

```bash
git clone <your-repo> katha-pipeline
cd katha-pipeline
cp .env.example .env
```

Now **edit `.env`**. At minimum set these three:

```bash
POSTGRES_PASSWORD=katha
KATHA_DATABASE_URL=postgresql://katha:katha@localhost:5432/katha
KATHA_SARVAM_API_KEY=<your key>
```

For local use, also override the paths and turn off denoising (see below for why):

```bash
KATHA_WATCH_DIR=./data/incoming
KATHA_WORK_DIR=./data/work
KATHA_DEVICE=cpu
KATHA_DENOISE_ENABLED=false
```

```bash
mkdir -p data/incoming data/work
```

> **Why denoising is off locally.** DeepFilterNet has no Python 3.13 wheels, so it lives in its own separate venv and is invoked as a subprocess. That's a deliberate isolation (its torch pin conflicts with the aligner's). Setting it up is a *deployment* step — see Part 4. Locally, just leave it off.

### Start Postgres

```bash
docker compose up -d db
docker compose exec db createdb -U katha katha_test   # the test database
```

The compose file pins `pgvector/pgvector:0.8.5-pg17-trixie`. Don't float that tag — versions below 0.8.2 have a buffer-overflow CVE.

> `POSTGRES_PASSWORD` is only read when Docker **first creates the volume**. Changing it in `.env` later won't change the password of an existing database. If you need to change it: `docker compose down -v` (this destroys the data).

### Install Python dependencies

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"            # Linux/macOS
# .venv\Scripts\pip install -e ".[dev]"      # Windows
```

Or with `uv`, which is much faster:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

### Install the forced aligner

This is a separate step because it pulls in torch (a large download).

```bash
.venv/bin/pip install -e ".[align]"

# CPU-only torch (smaller, no CUDA):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The MMS alignment model (~1.2 GB) downloads from HuggingFace on first use and is then cached.

### Create the schema

```bash
.venv/bin/python -m alembic upgrade head
```

### Verify it all works

```bash
.venv/bin/python -m pytest -m "not heavy"
```

Expect **182 passed, 1 deselected**. If that's green, your database, schema and config are all correct.

The deselected test is the real forced-alignment run. It downloads the model and takes **~50 minutes** on CPU. Run it once, when you have time:

```bash
.venv/bin/python -m pytest -m heavy
```

It proves the aligner returns **Gurmukhi** (`ਵਾਹਿਗੁਰੂ`), not romanized text — the single assumption the whole design rests on.

### Install the admin UI

```bash
cd web
npm install
npm run build
cd ..
```

> **Do not upgrade TypeScript.** It's pinned to 6.x deliberately. TypeScript 7 is a native-compiler rewrite whose package layout crashes Next.js 16's build-time TS bootstrap. `gurmukhi-utils` is likewise pinned to exactly `3.2.2` — 4.x is unpublished and the scoped package is a stale mirror.

---

## Part 3 — Running it

Three processes. Three terminals (or use the systemd units in Part 4).

```bash
# 1. The API
.venv/bin/uvicorn katha.api.main:app --port 8000

# 2. The admin UI
cd web && npm run dev

# 3. The worker — drains the queue once, then exits
.venv/bin/python -m katha.worker
```

Now drop a katha file into your watch folder and run the worker:

```bash
cp ~/some-katha.mp3 ./data/incoming/
.venv/bin/python -m katha.worker
```

You should see it register the clip, then walk it through four stages. A 90-minute clip takes a while — Sarvam's batch job alone is several minutes.

Then open **http://localhost:3000**. Your clip should be listed. Click it once it says `complete · done`.

### What to check in the review view

1. The transcript renders in Gurmukhi, with a romanized reading aid under each line.
2. Clicking ▶ on a segment plays **only that span** and stops at its end.
3. Reciter and kathakar segments are visually distinct.
4. The speaker-role buttons flip every segment's role. (The reciter is *guessed* from airtime — verify it by ear. Diarization can't reliably identify the reciter, which is exactly why this override exists.)
5. Double-click a line to correct it. It saves, and shows `edited`.
6. Seeking deep into the file is instant — check the browser network tab for `206` responses.

### The correction pass

After `assemble`, the worker runs one more stage, **`correct`**: Claude reads the transcript in windows (40 segments at a time, with 3 neighbours as context) and flags spans that are probably ASR errors — garbled names, broken verse lines, impossible dates, Latin-script leakage — each with one to five Gurmukhi candidates. Nothing is written to the transcript by the model itself.

In the review view a flagged span is **highlighted** in the Gurmukhi line, with a panel underneath:

- Click a **candidate** (shown in Gurmukhi with a romanized reading) to apply it. Or type your own and press **Apply**. Or **Reject**.
- Tick **Remember as** *person / place / …* before applying to teach the **correction ledger**: the misrecognition, the correct spelling, and the eight words around it are stored. Next time the same misrecognition appears *in a similar context*, the worker applies it automatically and shows an **auto-fixed** pill with the similarity score and a **Revert** button. If the context differs, the ledger's spelling is merely ranked first among the candidates.
- The **Ledger** page (top nav) lets you search, edit, add and retire entries, and tune each entry's context threshold. The **Settings** page holds the model id, window size, thresholds and an auto-apply on/off switch.

Each window's validated output is cached in `work/<clip>/flags/<n>.json`, so a failed or interrupted pass only re-asks the windows that never completed. A clip whose correction pass fails terminally is still reviewable — the transcript exists — and **Retry** re-runs only that stage.

---

## Part 4 — Deploying to Zira

### Layout

```
/opt/katha/              the repo
/opt/katha/.venv/        Python venv
/opt/katha/.env          config (chmod 600 — it holds the API key)
/opt/venvs/deepfilternet/  the denoiser, isolated
/srv/media/katha/incoming/  watch folder
/srv/media/katha/work/      per-clip working files
```

### Zira as built (2026-09-05)

The pipeline lives in its own **unprivileged LXC, CT 213 `katha`** on the Proxmox host `pve` (10.0.0.7), created with `pct create` next to the existing servarr (212) and vault (211) containers:

| | |
|---|---|
| Address | `10.0.0.14/24` on `vmbr0`, gateway 10.0.0.1 |
| OS / Python | Ubuntu 24.04 template, system Python 3.12; the denoiser venv uses a uv-managed 3.11 |
| Resources | 8 cores, 12 GB RAM, 2 GB swap, 40 GB rootfs on the `data` pool (`/mnt/data`) |
| `/srv/media/katha/archive` | bind mount of the host's `/mnt/media/KATHA` (the 432-file katha archive), **read-only** |
| `/srv/media/katha/work` | bind mount of the host's `/mnt/data/katha` (670 GB free), owned by uid 1000 |
| Denoiser | `/opt/deep-filter/deep-filter` (Rust, streaming — the one the pipeline uses); the python CLI is also installed at `/opt/venvs/deepfilternet` for experiments |
| GPU | the NVIDIA T550's device nodes are passed through (frigate's pattern, with the current `504` major for `nvidia-uvm`), but **no NVIDIA userspace is installed inside the CT yet** — `KATHA_DEVICE=cuda` needs the driver's user libs (`NVIDIA-Linux-x86_64-595.80.run --no-kernel-module`, matching the host's 595.80) |
| Access | `pct exec 213 -- …` from the host, or `pct enter 213`. Root **password** SSH is disabled by the template (`PermitRootLogin prohibit-password`); add your SSH key or flip that line in `/etc/ssh/sshd_config` yourself |

The archive is mounted read-only on purpose: the watcher only *reads* the watch folder (it never moves files), so point `KATHA_WATCH_DIR` at `/srv/media/katha/archive/SGND JI` and the daily throttle will register three files a day.

### Set up

```bash
sudo useradd -r -s /bin/false katha
sudo mkdir -p /opt/katha /srv/media/katha/{incoming,work}
sudo chown -R katha:katha /opt/katha /srv/media/katha

# as the katha user, in /opt/katha:
git clone <your-repo> .
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,align]"
cp .env.example .env      # then edit it
chmod 600 .env
```

Set these in `/opt/katha/.env`:

```bash
KATHA_WATCH_DIR=/srv/media/katha/incoming
KATHA_WORK_DIR=/srv/media/katha/work
KATHA_DEVICE=cuda                    # or cpu
KATHA_DENOISE_ENABLED=true
KATHA_DENOISE_BIN=/opt/venvs/deepfilternet/bin/deepFilter
KATHA_SARVAM_API_KEY=<your key>
```

### Install the denoiser

DeepFilterNet ships two front-ends, and only one of them can handle a whole katha clip:

| | `deep-filter` (Rust release binary) | `deepFilter` (pip package CLI) |
|---|---|---|
| Memory on a 43-min clip | **1.0 GB**, constant (streams) | ~9.5 MB per second of audio → **24 GB, OOM-killed** on Zira |
| Speed on 8 CPU cores | 5 min 58 s for 43 min (≈7× realtime) | faster per second, but only on clips that fit in RAM |
| Delay compensation | **must pass `-D`** (the pipeline does) — otherwise the output trails by 30 ms | on by default |
| Quality | identical noise floor (−98 dBFS on the same excerpt) | — |

So the pipeline defaults to the Rust binary (`KATHA_DENOISE_FLAVOR=rust`). Install it with one download — it is a static musl build with the DeepFilterNet3 model embedded:

```bash
sudo mkdir -p /opt/deep-filter
sudo curl -L -o /opt/deep-filter/deep-filter \
  https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl
sudo chmod +x /opt/deep-filter/deep-filter
/opt/deep-filter/deep-filter --version      # deep_filter 0.5.6
```

Set `KATHA_DENOISE_BIN=/opt/deep-filter/deep-filter`. (There is a Windows build too, `deep-filter-0.5.6-x86_64-pc-windows-msvc.exe`, if you want denoising on locally.)

<details>
<summary>The python CLI, if you ever need it (short files only)</summary>

`deepfilterlib` 0.5.6 ships Linux wheels for CPython 3.10 and 3.11 only (newer Pythons make pip try to compile it and demand Rust), and it needs `numpy<2`. On Ubuntu 24.04 (system Python 3.12) build it on a uv-managed 3.11:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 /opt/venvs/deepfilternet
PY=/opt/venvs/deepfilternet/bin/python
uv pip install --python $PY torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $PY "deepfilternet==0.5.6" soundfile "numpy<2"
/opt/venvs/deepfilternet/bin/deepFilter --help
```

Then `KATHA_DENOISE_FLAVOR=python` and `KATHA_DENOISE_BIN=/opt/venvs/deepfilternet/bin/deepFilter`. Remember the memory figure above before pointing it at a real clip.
</details>

### Log the service user into Claude

The worker runs `claude -p` as `katha`, so that user needs its own login:

```bash
sudo -u katha -H claude login      # follow the browser OAuth prompt
sudo -u katha -H claude -p --safe-mode --output-format json --tools "" "say ok"
```

The second command must print a JSON object with `"is_error":false`. The token expires periodically; re-run the login when the dashboard reports `Not logged in`. Keep `KATHA_ANTHROPIC_API_KEY` in `.env` as the fallback and flip `KATHA_LLM_BACKEND=anthropic_api` if you need the pipeline to keep going while you're away.

### Database and schema

```bash
cd /opt/katha
docker compose up -d db
.venv/bin/python -m alembic upgrade head
```

### Build the UI

The UI needs to know where the API is. **This is the easiest thing to get wrong**, so read the box below.

```bash
cd /opt/katha/web
cp .env.local.example .env.local
```

Edit `web/.env.local`:

```bash
NEXT_PUBLIC_API=http://zira.local:8000     # NOT localhost — see below
```

```bash
npm ci
npm run build
```

> **Two traps here.**
>
> **`localhost` means the *browser's* machine, not Zira.** The default is `http://localhost:8000`, which works only when you're browsing *on* Zira. Open the UI from your laptop and it will try to reach port 8000 on **your laptop** and show a connection error. Use Zira's actual hostname or IP.
>
> **`NEXT_PUBLIC_*` is baked in at build time, not read at runtime.** If you change it, you must re-run `npm run build`. Restarting the service is not enough.
>
> Note this also means the API must be reachable from your browser — which conflicts with binding it to `127.0.0.1`. Until you add authentication (Part 7), the safe way to reach the UI remotely is an **SSH tunnel**, not opening the port:
> ```bash
> ssh -L 3000:localhost:3000 -L 8000:localhost:8000 zira
> ```
> Then browse `http://localhost:3000` on your own machine, and the default `NEXT_PUBLIC_API` is correct after all.

### Install the services

```bash
sudo cp /opt/katha/deploy/*.service /opt/katha/deploy/*.timer /etc/systemd/system/
sudo systemctl daemon-reload

sudo systemctl enable --now katha-api.service
sudo systemctl enable --now katha-web.service
sudo systemctl enable --now katha-worker.timer

systemctl list-timers katha-worker.timer
```

The worker is a **oneshot** unit fired by a timer every 15 minutes — it drains the queue and exits. That's intentional: crashes are cleaned up by systemd, and the daily throttle lives in the database. Overlapping runs are safe (the job claim uses `FOR UPDATE SKIP LOCKED`).

Watch it work:

```bash
journalctl -u katha-worker -f
```

### Confirm end to end

```bash
cp /path/to/one-katha.mp3 /srv/media/katha/incoming/
# wait for the timer, or force it:
sudo systemctl start katha-worker.service
journalctl -u katha-worker -n 50
```

---

## Part 5 — Operating it

### The throttle

`files_per_day` lives in the **database**, not `.env`:

```sql
UPDATE settings SET value = '5' WHERE key = 'files_per_day';
```

It defaults to `3`. It gates only the **start of a new clip** — a clip already in flight always finishes, regardless of quota. At 3 hours/day, a 100-hour archive takes about 33 days.

### Disk

This is the thing that will bite you first if you ignore it.

Per clip the worker writes `native.wav` (~520 MB for 90 min), a denoised copy (~520 MB), and `asr.wav` (~170 MB). After `assemble` succeeds it **deletes the denoised copy and `asr.wav`** — but **`native.wav` is kept forever**, because the API streams it to the review UI.

So `WORK_DIR` grows by roughly **520 MB per clip** permanently — about **570 GB/year** at 3 clips/day. Put it on the media LV with room, and plan an archive policy.

### When a clip fails

The dashboard shows the stage, the error, and a **Retry** button.

Retries are per-stage: each stage gets `KATHA_MAX_ATTEMPTS` (3) tries, and the counter resets when a stage succeeds. A previously-failed job waits `KATHA_RETRY_BACKOFF_SECONDS` (5 min) before being retried, so a brief network blip can't burn the budget instantly. After 3 failed tries a job goes **terminal** — only the Retry button brings it back.

If the worker is killed mid-stage, the job sits in `running` until `KATHA_STALE_RUNNING_MINUTES` (6h) passes, then is automatically requeued.

### "Loose timing" badges

A non-zero **Loose timings** count on the dashboard means forced alignment failed to map onto that clip, and the pipeline fell back to Sarvam's sentence-level spans. Those segments' playback spans are only sentence-accurate. Check `journalctl -u katha-worker` — the fallback logs both the expected and actual word counts.

---

## Part 6 — Troubleshooting

**`ModuleNotFoundError: No module named 'psycopg2'` during migrations.**
Your `KATHA_DATABASE_URL` uses the `postgres://` alias. Use `postgresql://` — SQLAlchemy resolves the alias to the psycopg2 driver, which this project doesn't install (it uses psycopg 3).

**Tests fail to connect to the database.**
The test DB is separate. Create it: `docker compose exec db createdb -U katha katha_test`. Its password must match `POSTGRES_PASSWORD` in your `.env`, or set `KATHA_TEST_DATABASE_URL` explicitly.

**The aligner returns romanized text (`vaahiguroo` instead of `ਵਾਹਿਗੁਰੂ`).**
Something has broken the `romanize=True` plumbing in `src/katha/stages/align.py`. That flag is *mandatory* — the model's vocabulary is 26 Latin letters with zero Gurmukhi codepoints — but romanization is meant to stay internal. If Gurmukhi isn't coming back out, stop and fix it; every downstream timestamp is meaningless.

**Alignment produces garbage timings but doesn't error.**
Check `KATHA_ALIGN_LANGUAGE` is `pan` (ISO-639-3), **not** `pa`. And check `KATHA_SARVAM_LANGUAGE` is still `pa-IN`. These are two different standards for the same language and unifying them silently degrades output.

**Transcripts are worse than expected.**
Confirm `KATHA_SARVAM_MODEL=saaras:v3`. The Sarvam SDK still *explicitly* sends the legacy `saarika:v2.5` if you don't override it, so the server-side default never applies.

**`TimeoutError` from Sarvam on long clips.**
`KATHA_SARVAM_TIMEOUT_SECONDS` must be large (7200). The SDK's own default is 600s, which is far too short for a 90-minute clip.

**A clip is rejected before upload.**
Sarvam's Batch API caps at **2 hours per file**. Split it.

**Everything fails at `prepare`.**
Almost always a full disk (see § Disk) or `ffmpeg`/`ffprobe` missing from `PATH`.

---

## Part 7 — Read this before exposing anything

**The API has no authentication.** None. CORS is a browser-only control — `curl` ignores it entirely. Anyone who can reach port 8000 can rewrite transcripts, requeue jobs, and stream your audio.

Both systemd units bind to `127.0.0.1` deliberately. Before putting this behind your reverse proxy, **add authentication**. Changing `KATHA_CORS_ORIGIN` alone gives you no protection whatsoever.

---

## What isn't built yet

This is a deliberate vertical slice — ingest through reviewable transcript. Still to come, roughly in order:

- **Correction ledger** with auto-apply, so recurring proper nouns stop needing manual fixes
- **Nanak Prakash verse database** — the big accuracy win. Turns the reciter's track from a speech-recognition gamble into a lookup.
- **BaniDB** for the rare Guru Granth Sahib quotations
- **Verified facts table** and admin ratification
- **English translation**, embeddings, and the retrieval bot

> **One finding worth carrying into the verse work.** Verses are **not** confined to the reciter. In the sample transcript, the kathakar re-quotes verse lines *inside* his own commentary. So verse detection must run on **spans within segments**, not on whole reciter turns. Diarization is a useful prior — it is not a classifier.

**`correct: claude -p reported an error: Not logged in · Please run /login`.**
The worker's user has no valid Claude login. Run `claude login` as that user (on Zira: `sudo -u katha -H claude login`), then press **Retry** on the dashboard. Or switch to `KATHA_LLM_BACKEND=anthropic_api`.

**`correct: claude CLI not found`.**
Set `KATHA_CLAUDE_BIN` to the absolute path of the `claude` binary for the worker's user (systemd units have a minimal `PATH`).

**`correct: … output truncated at max_tokens`** (API backend).
Raise `KATHA_LLM_MAX_TOKENS`, or lower `correct_window_segments` on the Settings page so each call returns fewer flags.

**A correction was applied that shouldn't have been.**
Press **Revert** on the auto-fixed pill — the segment text is restored and the flag reopens. Then open the Ledger page and either raise that entry's threshold or retire it.
