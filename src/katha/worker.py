import json
import logging
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg

from katha import repo
from katha.config import Settings, get_settings
from katha.db import connect
from katha.embed import HashingEmbedder
from katha.fsutil import write_text_atomic
from katha.ingest.watcher import scan_once
from katha.llm import make_provider
from katha.stages.asr import SarvamAsr, parse_diarized
from katha.stages.assemble import (
    alignment_text,
    assemble_segments,
    infer_reciter_speaker_id,
)
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
from katha.stages.prepare import prepare_clip
from katha.stages.types import AlignedWord, DiarizedEntry

log = logging.getLogger("katha.worker")

NEXT_STAGE = {
    "prepare": "asr",
    "asr": "align",
    "align": "assemble",
    "assemble": "correct",
    "correct": "complete",
}


class WorkerError(RuntimeError):
    """Raised when a stage's durable, on-disk input is missing or unreadable —
    e.g. align running before asr's Sarvam JSON exists on this filesystem."""


# --- on-disk stage handoff --------------------------------------------------
#
# The worker is a `Type=oneshot` systemd unit: a Pipeline object lives for
# exactly one process invocation, fired every 15 minutes. A crash, reboot, or
# retry-after-failure can hand a clip's next stage to a *different* Pipeline
# instance than the one that ran the previous stage. So no stage output may
# live only in Python memory (self._whatever) — every stage re-derives its
# input from durable artifacts on disk (or the DB), and every process is
# treated as a fresh one:
#
#   asr      -> writes  work/<clip_id>/sarvam/*.json  (Sarvam's own output)
#   align    -> reads   work/<clip_id>/sarvam/*.json,  writes work/<clip_id>/words.json
#   assemble -> reads   work/<clip_id>/sarvam/*.json and work/<clip_id>/words.json
#   correct  -> reads   segments (DB); writes work/<clip_id>/flags/<n>.json per LLM
#               window (validated output, atomic), then flags (DB), then runs the
#               ledger pass. A retry after a failed window reuses every window
#               file that already exists, so only the failed window is re-asked.


def _load_entries(work_dir: Path) -> list[DiarizedEntry]:
    """Re-derive the diarized ASR entries for a clip from the durable Sarvam JSON
    that asr.transcribe() downloaded into work/<clip_id>/sarvam/*.json. This is
    what lets `align` and `assemble` run in a process that never ran `asr`."""
    sarvam_dir = work_dir / "sarvam"
    outputs = sorted(sarvam_dir.glob("*.json"))
    if not outputs:
        raise WorkerError(
            f"no Sarvam ASR output found in {sarvam_dir}; the 'asr' stage must "
            "run (and succeed) before this stage can resume"
        )
    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    return parse_diarized(payload)


def _words_path(work_dir: Path) -> Path:
    return work_dir / "words.json"


def _save_words(work_dir: Path, words: list[AlignedWord]) -> None:
    """Persist the aligner's output so `assemble` can run in a fresh process.
    AlignedWord is a frozen dataclass — not picklable/JSON-serialisable as-is —
    so it is explicitly converted to plain dicts here.

    Written atomically: the payload is serialised to a temp file in the same
    directory (guaranteeing the same filesystem) and then moved into place
    with os.replace(), which is atomic on both POSIX and Windows. The stage
    machine advances to 'assemble' the moment this call returns, so a plain
    write_text() leaves a window where Postgres can durably commit
    stage='assemble' while the OS page cache has not yet flushed a
    *partially written* words.json to disk (a power loss or kernel panic in
    that window). Without atomicity, the next _load_words on reboot would
    raise a bare JSONDecodeError forever, and the admin's Retry button would
    keep retrying straight into the same corrupt file — a permanently
    unrecoverable clip. Atomic replace ensures a reader only ever observes
    the fully-old file or the fully-new one, never a partial one.
    """
    payload = [asdict(w) for w in words]
    target = _words_path(work_dir)
    fd, tmp_name = tempfile.mkstemp(dir=work_dir, prefix=".words.", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_words(work_dir: Path) -> list[AlignedWord]:
    path = _words_path(work_dir)
    if not path.exists():
        raise WorkerError(
            f"no aligned words found at {path}; the 'align' stage must run "
            "(and succeed) before this stage can resume"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [AlignedWord(**w) for w in payload]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        # A corrupt/partial words.json must fail loudly and diagnosably, not
        # as a bare JSONDecodeError that gives the admin no clue what's wrong
        # or how to fix it — this is the same failure shape B1 closed for the
        # Retry button, and it must not be reopened here.
        raise WorkerError(
            f"{path} is corrupt or unreadable ({exc!r}); the 'align' stage "
            "must be re-run to regenerate it"
        ) from exc


class Pipeline:
    def __init__(
        self, settings: Settings, conn: psycopg.Connection,
        aligner=None, asr=None, llm=None, embedder=None,
    ):
        self.settings = settings
        self.conn = conn
        self._aligner = aligner
        self._asr = asr
        self._llm = llm
        self.embedder = embedder or HashingEmbedder()

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

    def _provider(self):
        if self._llm is not None:
            return self._llm
        model = repo.get_setting(self.conn, "llm_model", "claude-sonnet-5")
        return make_provider(self.settings, model)

    def _work_dir(self, clip_id: int) -> Path:
        return self.settings.work_dir / str(clip_id)

    def _quota_left(self) -> int:
        limit = repo.get_setting_int(self.conn, "files_per_day", 3)
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        return limit - repo.count_jobs_started_since(self.conn, since)

    # --- stages -----------------------------------------------------------

    def _prepare(self, job: dict) -> None:
        clip = repo.get_clip_internal(self.conn, job["clip_id"])
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
        # transcribe() itself writes the durable JSON to work/sarvam/*.json; the
        # returned entries are not cached here (see the module docstring above) —
        # `_align` re-derives them from that JSON, even in a different process.
        _entries, sarvam_job_id = self.asr.transcribe(work / "asr.wav", work / "sarvam")
        repo.set_job(self.conn, job["id"], sarvam_job_id=sarvam_job_id)

    def _align(self, job: dict) -> None:
        clip_id = job["clip_id"]
        work = self._work_dir(clip_id)
        entries = _load_entries(work)
        words = self.aligner.align(work / "asr.wav", alignment_text(entries))
        _save_words(work, words)

    def _assemble(self, job: dict) -> None:
        clip_id = job["clip_id"]
        work = self._work_dir(clip_id)
        entries = _load_entries(work)
        words = _load_words(work)
        segments = assemble_segments(entries, words, clip_id=clip_id)
        reciter_id = infer_reciter_speaker_id(segments)
        repo.save_segments(self.conn, clip_id, segments, reciter_id)
        try:
            self._cleanup_intermediates(clip_id)
        except Exception:  # noqa: BLE001 — cleanup must never fail a finished clip
            log.exception(
                "clip %s: intermediate cleanup raised; leaving work files in place", clip_id
            )

    def _cleanup_intermediates(self, clip_id: int) -> None:
        """Reclaim the large per-clip intermediates that are dead weight once a
        clip has finished transcribing: the denoised WAV (~518 MB for a 90-minute
        clip) and the 16k ASR WAV (~173 MB). native.wav is deliberately kept —
        the /api/audio endpoint serves it to the admin review UI. Every unlink is
        independently non-fatal: a clip that transcribed fine must not be marked
        failed just because a file couldn't be removed (e.g. still open/locked)."""
        for kind in ("denoised", "asr_ready"):
            row = repo.get_audio_file(self.conn, clip_id, kind)
            if row is None:
                continue
            path = Path(row["path"])
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("clip %s: could not remove %s file %s: %s", clip_id, kind, path, exc)
                continue
            repo.delete_audio_file(self.conn, clip_id, kind)
            log.info("clip %s: reclaimed %s (%s, %s bytes)", clip_id, kind, path, size)

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
                # Written only AFTER validation succeeds: a window that raised
                # must be re-asked on retry, never replayed from a partial file.
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
        """Spec §6 auto-apply. Both gates (surface fuzz + context cosine) must
        pass for an automatic rewrite; a surface-only hit becomes the top-ranked
        suggestion instead. Every rewrite is logged in `corrections` and can be
        reverted from the review UI."""
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

    # --- driver -----------------------------------------------------------

    def run_stage(self, job: dict) -> str:
        {
            "prepare": self._prepare,
            "asr": self._asr_stage,
            "align": self._align,
            "assemble": self._assemble,
            "correct": self._correct,
        }[job["stage"]](job)
        return NEXT_STAGE[job["stage"]]

    def run_once(self) -> bool:
        job = repo.claim_next_job(
            self.conn, self._quota_left(), self.settings.retry_backoff_seconds
        )
        if job is None:
            return False

        try:
            next_stage = self.run_stage(job)
        except Exception as exc:  # noqa: BLE001 — the stage boundary is the retry boundary
            exhausted = job["attempts"] >= self.settings.max_attempts
            status = "failed" if exhausted else "queued"
            log.exception("clip %s failed in %s", job["clip_id"], job["stage"])
            repo.set_job(
                self.conn,
                job["id"],
                status=status,
                last_error=f"{job['stage']}: {exc}",
            )
            return True

        # Each stage boundary is also the retry-budget boundary: a clip that just
        # advanced past a stage successfully gets a full fresh set of attempts for
        # the next one, so a transient failure later (e.g. a CUDA OOM in `align`)
        # doesn't inherit exhaustion from stages that already succeeded.
        #
        # clear_error=True wipes any last_error left over from an earlier failed
        # attempt at *this* stage (or a prior one). Without it, a clip that failed
        # once and then succeeded on retry would carry a stale failure message
        # (e.g. "align: CUDA OOM") all the way to a fully successful clip, and
        # repo.list_clips/the dashboard would render it as a red pill forever.
        repo.set_job(
            self.conn,
            job["id"],
            stage=next_stage,
            status="done" if next_stage == "complete" else "queued",
            attempts=0,
            clear_error=True,
        )
        return True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    with connect(settings) as conn:
        older_than = datetime.now(timezone.utc) - timedelta(
            minutes=settings.stale_running_minutes
        )
        reclaimed = repo.reclaim_stale_jobs(conn, older_than)
        if reclaimed:
            log.warning("reclaimed %d stale running job(s) stuck in 'running'", reclaimed)

        new = scan_once(conn, settings)
        log.info("registered %d new clip(s)", len(new))
        pipeline = Pipeline(settings, conn)
        while pipeline.run_once():
            pass
    log.info("queue drained")


if __name__ == "__main__":
    main()
