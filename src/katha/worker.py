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
