import json
from pathlib import Path

from katha.config import Settings
from katha.stages.types import DiarizedEntry


class AsrError(RuntimeError):
    pass


def _required_float(entry: dict, field: str, idx: int) -> float:
    try:
        value = entry[field]
    except KeyError:
        raise AsrError(f"entry {idx}: missing required field '{field}'") from None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise AsrError(
            f"entry {idx}: field '{field}' is not a valid number: {value!r}"
        ) from None


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
            start_s=_required_float(e, "start_time_seconds", i),
            end_s=_required_float(e, "end_time_seconds", i),
        )
        for i, e in enumerate(entries)
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

        # A retried stage reuses this same out_dir. Clear any JSON left behind
        # by a prior attempt so `sorted(...)[0]` below can't pick up a stale
        # result instead of this run's.
        for stale in out_dir.glob("*.json"):
            stale.unlink()

        job.download_outputs(output_dir=str(out_dir))
        outputs = sorted(out_dir.glob("*.json"))
        if not outputs:
            raise AsrError(f"Sarvam job completed but wrote no JSON to {out_dir}")

        payload = json.loads(outputs[0].read_text(encoding="utf-8"))
        return parse_diarized(payload), str(getattr(job, "job_id", ""))
