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
