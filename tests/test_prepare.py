import sys
from pathlib import Path

import pytest

from katha.stages import prepare


@pytest.mark.parametrize("flavor, expected", [
    # the python CLI writes <stem>_<model>.wav into its --output-dir
    ("python", Path("/w/1/native_DeepFilterNet3.wav")),
    # the rust binary writes <same basename> into its -o dir, so that dir is a subdir
    ("rust", Path("/w/1/denoised/native.wav")),
])
def test_denoised_path_per_flavor(flavor, expected):
    assert prepare.denoised_path(
        Path("/w/1/native.wav"), Path("/w/1"), "DeepFilterNet3", flavor
    ) == expected


def test_denoise_cmd_python_flavor_is_the_deepfilter_cli_shape():
    src, out = Path("/w/1/native.wav"), Path("/w/1")
    assert prepare.denoise_cmd(src, out, "/v/bin/deepFilter", "DeepFilterNet3", "python") == [
        "/v/bin/deepFilter", str(src), "-o", str(out), "-m", "DeepFilterNet3",
    ]


def test_denoise_cmd_rust_flavor_compensates_delay_and_writes_to_a_subdir():
    src, out = Path("/w/1/native.wav"), Path("/w/1")
    cmd = prepare.denoise_cmd(src, out, "/opt/deep-filter/deep-filter", "DeepFilterNet3", "rust")
    # -D is load-bearing: without it deep-filter's output trails the input by 30 ms
    assert cmd == ["/opt/deep-filter/deep-filter", "-D", "-o", str(out / "denoised"), str(src)]
    # a model *name* is not a path: only a .tar.gz is passed through as -m
    with_model = prepare.denoise_cmd(src, out, "/opt/deep-filter/deep-filter", "/m/DF3.tar.gz", "rust")
    assert with_model[-3:] == ["-m", "/m/DF3.tar.gz", str(src)]


def test_denoise_cmd_and_path_reject_unknown_flavor():
    with pytest.raises(prepare.PrepareError, match="denoise_flavor"):
        prepare.denoise_cmd(Path("/w/1/native.wav"), Path("/w/1"), "x", "m", "cpp")
    with pytest.raises(prepare.PrepareError, match="denoise_flavor"):
        prepare.denoised_path(Path("/w/1/native.wav"), Path("/w/1"), "m", "cpp")


def test_to_mono_wav_cmd_pins_16k_pcm_for_asr():
    cmd = prepare.to_mono_wav_cmd(Path("/a/in.wav"), Path("/a/asr.wav"), 16000)
    assert cmd[:2] == ["ffmpeg", "-y"]
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[-1] == str(Path("/a/asr.wav"))  # platform-native separator on Windows dev


def test_to_mono_wav_cmd_omits_rate_when_preserving_native():
    cmd = prepare.to_mono_wav_cmd(Path("/a/in.mp3"), Path("/a/native.wav"), None)
    assert "-ar" not in cmd
    assert cmd[cmd.index("-ac") + 1] == "1"


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


# --- FIX A: subprocess calls must be bounded, never hang forever ---------------


def test_run_raises_prepare_error_not_timeoutexpired_on_hang():
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(prepare.PrepareError, match="timed out after 1s"):
        prepare._run(cmd, timeout=1)


# --- FIX B: probe() must not crash with a raw IndexError/KeyError on malformed
# ffprobe output (no audio stream, no duration, no sample_rate, unparsable JSON) --


def test_probe_raises_prepare_error_when_no_audio_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prepare,
        "_run",
        lambda cmd, timeout=None: '{"streams": [], "format": {"duration": "10.0"}}',
    )
    with pytest.raises(prepare.PrepareError, match="no audio stream"):
        prepare.probe(tmp_path / "video_only.mp4")


def test_probe_raises_prepare_error_when_no_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prepare,
        "_run",
        lambda cmd, timeout=None: '{"streams": [{"sample_rate": "48000"}], "format": {}}',
    )
    with pytest.raises(prepare.PrepareError, match="duration"):
        prepare.probe(tmp_path / "no_duration.wav")


def test_probe_raises_prepare_error_when_no_sample_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prepare,
        "_run",
        lambda cmd, timeout=None: '{"streams": [{}], "format": {"duration": "10.0"}}',
    )
    with pytest.raises(prepare.PrepareError, match="sample_rate"):
        prepare.probe(tmp_path / "no_sample_rate.wav")


def test_probe_raises_prepare_error_on_unparsable_json(monkeypatch, tmp_path):
    monkeypatch.setattr(prepare, "_run", lambda cmd, timeout=None: "not json")
    with pytest.raises(prepare.PrepareError, match="invalid JSON"):
        prepare.probe(tmp_path / "corrupt.bin")


# --- FIX C: the denoiser's silent-failure guard (no output file => PrepareError) -


def test_prepare_clip_raises_when_denoiser_produces_no_output(tone_wav, tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "denoise_enabled", True)
    monkeypatch.setattr(settings, "denoise_flavor", "python")

    # A stub "denoiser" that exits 0 without writing anything, despite being handed
    # denoise_cmd()'s full arg list ([bin_path, src, "-o", out_dir, "-m", model]).
    # It must be directly executable (no shell=True in _run), so on Windows that's
    # a no-op .bat; elsewhere a no-op shell script.
    if sys.platform == "win32":
        stub = tmp_path / "noop_denoise.bat"
        stub.write_text("@echo off\r\nexit /b 0\r\n")
    else:
        stub = tmp_path / "noop_denoise.sh"
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    monkeypatch.setattr(settings, "denoise_bin", str(stub))

    with pytest.raises(prepare.PrepareError, match="produced no output"):
        prepare.prepare_clip(tone_wav, tmp_path / "work", settings)


def test_prepare_clip_rust_flavor_reads_the_denoised_subdir(tone_wav, tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "denoise_enabled", True)
    monkeypatch.setattr(settings, "denoise_flavor", "rust")
    # A stand-in for deep-filter: argv is [-D, -o, <out_dir>, <src>]; it copies
    # <src> into <out_dir> under the SAME basename, exactly as the real binary does,
    # and refuses to run without -D so a dropped flag fails the test.
    if sys.platform == "win32":
        stub = tmp_path / "fake_deep_filter.bat"
        stub.write_text('@echo off\r\nif not "%~1"=="-D" exit /b 3\r\nif not exist "%~3" mkdir "%~3"\r\ncopy /y "%~4" "%~3\\" >nul\r\nexit /b 0\r\n')
    else:
        stub = tmp_path / "fake_deep_filter.sh"
        stub.write_text('#!/bin/sh\n[ "$1" = "-D" ] || exit 3\nmkdir -p "$3" && cp "$4" "$3/"\n')
        stub.chmod(0o755)
    monkeypatch.setattr(settings, "denoise_bin", str(stub))

    result = prepare.prepare_clip(tone_wav, tmp_path / "work", settings)
    assert result.denoised == tmp_path / "work" / "denoised" / "native.wav"
    assert result.denoised.exists()
    assert result.asr_ready.exists()


def test_prepare_clip_rejects_unknown_flavor_before_doing_any_work(tone_wav, tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "denoise_enabled", True)
    monkeypatch.setattr(settings, "denoise_flavor", "cpp")
    with pytest.raises(prepare.PrepareError, match="denoise_flavor"):
        prepare.prepare_clip(tone_wav, tmp_path / "work", settings)
    assert not (tmp_path / "work" / "native.wav").exists()
