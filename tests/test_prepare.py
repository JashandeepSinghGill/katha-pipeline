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
