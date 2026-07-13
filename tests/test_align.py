from pathlib import Path

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


def _write_synthetic_wav(path: Path, seconds: float = 3.0, sr: int = 16000) -> None:
    """A few seconds of shaped noise, 16 kHz mono — enough for ctc-forced-aligner's
    ffmpeg + feature-extraction pipeline to run end to end. The audio carries no
    real speech, so alignment *accuracy* is meaningless here; only the contract
    (word count, Gurmukhi text, monotonic timestamps) is under test.
    """
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(0)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    tone = 0.05 * np.sin(2 * np.pi * 220 * t)
    noise = 0.02 * rng.standard_normal(n)
    audio = (tone + noise).astype(np.float32)
    sf.write(str(path), audio, sr, subtype="PCM_16")


@pytest.mark.heavy
def test_align_returns_monotonic_gurmukhi_words(settings, tmp_path):
    """Genuinely exercises ctc-forced-aligner + the MMS model offline (no live
    Sarvam run required). Requires torch + `pip install -e ".[align]"`; the
    model (~1.2GB) downloads from HuggingFace on first run.

    The audio is synthetic (shaped noise, no real speech), so the *timings*
    are meaningless — that's expected. What's under test is the contract:
    - one AlignedWord per input word
    - words[0].text is Gurmukhi ("ਵਾਹਿਗੁਰੂ"), NOT romanized — this is the load-
      bearing assertion that proves romanize=True's romanization stays internal
      and never leaks into the returned text.
    - timestamps are monotonically non-decreasing
    """
    try:
        import torch  # noqa: F401
        from ctc_forced_aligner import load_alignment_model  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"align extras not installed: {exc}")

    wav = tmp_path / "synthetic.wav"
    _write_synthetic_wav(wav)

    text = "ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਿਹ"

    try:
        aligner = align.Aligner(settings)
    except Exception as exc:  # model unavailable (e.g. no network for the HF download)
        pytest.skip(f"forced-alignment model unavailable: {exc}")

    words = aligner.align(wav, text)

    assert len(words) == 8
    assert words[0].text == "ਵਾਹਿਗੁਰੂ"  # Gurmukhi out, not romanized
    starts = [w.start_s for w in words]
    assert starts == sorted(starts)  # monotonic
