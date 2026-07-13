from pathlib import Path

from katha.config import Settings
from katha.stages.types import AlignedWord

MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"


class AlignError(RuntimeError):
    pass


def to_aligned_words(results: list[dict]) -> list[AlignedWord]:
    if not results:
        raise AlignError("aligner returned no words")
    return [
        AlignedWord(
            text=r["text"],
            start_s=float(r["start"]),
            end_s=float(r["end"]),
            score=float(r.get("score", 0.0)),
        )
        for r in results
    ]


class Aligner:
    """Wraps ctc-forced-aligner. Loads the model once; reuse across clips."""

    def __init__(self, settings: Settings):
        import torch
        from ctc_forced_aligner import load_alignment_model

        self.settings = settings
        self.device = settings.device
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model, self.tokenizer = load_alignment_model(
            self.device, model_path=MODEL, dtype=dtype
        )

    def align(self, wav: Path, text: str) -> list[AlignedWord]:
        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            load_audio,
            postprocess_results,
            preprocess_text,
        )

        if not text.strip():
            raise AlignError("nothing to align: transcript is empty")

        # load_audio shells out to ffmpeg and resamples to 16 kHz mono for us.
        waveform = load_audio(str(wav), self.model.dtype, self.model.device)

        emissions, stride = generate_emissions(
            self.model,
            waveform,
            window_length=30,
            context_length=2,
            batch_size=self.settings.align_batch_size,
        )

        # romanize=True is MANDATORY: the model vocab is 26 Latin letters and has
        # zero Gurmukhi codepoints. Romanization is internal — the words that come
        # back out are the original Gurmukhi. language is ISO-639-3 ('pan'), not 'pa'.
        tokens_starred, text_starred = preprocess_text(
            text,
            romanize=True,
            language=self.settings.align_language,
            split_size="word",
            star_frequency="edges",  # the library default is 'segment'; the CLI's is 'edges'
        )

        segments, scores, blank = get_alignments(emissions, tokens_starred, self.tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        results = postprocess_results(text_starred, spans, stride, scores, merge_threshold=0.0)
        return to_aligned_words(results)
