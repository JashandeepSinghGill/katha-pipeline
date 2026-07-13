from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiarizedEntry:
    """One Sarvam diarized chunk. Sentence-level, NOT word-level."""
    speaker_id: str
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class AlignedWord:
    """One word from forced alignment. The only source of word-level timing."""
    text: str
    start_s: float
    end_s: float
    score: float


@dataclass(frozen=True)
class AssembledSegment:
    idx: int
    speaker_id: str
    text_pa: str
    start_ms: int
    end_ms: int
    align_score: float | None
    timing_source: str  # 'aligned' | 'asr'
    words: list[AlignedWord] = field(default_factory=list)
