from katha.stages.types import AlignedWord, AssembledSegment, DiarizedEntry


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def alignment_text(entries: list[DiarizedEntry]) -> str:
    """The exact text handed to the aligner. Building it FROM the entries, in order,
    is what makes the word-to-entry walk in assemble_segments() sound."""
    return " ".join(e.text.strip() for e in entries if e.text.strip())


def _from_asr(idx: int, entry: DiarizedEntry) -> AssembledSegment:
    return AssembledSegment(
        idx=idx,
        speaker_id=entry.speaker_id,
        text_pa=entry.text.strip(),
        start_ms=_ms(entry.start_s),
        end_ms=_ms(entry.end_s),
        align_score=None,
        timing_source="asr",
        words=[],
    )


def assemble_segments(
    entries: list[DiarizedEntry], words: list[AlignedWord]
) -> list[AssembledSegment]:
    entries = [e for e in entries if e.text.strip()]
    if not entries:
        return []

    expected = sum(len(e.text.split()) for e in entries)
    if len(words) != expected:
        # The sequential walk is only safe if every token got a word. It didn't —
        # so don't guess. Fall back to the ASR's own (looser) chunk timings and
        # let the UI flag them.
        return [_from_asr(i, e) for i, e in enumerate(entries)]

    segments: list[AssembledSegment] = []
    cursor = 0
    for idx, entry in enumerate(entries):
        n = len(entry.text.split())
        span = words[cursor : cursor + n]
        cursor += n
        segments.append(
            AssembledSegment(
                idx=idx,
                speaker_id=entry.speaker_id,
                text_pa=entry.text.strip(),
                start_ms=_ms(span[0].start_s),
                end_ms=_ms(span[-1].end_s),
                align_score=sum(w.score for w in span) / n,
                timing_source="aligned",
                words=span,
            )
        )
    return segments


def infer_reciter_speaker_id(segments: list[AssembledSegment]) -> str | None:
    """Heuristic only: the kathakar talks far more than the reciter recites, so the
    speaker with the least total airtime is the reciter. Admin can override per clip —
    see the Global Constraints on why diarization cannot classify verses on its own."""
    totals: dict[str, int] = {}
    for s in segments:
        totals[s.speaker_id] = totals.get(s.speaker_id, 0) + (s.end_ms - s.start_ms)
    if len(totals) < 2:
        return None
    return min(totals, key=lambda k: totals[k])


def role_for(speaker_id: str, reciter_id: str | None) -> str:
    if reciter_id is None:
        return "unknown"
    return "reciter" if speaker_id == reciter_id else "kathakar"
