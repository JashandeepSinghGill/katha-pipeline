"""Pure half of the `correct` stage: how a clip is cut into LLM windows, what
the LLM is asked, and how its answer is validated into ProposedFlags. No I/O,
no DB — the worker owns those (worker._correct)."""
import json
from dataclasses import dataclass

KINDS = ("proper_noun", "verse", "date", "number", "script", "garbled", "other")
MAX_CANDIDATES = 5

FLAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["flags"],
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["segment_idx", "span_text", "kind", "reason", "candidates"],
                "properties": {
                    "segment_idx": {"type": "integer"},
                    "span_text": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "reason": {"type": "string"},
                    "candidates": {
                        "type": "array", "minItems": 1, "maxItems": MAX_CANDIDATES,
                        "items": {"type": "string"},
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True)
class SegmentText:
    idx: int
    role: str
    text: str


@dataclass(frozen=True)
class Window:
    index: int
    target_idx: tuple[int, ...]
    segments: list[SegmentText]  # targets plus context, in transcript order


@dataclass(frozen=True)
class ProposedFlag:
    segment_idx: int
    span_text: str
    kind: str
    reason: str
    candidates: tuple[str, ...]


def make_windows(segments: list[SegmentText], size: int, context: int) -> list[Window]:
    if not segments:
        return []
    size = max(1, size)
    context = max(0, context)
    windows: list[Window] = []
    for w, start in enumerate(range(0, len(segments), size)):
        stop = min(start + size, len(segments))
        lo = max(0, start - context)
        hi = min(len(segments), stop + context)
        windows.append(Window(
            index=w,
            target_idx=tuple(s.idx for s in segments[start:stop]),
            segments=list(segments[lo:hi]),
        ))
    return windows


SYSTEM_PROMPT = """You are proofreading an automatic speech-recognition (ASR) transcript of Sikh katha in Punjabi, written in Gurmukhi script. The discourse expounds Bhai Santokh Singh's Sri Nanak Prakash (the life of Guru Nanak Sahib, in Braj verse written in Gurmukhi). One speaker recites verses (role "reciter"); another explains them in spoken Punjabi (role "kathakar"). Verse lines also get re-quoted inside the commentary.

Your job is to find spans the ASR probably got WRONG and propose fixes. Flag only when you would bet the transcript differs from what was said. Typical errors:
- proper_noun: garbled names of Gurus, Sikhs, places, rulers, books (a Gurmukhi spelling that is not a real name but sounds like one).
- verse: a broken or nonsensical line of Braj verse.
- date: an implausible year, Bikrami/Nanakshahi date, or age.
- number: a number that contradicts the surrounding text.
- script: Latin-script words that should be Gurmukhi (e.g. "Guru" for ਗੁਰੂ).
- garbled: a nonsense word or phrase that is clearly a mishearing.
- other: anything else you are confident is a recognition error.

Do NOT flag: dialect, colloquial grammar, disfluencies, honorific style, spelling variants that are both acceptable, or anything you merely find unusual. Never rewrite whole sentences. Prefer the smallest span that contains the error.

Rules for each flag:
- span_text must be copied VERBATIM from the segment text (exact substring, same codepoints, no added or removed characters).
- Only flag segments marked "target": true. Segments with "target": false are context only.
- candidates: 1 to 5 corrected Gurmukhi spellings of the span, most likely first. Gurmukhi only (except when the correct form genuinely is a Latin word). Never repeat the original span as a candidate.
- reason: one short sentence, in English.

Respond with JSON only, matching the schema you were given: {"flags": [...]}. If nothing needs flagging, respond {"flags": []}.
"""


def build_system_prompt(gazetteer: list[tuple[str, str]]) -> str:
    if not gazetteer:
        return SYSTEM_PROMPT
    lines = "\n".join(f"- {canonical} ({entity_type})" for canonical, entity_type in gazetteer)
    return (
        SYSTEM_PROMPT
        + "\nKnown correct spellings (from the correction ledger). Prefer these forms "
        "when a span looks like one of them:\n" + lines + "\n"
    )


def build_user_prompt(window: Window) -> str:
    targets = set(window.target_idx)
    head = (
        "Transcript segments, one JSON object per line, in order. "
        f"Flag only segments with \"target\": true (indices {window.target_idx[0]}"
        f"-{window.target_idx[-1]}).\n\n"
    )
    body = "\n".join(
        json.dumps(
            {"idx": s.idx, "role": s.role, "target": s.idx in targets, "text": s.text},
            ensure_ascii=False,
        )
        for s in window.segments
    )
    return head + body + "\n"


def validate_flags(payload: object, window: Window) -> tuple[list[ProposedFlag], int]:
    """Keep only flags the worker can act on. Returns (flags, dropped_count)."""
    raw = payload.get("flags") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return [], 0
    text_by_idx = {s.idx: s.text for s in window.segments}
    targets = set(window.target_idx)
    kept: list[ProposedFlag] = []
    dropped = 0
    for item in raw:
        flag = _validate_one(item, targets, text_by_idx)
        if flag is None:
            dropped += 1
        else:
            kept.append(flag)
    return kept, dropped


def _validate_one(
    item: object, targets: set[int], text_by_idx: dict[int, str]
) -> ProposedFlag | None:
    if not isinstance(item, dict):
        return None
    idx = item.get("segment_idx")
    if not isinstance(idx, int) or isinstance(idx, bool) or idx not in targets:
        return None
    span = str(item.get("span_text") or "").strip()
    if not span or span not in text_by_idx[idx]:
        return None
    seen: set[str] = set()
    candidates: list[str] = []
    raw_candidates = item.get("candidates")
    for c in raw_candidates if isinstance(raw_candidates, list) else []:
        c = str(c).strip()
        if c and c != span and c not in seen:
            seen.add(c)
            candidates.append(c)
    if not candidates:
        return None
    kind = item.get("kind")
    if kind not in KINDS:
        kind = "other"
    return ProposedFlag(
        segment_idx=idx,
        span_text=span,
        kind=kind,
        reason=str(item.get("reason") or ""),
        candidates=tuple(candidates[:MAX_CANDIDATES]),
    )


def flags_to_json(flags: list[ProposedFlag]) -> str:
    return json.dumps(
        [{"segment_idx": f.segment_idx, "span_text": f.span_text, "kind": f.kind,
          "reason": f.reason, "candidates": list(f.candidates)} for f in flags],
        ensure_ascii=False,
    )


def flags_from_json(text: str) -> list[ProposedFlag]:
    return [
        ProposedFlag(d["segment_idx"], d["span_text"], d["kind"], d["reason"], tuple(d["candidates"]))
        for d in json.loads(text)
    ]
