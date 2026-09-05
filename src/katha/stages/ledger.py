"""Pure ledger matching (spec §6). Two independent gates:

  1. surface:  rapidfuzz ratio between the flagged span and the entry's
               canonical form or any recorded misrecognition >= threshold
  2. context:  cosine between the embedding of the words around the span and
               the entry's stored context embedding >= the entry's own threshold

Both pass -> 'auto' (the worker applies it and logs a reversible correction).
Only (1) -> 'suggest' (the entry's canonical form is ranked first in the UI).
Entries embedded by a different model than the one running can only suggest:
a model swap must never produce silent bad auto-applies.
"""
import math
from dataclasses import dataclass

from rapidfuzz import fuzz

from katha.embed import cosine
from katha.text import normalize


@dataclass(frozen=True)
class LedgerEntry:
    id: int
    canonical: str
    surface_forms: tuple[str, ...]
    embedding: tuple[float, ...] | None
    embedding_model: str | None
    threshold: float


@dataclass(frozen=True)
class Match:
    entry: LedgerEntry
    surface_score: float
    context_similarity: float | None
    decision: str  # 'auto' | 'suggest'


def surface_score(span: str, entry: LedgerEntry) -> float:
    s = normalize(span)
    if not s:
        return 0.0
    forms = (entry.canonical, *entry.surface_forms)
    return max((fuzz.ratio(s, normalize(f)) for f in forms if normalize(f)), default=0.0)


def context_text(text: str, span: str, words: int = 8) -> str:
    """The `words` whitespace tokens on each side of the span's first
    occurrence, span excluded. Empty when the span is absent or alone."""
    pos = text.find(span) if span else -1
    if pos < 0:
        return ""
    before = text[:pos].split()[-words:]
    after = text[pos + len(span):].split()[:words]
    return " ".join(before + after)


def match(
    span: str,
    context_vec: list[float] | None,
    embedder_name: str,
    entries: list[LedgerEntry],
    surface_threshold: float,
) -> Match | None:
    span_norm = normalize(span)
    best: tuple[float, LedgerEntry] | None = None
    for e in entries:
        if normalize(e.canonical) == span_norm:
            continue  # nothing to correct
        score = surface_score(span, e)
        if score >= surface_threshold and (best is None or score > best[0]):
            best = (score, e)
    if best is None:
        return None
    score, e = best
    similarity: float | None = None
    if (
        context_vec is not None
        and e.embedding is not None
        and e.embedding_model == embedder_name
        and len(e.embedding) == len(context_vec)
    ):
        similarity = cosine(list(e.embedding), context_vec)
    decision = "auto" if similarity is not None and similarity >= e.threshold else "suggest"
    return Match(entry=e, surface_score=float(score), context_similarity=similarity,
                 decision=decision)


def apply_span(text: str, span: str, replacement: str) -> str | None:
    if not span or span not in text:
        return None
    return text.replace(span, replacement, 1)


def merge_embeddings(old: list[float] | None, new: list[float]) -> list[float]:
    if not old or len(old) != len(new):
        return list(new)
    summed = [a + b for a, b in zip(old, new)]
    norm = math.sqrt(sum(v * v for v in summed))
    return [v / norm for v in summed] if norm else list(new)
