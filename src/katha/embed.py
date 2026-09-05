"""Context embeddings for the correction ledger.

The Embedder protocol is the seam the retrieval slice will plug a real
multilingual model into. HashingEmbedder is the dependency-free default: a
signed character-n-gram hash, L2-normalised, so cosine similarity measures
lexical overlap of the eight-word context around a flagged span — which is
the right fidelity for "is this the same kind of sentence the correction was
learned in". Its `name` is persisted on every ledger row; rows embedded by a
different model can only ever *suggest*, never auto-apply (see stages/ledger).
"""
import hashlib
import math
from typing import Protocol

from katha.text import normalize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    name = "hash-cng-256"
    dim = 256

    def __init__(self, ngrams: tuple[int, ...] = (2, 3, 4)):
        self.ngrams = ngrams

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        t = f" {normalize(text)} "
        if not t.strip():
            return vec
        for n in self.ngrams:
            for i in range(len(t) - n + 1):
                h = hashlib.blake2b(t[i : i + n].encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                vec[idx] += 1.0 if h[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def vector_to_text(v: list[float]) -> str:
    """pgvector's text input form. Written with `%s::vector`."""
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def vector_from_text(s: str) -> list[float]:
    body = s.strip()[1:-1].strip()
    return [float(x) for x in body.split(",")] if body else []
