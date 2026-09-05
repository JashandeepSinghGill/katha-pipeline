"""Gurmukhi text helpers shared by the ledger matcher, the embedder and the API.

gurmukhiutils is the Python port of the Shabad OS library the web UI already
uses (gurmukhi-utils@3.2.2), so server- and client-side romanization agree.
"""
import re
import unicodedata

from gurmukhiutils.convert import convert
from gurmukhiutils.unicode import unicode_normalize

# Dandas, Latin punctuation, and the dash family. Gurmukhi letters and matras
# are untouched. '॥' (U+0965) and '।' (U+0964) are Devanagari-block
# punctuation that Gurmukhi text uses too.
_PUNCT = re.compile(r"[॥।‖,.;:!?\"'()\[\]{}\-–—]+")


def normalize(s: str) -> str:
    try:
        s = unicode_normalize(s)
    except Exception:  # noqa: BLE001 — a normaliser bug must never break matching
        pass
    s = unicodedata.normalize("NFC", s)
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split()).casefold()


def romanize(s: str) -> str:
    """Display aid only (spec §3.3). Never write the result back as canonical text."""
    if not s:
        return ""
    try:
        return convert(s, "guru_latn_pa")
    except Exception:  # noqa: BLE001
        return s
