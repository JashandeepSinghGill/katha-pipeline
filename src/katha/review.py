"""Admin actions on flags and corrections. Thin, transactional-per-call, and
HTTP-agnostic so the API layer only maps ReviewError.status to a response."""
import psycopg

from katha import repo
from katha.embed import Embedder
from katha.stages.ledger import apply_span, context_text
from katha.text import romanize


class ReviewError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def accept_flag(
    conn: psycopg.Connection, embedder: Embedder, flag_id: int, text: str, remember: dict | None
) -> dict:
    text = (text or "").strip()
    if not text:
        raise ReviewError(400, "replacement text must not be empty")
    flag = repo.get_flag(conn, flag_id)
    if flag is None:
        raise ReviewError(404, "no such flag")
    if flag["status"] != "open":
        raise ReviewError(409, f"flag is {flag['status']}, not open")
    original = flag["text_pa"]
    new_text = apply_span(original, flag["span_text"], text)
    if new_text is None:
        repo.resolve_flag(conn, flag_id, "stale")
        raise ReviewError(
            409, "the flagged span is no longer present in the segment (marked stale)"
        )

    ledger_id = None
    if remember:
        ctx = context_text(original, flag["span_text"])
        vec = embedder.embed(ctx) if ctx else None
        ledger_id = repo.upsert_ledger(
            conn,
            canonical=text,
            romanized=romanize(text),
            surface_form=flag["span_text"],
            entity_type=remember.get("entity_type") or "other",
            english_gloss=remember.get("english_gloss") or None,
            embedding=vec,
            embedding_model=embedder.name if vec else None,
            threshold=repo.get_setting_float(conn, "ledger_context_threshold", 0.8),
        )

    repo.update_segment_text(conn, flag["segment_id"], new_text)
    correction_id = repo.add_correction(
        conn, segment_id=flag["segment_id"], flag_id=flag_id, ledger_id=ledger_id,
        source="admin", before=flag["span_text"], after=text, similarity=None,
    )
    repo.resolve_flag(conn, flag_id, "accepted", resolved_text=text, ledger_id=ledger_id)
    return {"segment_id": flag["segment_id"], "text_pa": new_text,
            "correction_id": correction_id, "ledger_id": ledger_id}


def reject_flag(conn: psycopg.Connection, flag_id: int) -> None:
    flag = repo.get_flag(conn, flag_id)
    if flag is None:
        raise ReviewError(404, "no such flag")
    if flag["status"] != "open":
        raise ReviewError(409, f"flag is {flag['status']}, not open")
    repo.resolve_flag(conn, flag_id, "rejected")


def revert_correction(conn: psycopg.Connection, correction_id: int) -> dict:
    c = repo.get_correction(conn, correction_id)
    if c is None:
        raise ReviewError(404, "no such correction")
    if c["reverted_at"] is not None:
        raise ReviewError(409, "already reverted")
    seg = repo.get_segment(conn, c["segment_id"])
    restored = apply_span(seg["text_pa"], c["after_text"], c["before_text"])
    if restored is None:
        raise ReviewError(409, "the corrected text is no longer present in the segment")
    repo.update_segment_text(conn, seg["id"], restored)
    repo.mark_reverted(conn, correction_id)
    if c["flag_id"] is not None:
        repo.reopen_flag(conn, c["flag_id"])
    return {"segment_id": seg["id"], "text_pa": restored}
