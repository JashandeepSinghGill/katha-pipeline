from pathlib import Path
from typing import Iterator, Literal

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from katha import appsettings, repo, review
from katha.api.ranges import parse_range
from katha.config import Settings, get_settings
from katha.db import connect
from katha.embed import HashingEmbedder
from katha.text import romanize

app = FastAPI(title="Katha Pipeline")
# CORS is a browser-only control: curl and other non-browser HTTP clients ignore
# it entirely. This API has no authentication (out of scope for this slice) and
# must not be exposed beyond localhost until it does.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().cors_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK = 64 * 1024


def get_app_settings() -> Settings:
    return get_settings()


def get_conn(settings: Settings = Depends(get_app_settings)) -> Iterator[psycopg.Connection]:
    conn = connect(settings)
    try:
        yield conn
    finally:
        conn.close()


class ReciterPatch(BaseModel):
    reciter_speaker_id: str | None


class SegmentPatch(BaseModel):
    text_pa: str


ENTITY_TYPES = Literal["person", "place", "date", "book", "title", "term", "other"]
# Must be the same embedder the worker uses (worker.Pipeline defaults to it too):
# a ledger row learned here is only auto-applicable if its embedding_model
# matches what the worker embeds with.
_embedder = HashingEmbedder()


class RememberSpec(BaseModel):
    entity_type: ENTITY_TYPES
    english_gloss: str | None = None


class ResolveBody(BaseModel):
    action: Literal["accept", "reject"]
    text: str | None = None
    remember: RememberSpec | None = None


class LedgerCreate(BaseModel):
    canonical_gurmukhi: str
    surface_forms: list[str] = []
    entity_type: ENTITY_TYPES
    english_gloss: str | None = None
    confidence_threshold: float | None = None


class LedgerPatch(BaseModel):
    canonical_gurmukhi: str | None = None
    surface_forms: list[str] | None = None
    entity_type: ENTITY_TYPES | None = None
    english_gloss: str | None = None
    confidence_threshold: float | None = None
    retired: bool | None = None


class SettingsBody(BaseModel):
    values: dict[str, str]


@app.get("/api/clips")
def list_clips(conn=Depends(get_conn)):
    return {"clips": repo.list_clips(conn)}


@app.get("/api/clips/{clip_id}")
def get_clip(clip_id: int, conn=Depends(get_conn)):
    clip = repo.get_clip(conn, clip_id)
    if clip is None:
        raise HTTPException(404, "no such clip")
    return {"clip": clip, "segments": repo.list_segments(conn, clip_id)}


@app.patch("/api/clips/{clip_id}")
def patch_clip(clip_id: int, patch: ReciterPatch, conn=Depends(get_conn)):
    if repo.get_clip(conn, clip_id) is None:
        raise HTTPException(404, "no such clip")
    repo.set_reciter(conn, clip_id, patch.reciter_speaker_id)
    return {"ok": True}


@app.patch("/api/segments/{segment_id}")
def patch_segment(segment_id: int, patch: SegmentPatch, conn=Depends(get_conn)):
    if not repo.update_segment_text(conn, segment_id, patch.text_pa):
        raise HTTPException(404, "no such segment")
    return {"ok": True}


@app.post("/api/clips/{clip_id}/retry")
def retry_clip(clip_id: int, conn=Depends(get_conn)):
    if repo.get_clip(conn, clip_id) is None:
        raise HTTPException(404, "no such clip")
    requeued = repo.requeue_job(conn, clip_id)
    return {"ok": True, "requeued": requeued}


@app.get("/api/audio/{clip_id}")
def get_audio(clip_id: int, range: str | None = Header(default=None), conn=Depends(get_conn)):
    # Serve the NATIVE file, not the denoised one: the admin is checking what was
    # actually said, and the denoiser is a machine-listening aid, not ground truth.
    row = repo.get_audio_file(conn, clip_id, "native")
    if row is None:
        raise HTTPException(404, "clip has no prepared audio yet")

    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(404, f"audio missing from disk: {path}")

    size = path.stat().st_size
    try:
        span = parse_range(range, size)
    except ValueError:
        return Response(
            status_code=416, headers={"Content-Range": f"bytes */{size}"}
        )

    headers = {"Accept-Ranges": "bytes"}
    if span is None:
        start, end, status = 0, size - 1, 200
    else:
        start, end = span
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)

    def stream() -> Iterator[bytes]:
        remaining = end - start + 1
        with path.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        stream(), status_code=status, headers=headers, media_type="audio/wav"
    )


# --- correction review ------------------------------------------------------


def _review(fn, *args):
    try:
        return fn(*args)
    except review.ReviewError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@app.post("/api/flags/{flag_id}/resolve")
def resolve_flag(flag_id: int, body: ResolveBody, conn=Depends(get_conn)):
    if body.action == "reject":
        _review(review.reject_flag, conn, flag_id)
        return {"ok": True, "status": "rejected"}
    if body.text is None or not body.text.strip():
        raise HTTPException(400, "accept requires non-empty text")
    remember = body.remember.model_dump() if body.remember else None
    out = _review(review.accept_flag, conn, _embedder, flag_id, body.text, remember)
    return {"ok": True, "status": "accepted", **out}


@app.post("/api/corrections/{correction_id}/revert")
def revert_correction(correction_id: int, conn=Depends(get_conn)):
    return {"ok": True, **_review(review.revert_correction, conn, correction_id)}


@app.get("/api/ledger")
def list_ledger(q: str | None = None, include_retired: bool = False, conn=Depends(get_conn)):
    return {"entries": repo.list_ledger(conn, q, include_retired)}


@app.post("/api/ledger")
def create_ledger(body: LedgerCreate, conn=Depends(get_conn)):
    canonical = body.canonical_gurmukhi.strip()
    if not canonical:
        raise HTTPException(400, "canonical_gurmukhi must not be empty")
    threshold = body.confidence_threshold
    if threshold is None:
        threshold = repo.get_setting_float(conn, "ledger_context_threshold", 0.8)
    elif not 0.0 <= threshold <= 1.0:
        raise HTTPException(400, "confidence_threshold must be within 0..1")
    ledger_id = repo.upsert_ledger(
        conn, canonical=canonical, romanized=romanize(canonical), surface_form=None,
        entity_type=body.entity_type, english_gloss=body.english_gloss, embedding=None,
        embedding_model=None, threshold=threshold,
    )
    forms = [f.strip() for f in body.surface_forms if f.strip() and f.strip() != canonical]
    if forms:
        existing = repo.get_ledger(conn, ledger_id)["surface_forms"]
        merged = list(dict.fromkeys([*existing, *forms]))
        repo.update_ledger(conn, ledger_id, surface_forms=merged)
    return {"ok": True, "id": ledger_id}


@app.patch("/api/ledger/{ledger_id}")
def patch_ledger(ledger_id: int, body: LedgerPatch, conn=Depends(get_conn)):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if "canonical_gurmukhi" in fields:
        fields["canonical_gurmukhi"] = fields["canonical_gurmukhi"].strip()
        if not fields["canonical_gurmukhi"]:
            raise HTTPException(400, "canonical_gurmukhi must not be empty")
        fields["romanized"] = romanize(fields["canonical_gurmukhi"])
    if "surface_forms" in fields:
        fields["surface_forms"] = [f.strip() for f in fields["surface_forms"] if f.strip()]
    if "confidence_threshold" in fields and not 0.0 <= fields["confidence_threshold"] <= 1.0:
        raise HTTPException(400, "confidence_threshold must be within 0..1")
    if not fields:
        raise HTTPException(400, "nothing to update")
    if repo.get_ledger(conn, ledger_id) is None:
        raise HTTPException(404, "no such ledger entry")
    repo.update_ledger(conn, ledger_id, **fields)
    return {"ok": True}


@app.get("/api/settings")
def get_settings_endpoint(conn=Depends(get_conn)):
    specs = [
        {"key": s.key, "kind": s.kind, "default": s.default, "minimum": s.minimum,
         "maximum": s.maximum, "help": s.help}
        for s in appsettings.SPECS.values()
    ]
    return {"values": repo.get_all_settings(conn), "specs": specs}


@app.put("/api/settings")
def put_settings(body: SettingsBody, conn=Depends(get_conn)):
    canonical: dict[str, str] = {}
    for key, value in body.values.items():  # validate everything before writing anything
        try:
            canonical[key] = appsettings.validate(key, value)
        except KeyError:
            raise HTTPException(400, f"unknown setting: {key}") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
    for key, value in canonical.items():
        repo.set_setting(conn, key, value)
    return {"ok": True, "values": repo.get_all_settings(conn)}
