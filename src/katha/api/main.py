from pathlib import Path
from typing import Iterator

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from katha import repo
from katha.api.ranges import parse_range
from katha.config import Settings, get_settings
from katha.db import connect

app = FastAPI(title="Katha Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
    repo.requeue_job(conn, clip_id)
    return {"ok": True}


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
