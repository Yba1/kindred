"""Kindred backend — FastAPI app (Workstream A: routes + schema + Actian client).

Routes:
  GET  /health        capability + store status
  POST /profile       {context} -> {roles, trajectory, seeking, tags, embedding}
  POST /graph         {context|profile} -> {center, nodes, edges, reasons}
  GET  /people        seeded + registered people (debug/frontend convenience)
  GET  /agent-stream  SSE deliberation feed for the village viz (bonus)

Runnable: `uvicorn app.main:app --reload` (see backend/README.md for curl examples).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import __version__
from .config import settings
from .embeddings import backend_mode, embed
from .graph import build_graph
from .matcher import rank
from .profiler import build_profile, profiler_mode
from .schemas import GraphRequest, GraphResponse, Profile, ProfileOut, ProfileRequest
from .store import store

try:
    from band.client import band_enabled, send_intro
except ImportError:
    # band/ lives at the repo root, one level above backend/ — not on
    # sys.path when uvicorn is launched with cwd=backend/. Add it once and
    # retry before giving up (rather than silently degrading /introduce).
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    try:
        from band.client import band_enabled, send_intro
    except ImportError:
        band_enabled = None
        send_intro = None

app = FastAPI(title="Kindred backend", version=__version__)

# Frontend + viz are separate origins; wide-open CORS is fine for the event.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_CARDS = Path(__file__).resolve().parents[2] / "viz" / "phase_cards.backend.json"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "profiler": profiler_mode(),          # "gemini" | "heuristic"
        "embeddings": backend_mode(),         # "gemini" | "hash-fallback"
        "actian": store._vecs.mode,           # "actian" | "numpy"
        "gemini": settings.gemini_enabled,
        "people": store.count(),
    }


@app.post("/profile", response_model=ProfileOut)
def profile(req: ProfileRequest) -> ProfileOut:
    """Intake context in, structured profile out — embedded into the store on write."""
    p = build_profile(req.context, name=req.name, id=req.id)
    embedding = store.register(p)             # embed-on-write; returns [domain||traj]
    return ProfileOut(
        id=p.id, name=p.name, roles=p.roles, trajectory=p.trajectory,
        seeking=p.seeking, tags=p.tags, embedding=embedding,
    )


@app.post("/graph", response_model=GraphResponse)
def graph(req: GraphRequest) -> GraphResponse:
    """Pull nearest neighbours, rank them, return the graph the frontend renders."""
    user: Profile = req.profile or build_profile(req.context or "", name=req.name)

    neighbours = store._vecs.query(
        domain_vec=embed(user.domain_text()),
        traj_vec=embed(user.trajectory_text()),
        top_k=30,                             # wide pull; Matcher trims after ranking
        exclude_id=user.id,
    )
    if not neighbours:
        raise HTTPException(status_code=503, detail="vector store is empty")

    matches = rank(user, neighbours, limit=req.top_k)
    return build_graph(user, matches)


@app.get("/people")
def people() -> dict:
    return {"count": store.count(), "people": [p.model_dump() for p in store.all()]}


class IntroduceRequest(BaseModel):
    user_id: str = "user"
    match_id: str
    reasoning: list[str] = []


@app.post("/introduce")
def introduce(req: IntroduceRequest) -> dict:
    """Open the intro thread BAND would send. SIMULATED — see band/README.md.
    Real reasoning in, a real opener out; no real network call to BAND."""
    if send_intro is None:
        raise HTTPException(status_code=503, detail="band/ package not importable")
    thread = send_intro(req.user_id, req.match_id, req.reasoning)
    return {
        "thread_id": thread.thread_id,
        "match_id": thread.match_id,
        "opener": thread.opener,
        "simulated": thread.simulated,
    }


# --------------------------------------------------------------------------- #
#  Bonus: drive the village viz live from the backend deliberation cards.
#  Point the viz at this: `window.KINDRED_STREAM_URL = ".../agent-stream"`.
# --------------------------------------------------------------------------- #
def _load_events() -> list[dict]:
    try:
        data = json.loads(_CARDS.read_text())
    except Exception:
        return []
    events: list[dict] = []
    for card in data.get("cards", []):
        if card.get("objective"):
            events.append({"speaker": "profiler", "text": card["objective"],
                           "action": "speak", "consensus": 0.05,
                           "banner": card.get("round", "")})
        events.extend(card.get("script", []))
    return events


@app.get("/agent-stream")
async def agent_stream(delay: float = 1.4) -> StreamingResponse:
    events = _load_events()

    async def gen():
        for ev in events:
            yield f"data: {json.dumps(ev)}\n\n"
            await asyncio.sleep(delay)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
