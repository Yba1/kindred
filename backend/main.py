"""Kindred backend -- FALLBACK STUB.

This is a minimal, deterministic FastAPI service standing in for the real
Workstream A backend (owner: P1, Pro account, Gemini + Actian). It exists so
the demo isn't blocked if P1's build isn't ready. No external API calls, no
network dependency -- everything here is keyword/hash heuristics so the
service always returns *something* sensible.

Endpoints (see backend/README.md for the contract):
    POST /profile  -> semantic profile from raw context text
    POST /graph    -> graph payload the frontend renders (A -> B contract)
    GET  /health   -> {"status": "ok"}
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the repo root importable so `loop.personas` resolves regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from loop.personas import generate_personas
    from loop.contracts import EMBED_DIM, STAGES, SEEKING, TOPICS
except Exception:
    # If loop/ isn't importable for some reason, fall back to bare constants
    # so this stub still runs standalone.
    generate_personas = None
    EMBED_DIM = 16
    STAGES = ["student", "ic", "senior", "founder", "investor", "operator"]
    SEEKING = ["cofounder", "hire", "mentor", "investor", "peer", "customer"]
    TOPICS = ["ai-infra", "climate", "fintech", "biotech", "consumer", "devtools", "robotics", "edtech"]

app = FastAPI(title="Kindred backend (fallback stub)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

N_SAMPLE_PEOPLE = 15
SAMPLE_SEED = 42


# ---------------------------------------------------------------------------
# Deterministic helpers (no LLM, no network -- just hashing/keyword parsing)
# ---------------------------------------------------------------------------

def _hash_unit_vector(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic pseudo-embedding derived from a sha256 hash of `text`."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand the digest to `dim` floats in [-1, 1] by cycling through bytes.
    vals = []
    for i in range(dim):
        b = h[i % len(h)]
        # mix in position so repeated bytes don't repeat identical values
        mixed = (b + i * 31) % 256
        vals.append((mixed / 255.0) * 2 - 1)
    norm = math.sqrt(sum(v * v for v in vals))
    if norm == 0.0:
        return vals
    return [v / norm for v in vals]


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def _pick_keyword(text: str, options: list[str], fallback_index: int = 0) -> str:
    """Return the option whose name appears (case-insensitively) in `text`,
    else a deterministic hash-based fallback."""
    low = text.lower()
    for opt in options:
        if opt.lower() in low:
            return opt
    if not text:
        return options[fallback_index]
    idx = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % len(options)
    return options[idx]


def _extract_tags(text: str, limit: int = 6) -> list[str]:
    """Very small heuristic keyword extractor: pull distinct alpha tokens
    longer than 3 chars, longest-first, deduped, deterministic order."""
    import re

    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
    seen: list[str] = []
    for tok in tokens:
        t = tok.lower()
        if t not in seen:
            seen.append(t)
    # bias toward domain vocabulary if present
    vocab_hits = [t for t in TOPICS + SEEKING + STAGES if t.lower() in text.lower()]
    ordered = vocab_hits + [t for t in seen if t not in vocab_hits]
    return ordered[:limit] if ordered else ["general"]


def _build_profile(context: str) -> dict[str, Any]:
    context = context or ""
    stage = _pick_keyword(context, STAGES)
    seeking = _pick_keyword(context, SEEKING)
    topic = _pick_keyword(context, TOPICS)

    end = STAGES.index(stage)
    # deterministic trajectory length from hash, 1..min(end+1, 4)
    max_len = min(end + 1, 4)
    h = int(hashlib.sha256((context + "|traj").encode("utf-8")).hexdigest(), 16)
    length = 1 + (h % max_len) if max_len > 0 else 1
    start = end - length + 1
    trajectory = STAGES[start:end + 1]

    roles = sorted({topic, stage})
    tags = _extract_tags(context)
    embedding = _hash_unit_vector(context or "empty-context")

    return {
        "roles": roles,
        "trajectory": trajectory,
        "seeking": seeking,
        "tags": tags,
        "embedding": embedding,
    }


_SAMPLE_PEOPLE_CACHE: list[dict[str, Any]] | None = None


def _sample_people() -> list[dict[str, Any]]:
    """~15 sample people with embeddings, used to populate the /graph stub."""
    global _SAMPLE_PEOPLE_CACHE
    if _SAMPLE_PEOPLE_CACHE is not None:
        return _SAMPLE_PEOPLE_CACHE

    people: list[dict[str, Any]] = []
    if generate_personas is not None:
        try:
            personas = generate_personas(n=N_SAMPLE_PEOPLE, seed=SAMPLE_SEED)
            for p in personas:
                people.append({
                    "id": p.id,
                    "name": p.name,
                    "topic": p.topic,
                    "focus": p.focus,
                    "stage": p.stage,
                    "trajectory": list(p.trajectory),
                    "seeking": p.seeking,
                    "style": p.style,
                    "expertise": p.expertise,
                    "embedding": list(p.embedding),
                })
        except Exception:
            people = []

    if not people:
        # Hardcoded fallback if loop.personas couldn't be imported/run.
        for i in range(N_SAMPLE_PEOPLE):
            topic = TOPICS[i % len(TOPICS)]
            stage = STAGES[i % len(STAGES)]
            seeking = SEEKING[i % len(SEEKING)]
            name = f"Sample Person {i + 1}"
            people.append({
                "id": f"p{i + 1:04d}",
                "name": name,
                "topic": topic,
                "focus": "product",
                "stage": stage,
                "trajectory": STAGES[:STAGES.index(stage) + 1],
                "seeking": seeking,
                "style": "async",
                "expertise": "mid",
                "embedding": _hash_unit_vector(name),
            })

    _SAMPLE_PEOPLE_CACHE = people
    return people


def _reasons_for(person: dict[str, Any], profile: dict[str, Any], score: float) -> list[str]:
    reasons: list[str] = []
    if person["seeking"] == profile["seeking"]:
        reasons.append(f"both seeking {profile['seeking']}")
    if profile["trajectory"] and person["trajectory"]:
        overlap = set(profile["trajectory"]) & set(person["trajectory"])
        if overlap:
            reasons.append(f"shares trajectory stage(s): {', '.join(sorted(overlap))}")
    if person["topic"] in profile["tags"] or person["topic"] in profile["roles"]:
        reasons.append(f"same domain focus: {person['topic']}")
    if not reasons:
        reasons.append(f"embedding similarity {score:.2f}")
    return reasons[:3]


def _build_graph(context: str, profile: dict[str, Any] | None) -> dict[str, Any]:
    if profile is None:
        profile = _build_profile(context or "")

    people = _sample_people()
    center_embedding = profile["embedding"]

    scored = []
    for person in people:
        score = _cosine(center_embedding, person["embedding"])
        # clamp/normalize into a friendly 0..1-ish range for the demo
        score = max(0.0, min(1.0, (score + 1) / 2))
        scored.append((person, score))

    scored.sort(key=lambda ps: ps[1], reverse=True)

    n = len(scored)
    nodes = [{"id": "user", "name": "You", "score": 1.0, "x": 0.0, "y": 0.0}]
    edges: list[dict[str, Any]] = []
    reasons: dict[str, list[str]] = {}

    radius = 10.0
    for i, (person, score) in enumerate(scored):
        angle = (2 * math.pi * i) / max(n, 1)
        # closer matches sit nearer the center
        r = radius * (1.15 - score)
        x = round(r * math.cos(angle), 3)
        y = round(r * math.sin(angle), 3)
        nodes.append({
            "id": person["id"],
            "name": person["name"],
            "score": round(score, 4),
            "x": x,
            "y": y,
        })
        edges.append({"source": "user", "target": person["id"], "weight": round(score, 4)})
        reasons[person["id"]] = _reasons_for(person, profile, score)

    return {
        "center": "user",
        "nodes": nodes,
        "edges": edges,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ProfileRequest(BaseModel):
    context: str


class GraphRequest(BaseModel):
    context: str | None = None
    profile: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/profile")
def profile(req: ProfileRequest) -> dict[str, Any]:
    return _build_profile(req.context)


@app.post("/graph")
def graph(req: GraphRequest) -> dict[str, Any]:
    profile_payload = req.profile
    if profile_payload is not None and "embedding" not in profile_payload:
        # caller sent a partial/foreign profile shape -- rebuild from context
        profile_payload = None
    return _build_graph(req.context or "", profile_payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
