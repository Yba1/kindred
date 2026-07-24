# Backend — Workstream A (owner: P1)

FastAPI service: **Profiler + Matcher + Actian client**. Turns intake text into a
semantic profile and returns the match graph the frontend renders. No learning
loop here (that's workstream D) — just routes + schema + vector store.

Every external dependency is optional. With no keys the service still boots and
serves a full demo: the **Profiler** falls back to a heuristic, **embeddings** fall
back to a deterministic hashing model, and the **Actian client** falls back to an
in-process numpy cosine index seeded with ~30 people.

## Run

```bash
cd backend
pip install -r requirements.txt          # fastapi, uvicorn, numpy, pydantic
uvicorn app.main:app --reload            # -> http://127.0.0.1:8000
# or: ./run.sh   (creates a venv on first run)
```

To wire the sponsors, copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`
(profiling + reasoning + embeddings) and/or the `ACTIAN_*` vars (vector store).
`GET /health` reports which path each subsystem is on.

## Routes

| Method | Path            | In                          | Out |
|--------|-----------------|-----------------------------|-----|
| GET    | `/health`       | –                           | capability + store status |
| POST   | `/profile`      | `{context, name?}`          | `{id, name, roles, trajectory, seeking, tags, embedding}` |
| POST   | `/graph`        | `{context \| profile, top_k?}` | the A→B graph payload |
| GET    | `/people`       | –                           | seeded + registered people |
| GET    | `/agent-stream` | –                           | SSE deliberation feed for the village viz (bonus) |

## curl

```bash
# health — shows profiler/embeddings/actian mode + people count
curl -s localhost:8000/health | jq

# profile — intake text in, structured profile out (embedded on write)
curl -s -X POST localhost:8000/profile -H 'content-type: application/json' -d '{
  "context": "Ex-quant from a derivatives desk, now building an evaluation harness for tool-using agents. Looking for a technical cofounder.",
  "name": "Karthik"
}' | jq
# -> { "roles": ["ex-quant","agent infra","founder"],
#      "trajectory": "quant -> agent infra", "seeking": "technical cofounder",
#      "tags": [...], "embedding": [512 floats] }

# graph — nearest neighbours, ranked, with reasoning per edge
curl -s -X POST localhost:8000/graph -H 'content-type: application/json' -d '{
  "context": "Investment banker moving into fintech, building an underwriting copilot for lenders. Looking for a technical cofounder.",
  "top_k": 6
}' | jq
```

## Graph payload contract (A→B — do not drift)

`/graph` returns EXACTLY this shape — the frontend renders it verbatim:

```json
{
  "center": "user",
  "nodes":  [{"id": "p_lena", "name": "Lena Fischer", "score": 0.71, "x": 128.4, "y": 0.0}],
  "edges":  [{"source": "user", "target": "p_lena", "weight": 0.71}],
  "reasons": {"p_lena": ["same trajectory: investment banking → fintech", "both seeking cofounder"]}
}
```

`reasons[id]` is 1–3 short strings explaining the match. `x,y` seed a radial
layout (higher score → smaller radius); the frontend re-lays-out with its force sim.

## How matching works

- **Two vector views per person** (Actian: "one client, two vector views") — a
  DOMAIN view (topic) and a TRAJECTORY view (arc + ask). `/graph` pulls a wide
  top-30 on the blend, then the Matcher trims to `top_k`.
- **Score** blends three axes with a fixed weighting:
  `0.34·domain + 0.36·trajectory + 0.30·seeking_fit`. Trajectory is weighted a
  touch above domain on purpose — *closest in meaning*, not just closest in topic.
  `seeking_fit` is mutual want↔offer complementarity.
- **Embed on write** — every `/profile` call lands the person in the vector store,
  so profiles you create become matchable for future queries.

## Agent events (D→viz)

`GET /agent-stream` replays the backend deliberation cards as SSE, in the shape the
village expects:
`{"speaker": "...", "text": "...", "action": "speak|agree|disagree|resolve", "consensus": 0.0..1.0, "banner": "OPTIONAL"}`.
Point the viz at it with `window.KINDRED_STREAM_URL = "http://localhost:8000/agent-stream"`.

## Tests

```bash
cd backend && python -m pytest tests -q      # contract + validation, offline
```

## Scope guard

Bounded, single-focus: routes + schema + Actian client. No fan-out, no loop work
(workstream D). `/health` and request validation are included.
