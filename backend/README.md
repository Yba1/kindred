# Backend — Workstream A (owner: P1, Pro account)

FastAPI service: Profiler, Matcher, Introducer, Evaluator.

## Deliverable

- `POST /profile` — raw context in, semantic profile out (Gemini for profiling/reasoning)
- `POST /graph` — returns the graph payload the frontend renders
- Actian vector-store client (numpy cosine fallback if Actian is down)

## Graph payload contract (A→B — do not drift)

```json
{
  "center": "user",
  "nodes": [{"id": "p1", "name": "…", "score": 0.87, "x": 0, "y": 0}],
  "edges": [{"source": "user", "target": "p1", "weight": 0.87}],
  "reasons": {"p1": ["same trajectory: finance → agent infra", "both seeking cofounder"]}
}
```

## Agent events (D→viz)

If you emit deliberation events, use the shape the village expects:
`{"speaker": "matcher", "text": "…", "action": "speak|agree|disagree|resolve", "consensus": 0.42, "banner": "OPTIONAL"}`
over SSE at the URL the viz reads from `window.KINDRED_STREAM_URL`.

## Scope guard

Bounded, single-focus task: routes + schema + Actian client. No fan-out, no loop work
(that's workstream D). If you finish early, add request validation and a `/health` route.
