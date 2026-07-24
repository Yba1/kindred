# Integration Runbook — H3 Merge (owner: Raj)

Practical checklist for merging `feat/backend`, `feat/graph`, `feat/pioneer`, `feat/loop`,
`feat/viz` into `main`. Raj owns `main` and does the merges. Integrate at H3, not before.

## 1. Merge order

Merge in dependency order, safest-first:

1. [ ] **`feat/backend`** — nothing depends on anything else here; land it first as the
   base. Routes + schema + Actian client.
2. [ ] **`feat/graph`** — the D3 frontend already stubs the backend payload during solo
   dev, so it merges cleanly on top and just needs to point at real `/graph` responses.
3. [ ] **`feat/pioneer`** — drops in `score_pair()` behind the existing scorer interface;
   low blast radius, no one else depends on it yet.
4. [ ] **`feat/loop`** — merge **last**. Its `run.json` (weight vectors + events) is what
   drives the graph's re-cluster animation and the village's live feed, so everything
   else needs to exist first for the wiring in steps 4–5 below to be testable end-to-end.

**Conflict rule:** if any two branches disagree on a shape (graph payload, weight vector,
event fields), **`loop/contracts.py` wins.** It's the single source of truth — resolve the
conflict by changing the other side to match `contracts.py`, not the reverse.

## 2. Contract checklist

Verify each before calling the merge done:

- [ ] **Graph payload shape** (A→B): `{center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight}], reasons:{id:[str]}}`
  matches `backend/README.md` and what `feat/graph` actually renders.
- [ ] **Weight vector length/order** (D→B): length `6`, ordered per `FEATURES` in
  `loop/contracts.py` — `["domain_sim", "focus_sim", "trajectory_sim", "seeking_match", "collab_sim", "expertise_fit"]`.
  Any consumer indexing into the weight vector must use this order.
- [ ] **Agent event shape for SSE** (D→viz): `{speaker, text, action, consensus, banner}`,
  `action` one of `speak|agree|disagree|resolve`, delivered as `data: {json}` over SSE.
- [ ] **Sponsor tool boundaries**: Actian = vectors (A), Pioneer = scorer (C),
  Gemini = profiling/reasoning (A), BAND = intro thread (Introducer) — no overlap, so
  Tool Use judging stays unambiguous.

## 3. Verify after merge

Run in order from repo root:

- [ ] `python -m loop.run` — regenerates `run.json` from the merged `loop/` modules
  (personas → pairs → features → scorer → evolve → narrate → replay). Confirm it exits
  clean and `run.json` is written.
- [ ] `python scripts/integration_check.py` — cross-checks the merged artifacts against
  the contracts in `loop/contracts.py`: validates `run.json`'s shape (meta/weights/
  generations/events), validates the backend's graph payload against the A→B contract,
  and validates the phase-card agent's output against what the village expects. Treat
  any failure here as a merge blocker, not a follow-up.
- [ ] `python -m unittest tests.test_loop` — unit tests over the loop pipeline (personas/
  pairs/features/scorer/evolve/narrate/replay); confirms the merged loop still produces a
  deterministic, contract-shaped result.

## 4. Wiring the graph to the loop

- [ ] Confirm `frontend` exposes `window.applyWeights(w)` (per `frontend/README.md`,
  triggers the re-cluster animation on a new weight vector).
- [ ] After merge, feed `run.json`'s `weights` (the ordered `weight_vectors` list) into
  `applyWeights` **one at a time**, in order — either:
  - a small driver script that reads `run.json` and calls `applyWeights(w)` on a timer, or
  - the frontend itself polling `run.json` and calling `applyWeights` as new entries appear.
- [ ] Verify the graph visibly re-clusters on each step, not just the last one — that's
  the "it's learning" moment the demo hinges on.

## 5. Wiring the village to real events

- [ ] `viz/village.html` reads `window.KINDRED_STREAM_URL` for a live SSE feed, or falls
  back to `?cards=<file>.json` / `phase_cards.json` for a scripted deck.
- [ ] After merge, point `KINDRED_STREAM_URL` at the real event stream served by
  `viz/sse_server.py` (streaming `run.json`'s `events`, shaped per the agent-event
  contract in section 2) instead of the scripted deck, for the live demo.
- [ ] **Keep the scripted deck (`phase_cards.json`) as the safety net.** If the live SSE
  wiring breaks or lags during the actual demo, fall back to `?cards=phase_cards.json`
  rather than running the demo on a broken live feed.

## 6. Freeze discipline

- [ ] Freeze feature work at **H4** — no new features after freeze, per `OWNERSHIP.md`.
- [ ] At freeze, record the deterministic replay: `python -m loop.run --replay run.json`
  and confirm it reproduces the same events/weights deck used in the live demo.
- [ ] Treat the frozen `run.json` + replay as the fallback if anything live breaks during
  the demo itself.
