# Frontend — Workstream B (owner: M, 1× Max account)

D3 force-directed semantic match graph + dashboard. Highest-risk single piece — iterate fast.

## Deliverable

- Graph renders from the backend payload (contract in `../backend/README.md`)
- Click a node → reasoning panel (the `reasons[id]` strings)
- Re-cluster animation when a new weight vector arrives (workstream D emits an ordered
  list of `w` vectors; re-layout the graph on each — this is the "it's learning" moment)

## Pages

- `/` — the graph (this workstream)
- `/village` — serve `../viz/village.html` as-is on its own route (owner: Raj; don't edit
  it here, just mount it)

## Scope guard

One focused surface. No backend logic, no loop math. If D3 fights you, land a static
layout first and add force simulation after — a rendered graph at H3 beats a perfect one
at H5.
