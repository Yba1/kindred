# Frontend — Workstream B (owner: M, 1× Max account)

D3 force-directed semantic match graph + reasoning panel. Vite + React + D3.

```bash
cd frontend
npm install
npm run dev          # → http://localhost:5173
```

Runs with no backend: if `POST /graph` doesn't answer, it renders
`public/sample_graph.json` (61 nodes / 437 edges) and flags `STUB DATA` in the
status bar. The 500 in the dev console is the proxied `/graph` with nothing
behind it — expected until workstream A is up.

## Deliverable status

| | |
|---|---|
| Intake → a graph built around you | ✅ paste your context, `POST /graph {context}`, graph rebuilds with you at the centre |
| Graph renders from the backend payload | ✅ edge thickness = weight, node size = tie to you, `center` pinned |
| Click a node → reasoning panel | ✅ `reasons[id]`, profile, path, the ask, driving weight dim |
| Re-cluster animation on new weights | ✅ 2s tween + layout re-settle + auto re-fit |
| `/village` mounted unmodified | ✅ served from `../viz/village.html`, never copied or edited |

## Intake — "drop your context"

`src/intake/IntakePanel.jsx` is the front door. It opens over the graph on first
load, `esc` / `×` / `SKIP` dismisses it, and **`YOUR CONTEXT` in the top bar
re-opens it** — that's the button to hit when re-running intake live on stage.
The draft survives a dismiss, so a half-typed context is still there.

Submitting calls `loadGraph({ context })` → `POST /graph {"context": "…"}` (the
same field `POST /profile` takes) with a 15s budget, because the profiler runs
before the matcher can answer. The empty first paint keeps the short 2.5s one.

It opens *over* the graph, never instead of it, so there is no blank screen at
any point in the flow — including these:

| what happened | what you see |
|---|---|
| `/graph` answered | `LIVE · YOUR GRAPH`, top bar echoes your context |
| `/graph` unreachable or malformed | the graph still renders from the stub, a `STUB` banner says *"couldn't reach the matcher… not your real matches"*, and the chip reads `STUB DATA · NOT YOUR MATCHES` |
| the stub is gone too | intake stays open with the reason on it; whatever was already rendered stays up |

## Driving the re-cluster (D→B)

The graph re-clusters on any weight vector, from anywhere:

```js
window.applyWeights([0.1, 0.6, 0.25, 0.05], { caption: 'shared trajectory beats shared topic' })
```

Weights are ordered over `['topic', 'trajectory', 'seeking', 'stage']`
(`FEATURE_NAMES` in `src/weights/rescore.js`) and are normalized on the way in,
so an unnormalized vector off the Evaluator is fine. The three hardcoded
generations in `src/weights/vectors.js` are stand-ins — swap them for D's real
`w` list and nothing else changes. `window.KINDRED_GENERATIONS` exposes them.

## Payload contract (A→B)

Consumes the contract in `../backend/README.md` as-is:
`{center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight}], reasons:{id:[str]}}`.

**One additive ask of workstream A:** put a `features` array on each edge — the
per-dim scores the weight vector multiplies. Without it an edge keeps its
payload weight and simply won't move when the weights learn, so the money shot
falls flat. Everything else is optional; these node fields light up the panel
when present: `to`, `from`, `seeking`, `stage`, and `meta.vocab` for labels.

Malformed rows are dropped with a console warning rather than blanking the
surface — see `src/data/validate.js`.

## Layout

```
src/
├── theme.css              design tokens — the ONLY file with colours/fonts in it
├── app.css                layout, all values via var()
├── intake/IntakePanel.jsx the "drop your context" surface
├── data/loadGraph.js      POST /graph {context} → stub fallback  (tested)
├── data/validate.js       payload → trusted graph  (tested)
├── weights/rescore.js     w · features → score, driver attribution  (tested)
├── weights/vectors.js     the 3 stand-in generations
├── graph/simulation.js    d3-force + the re-cluster tween
├── graph/render.js        SVG data joins, labels, tooltips, fit-to-view
├── graph/ForceGraph.jsx   React shell — never re-renders on a tick
└── panel/ReasoningPanel.jsx
```

D3 owns the SVG through refs; React owns the chrome and the panel. Nothing in
React re-renders per tick — that's what keeps 437 edges smooth. The intake draft
is state *inside* `IntakePanel`, not in `App`, for the same reason: typing a
context must not re-render the graph shell.

```bash
npm test        # 35 tests over the pure scoring, validation + loader modules
npm run build   # dist/ also gets village.html at /village
```

## Re-theming

Every colour, font and radius lives in `src/theme.css`. Graph node fills are the
one exception — they're data colours, so they sit in `DOMAIN_COLORS` in
`src/graph/render.js` where D3 can read them as values.

## Scope guard

One focused surface. No backend logic, no loop math.
