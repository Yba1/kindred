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
| Graph renders from the backend payload | ✅ edge thickness = weight, node size = tie to you, `center` pinned |
| Click a node → reasoning panel | ✅ `reasons[id]`, profile, path, the ask, driving weight dim |
| Re-cluster animation on new weights | ✅ 2s tween + layout re-settle + auto re-fit |
| `/village` mounted unmodified | ✅ served from `../viz/village.html`, never copied or edited |

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
├── data/loadGraph.js      POST /graph → stub fallback
├── data/validate.js       payload → trusted graph  (tested)
├── weights/rescore.js     w · features → score, driver attribution  (tested)
├── weights/vectors.js     the 3 stand-in generations
├── graph/simulation.js    d3-force + the re-cluster tween
├── graph/render.js        SVG data joins, labels, tooltips, fit-to-view
├── graph/ForceGraph.jsx   React shell — never re-renders on a tick
└── panel/ReasoningPanel.jsx
```

D3 owns the SVG through refs; React owns the chrome and the panel. Nothing in
React re-renders per tick — that's what keeps 437 edges smooth.

```bash
npm test        # 29 tests over the pure scoring + validation modules
npm run build   # dist/ also gets village.html at /village
```

## Re-theming

Every colour, font and radius lives in `src/theme.css`. Graph node fills are the
one exception — they're data colours, so they sit in `DOMAIN_COLORS` in
`src/graph/render.js` where D3 can read them as values.

## Scope guard

One focused surface. No backend logic, no loop math.
