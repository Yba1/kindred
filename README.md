# Kindred

An agent that finds the people closest to you *in meaning* — builds a semantic profile
of how you think, surfaces a live graph of your matches with the reasoning for each, and
**learns what "close" really means from which connections land**, re-clustering the graph
on screen as it improves.

Two surfaces:
- **`/` — the graph.** Drop your context → semantic match graph → click a node for the
  reasoning → connect. The graph reorganizes itself live as the matcher learns.
- **`/village` — watch the agents.** A pixel-art village where each agent is a villager;
  they deliberate a match in the town square and reach consensus. Themed, data-driven,
  live-stream ready. See `viz/village.html`.

## Structure

```
kindred/
├── README.md
├── OWNERSHIP.md            # who builds what, split by compute tier
├── backend/               # FastAPI: Profiler, Matcher, Introducer, Evaluator  (owner: P1)
│   └── README.md
├── frontend/              # D3 semantic graph + dashboard                       (owner: M)
├── agents/
│   └── phase_card_agent.md  # the one agent that generates village scenes
└── viz/
    ├── village.html         # themed agent-conversation visualizer  ← runs standalone
    └── phase_cards.example.json
```

## Quickstart (village viz, no backend)

```bash
# it's a single file — just open it
open viz/village.html          # macOS
# or serve it:
python3 -m http.server 8080    # then visit /viz/village.html
```
It plays the demo deliberation immediately. To drive it live, set
`window.KINDRED_STREAM_URL = "/agent-stream"` (an SSE endpoint emitting
`data: {json event}`), or call `window.pushAgentEvent({...})` from your own WebSocket.

## Full build

See `../kindred_build_spec.md` for the architecture, agent I/O contracts, the evolution
loop math, synthetic-data recipe, and the 5-hour hour-by-hour. See `OWNERSHIP.md` for the
per-person split.

## Push to GitHub

```bash
cd kindred-framework
git init && git add -A && git commit -m "Kindred: scaffold + village viz"
git branch -M main
git remote add origin git@github.com:<you>/kindred.git
git push -u origin main
# then branches per workstream:
git checkout -b feat/graph   # M
git checkout -b feat/backend # P1
git checkout -b feat/loop    # Raj
git checkout -b feat/pioneer # P2
git checkout -b feat/viz     # Raj
```

## Sponsors

Actian (profile vectors) · Pioneer/Fastino (match scorer, fine-tuned on landings) ·
DeepMind Gemini (profiling, reasoning) · BAND (intro thread). Guild optional (weight
versioning).
