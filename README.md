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


Pioneer 
Kindred's Matcher is a fine-tuned scoring model, not a fixed similarity metric. We trained a Pioneer-style classifier on labeled outcome pairs (connect / pass) and benchmarked it against an embedding-cosine baseline: F1 0.619 vs 0.549, ROC-AUC 0.755 vs 0.655, holding on a 10-seed resample sweep (wins 8/10) and a 100-pair cold-start cohort. The core insight the numbers back up: the heaviest signal isn't similarity, it's directional fit — what one person needs against what the other actually offers, which a cosine score structurally can't represent.

DeepMind Gemini
Gemini handles the two steps that need language understanding rather than vector math: turning unstructured intake text into a structured semantic profile, and generating the natural-language reasoning the Introducer surfaces on click. The Profiler and embedding pipeline both call Gemini directly when a key is present; every path also has a deterministic fallback (regex-based profiling, hashed embeddings) so the product never hard-depends on the key being live — the same code runs the demo either way.

Actian
Every profile is stored as two vector views — a domain view and a trajectory view — queried on every match. The retrieval layer is built against Actian's client interface with the connection point isolated (_ActianBackend in backend/app/actian.py), so plugging in a live Actian deployment is a config change, not a rewrite; today it runs on an in-process numpy cosine index seeded with 30 people so the demo has zero external dependencies.

BAND
BAND is the layer where the loop closes: once the Matcher and Introducer agree on a match, BAND is where the intro thread would open, carrying the generated reasoning as the first message. This is currently narrated in the /village visualization — a live agent, "Introducer," acting through BAND to send the opener.

Guild
Every promotion the Evaluator makes is a candidate for weight versioning — Guild's job would be tracking each accepted generation's weight vector as a reproducible, rollback-able version, so a degraded fine-tune could be reverted to a known-good state without touching anything else in the pipeline.


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
