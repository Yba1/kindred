# Ownership — work split by compute

Principle: **agent fan-out scales with account tier.** Max accounts get the work that
needs many parallel agents + heavy token burn (bulk generation, iterative loops,
integration across the whole tree). Pro accounts get well-bounded, single-focus tasks
that one agent can finish without fanning out.

| Owner | Accounts | Load | Workstream | Deliverable |
|---|---|---|---|---|
| **Raj** | 3× Max | Heaviest | **D — evolution loop + synthetic data + replay** AND **village viz integration + live agent stream** AND **final integration/glue** | 6-gen loop climbing 41→84%, `run.json` + `--replay`, village wired to live SSE, all streams glued |
| **Teammate M** | 1× Max | Medium | **B — D3 graph frontend** (highest-risk single piece) | Graph renders from payload, node-click → reasoning, re-cluster animation on new weights |
| **Teammate P1** | Pro | Low | **A — backend scaffold + Actian + Profiler/Matcher** | `POST /profile`, `POST /graph` returning nodes+edges+reasons; Actian client (numpy fallback) |
| **Teammate P2** | Pro | Low | **C — Pioneer fine-tune + wiring** AND sponsor slide + demo-script polish | `score_pair()` + F1 vs baseline; one-slide sponsor map; rehearsed 3-min script |
| **Phase-card agent** | 1 spare session | Background | Generate village scene cards (`agents/phase_card_agent.md`) | `viz/phase_cards.json` covering PROFILE / MATCH / LEARN / CONNECT |

## Why this split

- **Raj (3 Max):** the two token-hungriest jobs are (1) generating ~300 personas + ~250
  labeled pairs and re-fitting the loop each generation, and (2) integrating everyone's
  streams under time pressure — both want parallel agents running at once. Plus the
  village viz is the thing he personally owns.
- **M (1 Max):** the D3 force graph is the single hardest frontend piece and the most
  likely to eat an afternoon — it needs a strong solo owner with enough compute to
  iterate fast, but it's one focused surface, not a fan-out job.
- **P1 / P2 (Pro):** backend routes + schema, and a single Pioneer training job + wiring,
  are bounded and low-iteration — one agent finishes each without heavy parallelism.
  P2 also owns the non-code win conditions (sponsor slide, demo script) that don't burn
  compute but decide 40% of the rubric.

## Integration points (don't drift)

- **Graph payload contract** (A→B): `{center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight}], reasons:{id:[str]}}`.
- **Weight vector** (D→B): the Evaluator emits an ordered list of `w` vectors; B re-lays-out
  the graph on each. Same `w` drives edge weights everywhere.
- **Agent events** (D→viz): `{speaker, text, action, consensus, banner}` over SSE at
  `window.KINDRED_STREAM_URL`, or scripted cards as fallback.
- **Sponsors, one job each:** Actian=vectors (A), Pioneer=scorer (C), Gemini=profiling/reasoning (A),
  BAND=intro thread (Introducer). Keep them non-overlapping so Tool Use is unambiguous.

## Merge discipline (5-hour event)

- Branch per workstream: `feat/backend`, `feat/graph`, `feat/loop`, `feat/viz`, `feat/pioneer`.
- Raj owns `main` + merges. Integrate at H3, not before.
- Freeze at H4 to record the deterministic replay. No new features after freeze.
