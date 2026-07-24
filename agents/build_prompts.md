# Build prompts — paste one into Claude Code, per person

Each teammate `git clone`s the repo, checks out their branch, and pastes their prompt.
Contracts are embedded so no one has to hunt. Raj owns `main` and merges at H3.

Shared contract (everyone must honor):
- **Graph payload** (A→B): `{center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight}], reasons:{id:[str]}}`
- **Weight vector** (D→B): Evaluator emits an ordered list of `w` vectors; B re-lays-out the graph on each.
- **Agent event** (D→viz): `{speaker, text, action:"speak|agree|disagree|resolve", consensus:0..1, banner?}` over SSE.
- **Sponsors, one job each:** Actian=vectors, Pioneer=scorer, Gemini=profiling/reasoning, BAND=intro thread.

---

## A — Backend  ·  owner P1 (Pro)  ·  branch `feat/backend`

> You're building the backend for Kindred, a matchmaker that finds the people closest to
> someone *in meaning*. Work only on branch `feat/backend`. Build a FastAPI service with:
>
> 1. `POST /profile` — takes `{context: str}` (free text about a person), returns a semantic
>    profile `{roles, trajectory, seeking, tags, embedding}`. Use Gemini for the reasoning
>    (profiling), and produce an embedding vector.
> 2. `POST /graph` — takes `{profile}` (or a user id), pulls nearest neighbours from the
>    Actian vector store, ranks them, and returns EXACTLY this payload:
>    `{center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight}], reasons:{id:[str]}}`
>    where `reasons[id]` are 1–3 short human strings explaining the match.
> 3. An Actian client with a **numpy cosine-similarity fallback** so the demo runs even if
>    Actian is unreachable. Seed it with ~30 sample people.
> 4. A `GET /health` route.
>
> Scope guard: routes + schema + Actian client ONLY. No learning loop, no fine-tuning
> (that's workstreams C and D). Keep the score field on nodes so the frontend can size edges.
> Deliver a runnable `uvicorn` app + a README curl example for each route. Commit to
> `feat/backend`.

---

## B — Graph frontend  ·  owner M (1× Max)  ·  branch `feat/graph`

> You're building the Kindred frontend: a D3 force-directed "semantic match graph." Work only
> on branch `feat/graph`. Requirements:
>
> 1. Render nodes + edges from the backend payload:
>    `{center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight}], reasons:{id:[str]}}`.
>    Edge thickness/length reflects `weight`; the center node is the user.
> 2. Click a node → a side panel shows `reasons[id]` (why this person is a match).
> 3. **The money shot:** when a new weight vector `w` arrives (an ordered list emitted by the
>    evolution loop), re-run the layout and animate the graph re-clustering — topic clumps
>    visibly dissolving into trajectory-based clusters. Expose `window.applyWeights(w)` so the
>    loop can drive it; for now, drive it from a hardcoded sequence of 2–3 `w` vectors.
> 4. Mount `../viz/village.html` unmodified at route `/village` (don't edit it — Raj owns it).
>
> Scope guard: one focused surface. If D3 force sim fights you, ship a static layout first,
> add forces after — a rendered graph at H3 beats a perfect one at H5. Stub the backend with
> a local `sample_graph.json` so you're never blocked on P1. Commit to `feat/graph`.

---

## C — Pioneer scorer  ·  owner P2 (Pro)  ·  branch `feat/pioneer`

> You're building Kindred's match scorer using Pioneer/Fastino fine-tuning. Work only on
> branch `feat/pioneer`. Requirements:
>
> 1. `score_pair(person_a, person_b) -> float` — predicts probability two people actually
>    connect ("land"). Fine-tune a Pioneer model on labeled pairs (landed=1 / didn't=0).
> 2. Compare against an embedding-cosine baseline and report **F1 on a held-out split** — the
>    fine-tuned scorer must beat the baseline. Print the numbers.
> 3. Expose the scorer so the evolution loop (workstream D) can call it as its judge — a clean
>    `score_pair()` function or a tiny `POST /score` endpoint, your call.
> 4. Non-code (this is 40% of the rubric): a **one-slide sponsor map** (Actian=vectors,
>    Pioneer=scorer, Gemini=profiling, BAND=intro) and a rehearsed **3-minute demo script**.
>
> Scope guard: one training job + wiring. Use the synthetic labeled pairs Raj's loop
> generates (`feat/loop` produces them); until then, mock ~200 labeled pairs so you're not
> blocked. Commit to `feat/pioneer`.

---

## D — Evolution loop + viz + glue  ·  owner Raj (3× Max)  ·  branches `feat/loop`, `feat/viz`, `main`

> You're building the heart of Kindred and integrating everyone's work. You have 3 Max
> accounts — fan out agents freely. Three jobs:
>
> **D1 — evolution loop (`feat/loop`):**
> 1. Generate synthetic data: ~300 personas (role, trajectory, what they seek) + ~250 labeled
>    candidate pairs, calibrated so same-topic pairs land only ~41% (the base-rate to beat).
> 2. Run a generational loop: each generation refits match weights on landing outcomes, calls
>    Pioneer's `score_pair()` on a held-out split, and logs connection-rate. Target the climb
>    41% → 84% over ~6 generations by shifting weight off shared-topic onto shared-trajectory.
> 3. Emit an ordered list of weight vectors `w` (for the graph's `applyWeights`) and write a
>    deterministic `run.json`; support `--replay` to reproduce the run frame-for-frame.
>
> **D2 — viz integration (`feat/viz`):**
> 4. Wire real agent events into `viz/village.html` over SSE at `window.KINDRED_STREAM_URL`,
>    event shape `{speaker,text,action,consensus,banner?}`. Keep scripted phase cards as the
>    fallback/storyboard.
>
> **D3 — glue (`main`, at H3):**
> 5. Merge `feat/backend`, `feat/graph`, `feat/pioneer` into `main`; make the graph re-cluster
>    on the loop's weight vectors and the village stream from real events. Freeze at H4 and
>    record the deterministic replay.
>
> Honesty guard: it's ONE self-improving loop over a scoring function (41%→84%), not eight
> independent evolving agents — pitch it that way. Commit loop work to `feat/loop`, viz to
> `feat/viz`, integration to `main`.

---

## Spare session — phase-card agent (background, already run)

Regenerate village dialogue if the sponsor lineup or arc changes:

> You generate PhaseCard JSON for the Kindred village visualizer. Read
> `viz/phase_cards.example.json` for the exact schema and two worked examples. Produce N new
> cards covering PROFILE / MATCH / LEARN / CONNECT. Each card = one round, 5–9 lines,
> consensus climbing 0→1 (monotonic; disagreement via `action:"disagree"`, never a lower
> number), ending on an `action:"resolve"` line at 1.0, 1–3 banners, 4–6 speakers. speaker ∈
> {profiler,matcher,evaluator,introducer,actian,pioneer,band,gemini}. Output ONLY a JSON
> array/object of cards. Write to `viz/phase_cards.json`.
