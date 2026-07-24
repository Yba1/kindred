# loop/

## What this is

`loop/` is Kindred's ONE self-improving loop: a match-scoring function that
starts naive and, over a handful of generations, learns from outcomes which
profile signals actually predict a real connection. It goes from a ~41%
held-out connection-rate (a domain/topic-chasing gen-0 matcher) to ~84% by
re-weighting toward shared trajectory and compatible intent. This is **one
loop over one scoring function** — not eight independently-evolving agents.
The village visualization dramatizes the process as agents (profiler,
matcher, evaluator, introducer, etc.) debating and reaching consensus, but
underneath there is a single learning system: one weight vector, one
promotion gate, one held-out metric.

## The six profile dimensions

Defined in `contracts.FEATURES`, in order:

| # | feature | compares | naive gen-0 weight | learned weight |
|---|---|---|---|---|
| 1 | `domain_sim` | same TOPIC | over-weighted (matcher ranks by this first) | demoted — sharing a domain alone is a weak/negative signal |
| 2 | `focus_sim` | same FOCUS | over-weighted | demoted |
| 3 | `trajectory_sim` | overlap of STAGE trajectories | under-weighted | promoted — real signal |
| 4 | `seeking_match` | compatible SEEKING/intent | under-weighted | promoted — real signal |
| 5 | `collab_sim` | same STYLE | under-weighted | promoted (mild) |
| 6 | `expertise_fit` | ~one EXPERTISE step apart | under-weighted | promoted (rewards "one step ahead") |

The hidden ground-truth model in `pairs.py` bakes this in directly: shared
domain carries a small *penalty*, while trajectory + seeking are the
dominant terms. The scoring loop has to discover this from outcomes rather
than being told it.

## Module-by-module

| file | what it does | key signature | verified output |
|---|---|---|---|
| `contracts.py` | Shared dataclasses/constants (`Persona`, `Pair`, `Generation`, `EvolveResult`), the `FEATURES` order, and the `run.json` schema every other module conforms to. | n/a (types + constants) | n/a |
| `personas.py` | Deterministic persona generator; 6 profile dims drawn independently per persona, embeddings correlated with topic. | `generate_personas(n=300, seed=42) -> list[Persona]` | `python -m loop.personas`: generated 300 personas; mean within-topic cosine = 0.5997, mean cross-topic cosine = -0.0062 (sanity check passes). |
| `pairs.py` | Samples persona pairs (biased toward same-domain) and labels each with a ground-truth `landed` outcome from a hidden true model where trajectory+seeking drive landing and shared domain is a mild penalty. | `generate_pairs(personas, n=250, seed=43) -> list[Pair]` | `python -m loop.pairs`: n=250 pairs; same-domain land-rate = 0.404 (n=151, target 0.38-0.44 ✓); trajectory&seeking land-rate = 0.933 (n=15); overall land-rate = 0.436. |
| `features.py` | Pure feature extraction: turns a persona pair into the length-6 vector aligned to `FEATURES`. | `pair_features(a: Persona, b: Persona) -> list[float]` | `python -m loop.features`: sample pair → `[domain_sim=0.0, focus_sim=0.0, trajectory_sim=0.25, seeking_match=0.0, collab_sim=0.0, expertise_fit=1.0]`. |
| `scorer.py` | Logistic scorer: `sigmoid(dot(weights, features) + bias)`. Pure stdlib fallback is the default path; a `pioneer_score_pair` hook exists but is off (`USE_PIONEER=False`). | `score_pair(a, b, weights, bias=0.0) -> float` | `python -m loop.scorer`: pair p0001 x p0002, demo-truth weights `[0.2,0.2,1.3,1.6,1.1,0.9]`, bias -1.0 → score ≈ 0.556. |
| `evolve.py` | Runs the generational weight-learning loop and applies the promotion gate. | `evolve(personas, pairs, generations, seed) -> EvolveResult` | **(pending)** — file does not exist yet in `loop/`. |
| `narrate.py` | Turns an `EvolveResult` into a script of village-agent events (speaker/text/action/consensus) for the viz. Verified rules: speakers restricted to a fixed set, text ≤ 90 chars, consensus non-decreasing, ends on `resolve` at consensus 1.0. | `narrate(result: EvolveResult) -> list[dict]` | `python -m loop.narrate`: runs today only via its **mock fallback** (real `evolve` module not importable yet) — 13 events, first consensus 0.08, last consensus 1.0, all assertions pass. Will use the real `EvolveResult` once `evolve.py` lands. |
| `replay.py` | Serializes an `EvolveResult` + events to `run.json` and loads it back for deterministic replay. | `write_run(result, events, path) -> RunFile`, `load_run(path) -> RunFile` | **(pending)** — file does not exist yet in `loop/`. |
| `run.py` | CLI entrypoint wiring the whole pipeline together; degrades gracefully with a "not ready yet" message for any module (`evolve`, `narrate`, `replay`) that isn't importable yet. | `main(argv=None) -> int` | File exists and is complete, but end-to-end run currently fails at the `loop.evolve` import (module pending); `python -m loop.narrate`'s own self-check confirms this is the only missing link today. |

## Running it end to end

```
python -m loop.run
```

Generates personas, pairs, evolves weights across generations, narrates the
village script, and writes `run.json` (requires `evolve.py` and `replay.py`
to exist — currently pending).

```
python -m loop.run --replay run.json
```

Deterministic replay: loads a previously-written `run.json` and re-plays its
events with a configurable `--interval` between them, with no regeneration
and no randomness — the same file always plays back identically.

## The promotion gate

New weights are only accepted into the next generation if they improve the
held-out connection-rate over the current best; that one-way ratchet is why
the recorded rate (41% → 84%) never dips across generations.
