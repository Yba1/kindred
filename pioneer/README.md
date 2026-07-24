# Pioneer match scorer — Workstream C (owner: P2)

`score_pair(person_a, person_b) -> float`: the probability two people actually
connect. Fine-tuned on labeled outcomes (landed = 1 / didn't = 0), not on how
similar their profiles look — and it beats an embedding-cosine baseline on a
held-out split.

This is the **judge the evolution loop optimises against** (workstream D).

## Results

Held-out test split, 50 pairs, 40.5% landing rate. Both models pick their
threshold without touching test; test is scored once.

| model | F1 | precision | recall | accuracy | ROC-AUC |
|---|---|---|---|---|---|
| always-connect floor | 0.571 | 0.400 | 1.000 | 0.400 | 0.500 |
| cosine baseline | 0.549 | 0.452 | 0.700 | 0.540 | 0.655 |
| **kindred scorer** | **0.619** | 0.591 | 0.650 | 0.680 | **0.755** |

**F1 +0.070 over the baseline; ROC-AUC +0.100.** For scale: a model that knew
each pair's true landing probability would score F1 0.699 / AUC 0.798. Labels
are Bernoulli draws, so that — not 1.0 — is the ceiling.

Because 50 test rows is small, the claim doesn't rest on one split:

| check | scorer | baseline | delta |
|---|---|---|---|
| held-out split (n=50) | 0.619 | 0.549 | **+0.070** |
| bootstrap 95% CI on that delta | — | — | −0.060 … +0.195 |
| 10-seed split sweep | 0.623 ± 0.081 | 0.536 ± 0.049 | **+0.086**, wins 8/10 |
| cold-start cohort (n=100, unseen people) | 0.792 | 0.693 | **+0.100** |

Read honestly: on a single 50-row split the bootstrap interval crosses zero — at
that sample size it would take a much larger gap not to. The 10-seed sweep and
the 100-pair cold-start cohort are what carry the result, and both agree.

Regenerate everything with `python -m kindred_pioneer.train`; full write-up in
[`artifacts/report.md`](artifacts/report.md).

## Why it wins

The baseline asks *"are these two people similar?"*. The scorer asks *"does what
one needs match what the other has?"* — and cosine similarity structurally cannot
represent that:

- **Direction.** In a bag of words, "looking for seed capital" and "can offer
  seed capital" are the same tokens. Two founders both hunting capital look
  maximally similar and are a dead intro. `directional_fit` is the scorer's
  heaviest feature.
- **Gaps.** Stage and seniority distance are *penalties*. A similarity score is
  monotone — it has nowhere to put a negative.
- **Trajectory over topic.** Same prior domain ("both ex-finance") predicts a
  landing better than same current domain.

`cos_bio` is feature 0, so the scorer contains the baseline's entire signal and
adds to it. This is "more of the profile beats semantic similarity alone", not
two models reading different data.

## Use it

```python
from kindred_pioneer import score_pair, score_pairs, explain

score_pair(alice, bob)                    # 0.0 .. 1.0
score_pairs([(a, b), (c, d)])             # batch — one pass for a whole generation
explain(alice, bob)                       # score + the features that drove it
```

`Person` objects or plain dicts both work, so the loop can pass whatever the
Profiler produced. Required keys: `id`, `name`, `domain`, `prior_domain`,
`stage` (`exploring`/`building`/`scaling`), `seniority`, `city`; optional
`interests`, `seeking`, `offering`, `bio`. Unknown keys are ignored, so other
workstreams can attach their own fields.

Or over HTTP, if importing isn't convenient:

```bash
python -m kindred_pioneer.server --port 8099

curl -X POST localhost:8099/score \
  -H 'Content-Type: application/json' \
  -d '{"a": {...}, "b": {...}}'
# {"score": 0.81, "threshold": 0.403}
```

`POST /score` · `POST /score/batch` · `POST /explain` · `GET /health`.
Stdlib `http.server` — the loop shouldn't need a second web framework to ask one
question. In-process callers should just import and skip the network.

**The scorer is frozen once trained.** It loads `artifacts/model.json` and never
refits. If the judge moved every generation the loop would be chasing a ghost.

## Run it

```bash
cd pioneer
python -m kindred_pioneer.train              # train + evaluate + write artifacts
python -m unittest discover -s tests         # 64 tests
```

Needs Python 3.10+ and numpy; nothing else. Fully deterministic and offline —
same numbers on every machine, which is what a demo needs.

Flags: `--regenerate` rebuilds the mock pairs, `--seed N` changes the split,
`--sweep-seeds N` sets the sweep width. The command exits non-zero if the scorer
fails to beat the baseline, so it works as a regression gate.

## The Pioneer fine-tune

The published numbers above come from the local logistic head, because this
environment has no `PIONEER_API_KEY`. **No Pioneer job has been run** — the
report says so rather than implying otherwise.

The Pioneer path is wired and runnable:

```bash
export PIONEER_API_KEY=pio_sk_...
python -m kindred_pioneer.train --backend pioneer
```

which uploads the same pairs as a **classification** dataset
(`{"text": "<serialised pair>", "label": "connect"|"pass"}`), LoRA-fine-tunes
`fastino/gliner2-base-v1`, polls to completion, runs a Pioneer evaluation, and
scores the same held-out split through `POST /inference` — so its row lands in
the same table, measured the same way. See
[`data/pioneer_train.preview.jsonl`](data/pioneer_train.preview.jsonl) for the
exact upload format.

Why classification on a GLiNER2 encoder rather than an LLM: this is one yes/no
question asked thousands of times per generation. A 205M task-specific encoder
answering in milliseconds is the shape Fastino's TLMs exist for, and the loop
cannot afford a 70B model per candidate pair.

To serve from the fine-tune instead of the local head:

```bash
export KINDRED_SCORER_BACKEND=pioneer
export KINDRED_PIONEER_MODEL_ID=<training job id>
```

`score_pair()` and the HTTP routes are unchanged — only the backend moves.

API surface per <https://agent.pioneer.ai/llms-full.txt> (base
`https://api.pioneer.ai`, `X-API-Key` header).

## The data is mock — read this before quoting the numbers

`data/pairs.jsonl` is 200 generated pairs, standing in until the loop produces
real landings. The generator encodes the product thesis as coefficients
(`mockdata.py`, all of them visible): ask/offer fit and shared trajectory
dominate, shared topic is a weak positive. It's calibrated so the base landing
rate is ~40% — the 41% the rest of the repo quotes — with the best decile of
pairs landing ~96% and the worst ~8%.

So the result above measures whether the scorer **recovers that structure from
200 noisy labels**, not whether the thesis is true of real people. Swapping in
real data is one function: point `mockdata.read_jsonl` at the loop's output.
Everything downstream — features, training, evaluation, the report — is unchanged.

## Layout

```
pioneer/
├── kindred_pioneer/
│   ├── __init__.py         # score_pair, score_pairs, explain — the public surface
│   ├── schema.py           # Person / LabeledPair
│   ├── mockdata.py         # the 200 mock pairs + the generative thesis
│   ├── embeddings.py       # hashed profile embeddings (swap point for Actian/Gemini)
│   ├── features.py         # pair featurisation + Pioneer's text serialisation
│   ├── baseline.py         # cosine baseline + always-connect floor
│   ├── model.py            # logistic head, IRLS, JSON-serialisable
│   ├── metrics.py          # F1 / AUC / thresholds / bootstrap CI
│   ├── pioneer_client.py   # Pioneer API: dataset → fine-tune → evaluate → infer
│   ├── train.py            # the training job + the comparison
│   ├── scorer.py           # score_pair() and friends
│   └── server.py           # POST /score
├── data/                   # pairs.jsonl, cold-start cohort, Pioneer upload preview
├── artifacts/              # model.json (loaded at inference), metrics.json, report.md
└── tests/
```

Artifacts are committed on purpose: the demo must not depend on retraining, and
the exact numbers on the slide should be diffable.

## Integration notes

- `explain()` returns per-feature contributions, which is the same shape the
  graph frontend needs for per-edge reasoning (`reasons: {id: [str]}`).
- Swapping the embedding backend (Actian vectors, Gemini embeddings) means
  implementing one function in `embeddings.py`; the eval harness re-runs
  unchanged and will re-report the baseline honestly against the new space.
- The scorer never sees `p_true` — it's written to `pairs.jsonl` for diagnostics
  and is excluded from every feature path.
