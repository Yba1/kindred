# Kindred — sponsor map

> Slide version: [`sponsor_map.html`](sponsor_map.html) — open in a browser, print
> to PDF at Landscape with background graphics on for a single 16:9 page.

**One sponsor, one job, one artifact.** Nothing is used twice, nothing is
decorative. Remove any box and the loop stops turning — which is the point:
Tool Use is unambiguous because no two sponsors do the same work.

## The pipeline

| # | Stage | Sponsor | The one job | Artifact |
|---|---|---|---|---|
| 1 | Understand | **Gemini** | Reads raw context into a semantic profile — trajectory and the explicit ask, not keywords. Also writes the per-edge reasoning on the graph. | `POST /profile` → `{domain, prior_domain, stage, seeking, offering}` |
| 2 | Retrieve | **Actian** | Vector store for every profile. Pulls the candidate neighbourhood — wide, by trajectory as well as topic — before anyone ranks it. | `embed(profile)` → top-k neighbours |
| 3 | Judge | **Pioneer / Fastino** | The fine-tuned match scorer. Trained on outcomes (landed vs didn't), not similarity. One yes/no question, thousands of times a generation, in milliseconds. | `score_pair(a, b)` → 0.0–1.0 · LoRA on `gliner2-base-v1` |
| 4 | Act | **BAND** | Opens the intro thread for a match that clears the bar — and is where the outcome becomes observable. | intro thread → `landed = 1 / didn't = 0` |

*Optional:* **Guild** versions the weight vector each generation emits.

## The loop

```
BAND outcomes → Pioneer training set → refit → new weights
     ↑                                              ↓
  intro sent  ←  graph re-clusters  ←  Actian re-ranks
```

Every sponsor sits on exactly one edge of that cycle. That's what lets the
scorer be frozen after certification without touching anything else — and it's
why the graph visibly re-clusters on screen when the weights move.

## Why Pioneer is the judge, not an LLM

The loop asks one yes/no question — *will these two connect?* — thousands of
times per generation. A 205M task-specific encoder answers in milliseconds; a
70B model per candidate pair makes the loop unaffordable. That's the shape
Fastino's TLMs exist for, so the scorer is a LoRA fine-tune of
`fastino/gliner2-base-v1` framed as text classification (`connect` / `pass`).

## The proof

| model | F1 | ROC-AUC |
|---|---|---|
| always-connect floor | 0.571 | 0.500 |
| embedding-cosine baseline | 0.549 | 0.655 |
| **Pioneer match scorer** | **0.619** | **0.755** |

Held-out split (n=50, 40.5% landing rate); both models set their threshold
without touching test. **+0.086 mean F1 over 10 splits (wins 8/10); +0.100 on a
cold-start cohort of 100 pairs among people never seen in training.**

Full numbers, caveats and protocol: [`../pioneer/artifacts/report.md`](../pioneer/artifacts/report.md).
