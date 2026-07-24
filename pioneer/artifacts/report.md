# Kindred match scorer — held-out results

Generated 2026-07-24T21:06:41+00:00 · seed 0 · backend `local-logistic` · L2 100.0 · 5-fold CV

200 labeled pairs (40.5% landed), split 150 train / 50 test. mock (mockdata.build_datasets) — swap for the loop's real landings.

The scorer's L2 and threshold come from out-of-fold predictions inside train; the baseline's threshold is fitted on all of train. Test is scored once.

| model | F1 | precision | recall | accuracy | ROC-AUC |
|---|---|---|---|---|---|
| always-connect floor | 0.571 | 0.400 | 1.000 | 0.400 | 0.500 |
| cosine baseline | 0.549 | 0.452 | 0.700 | 0.540 | 0.655 |
| **kindred scorer** | 0.619 | 0.591 | 0.650 | 0.680 | 0.755 |

**F1 delta vs cosine baseline: +0.070** (bootstrap 95% CI -0.060 .. +0.195).

For scale: a model that knew each pair's true landing probability would score F1 0.699 / AUC 0.798. Labels are Bernoulli draws, so that — not 1.0 — is the ceiling.

## Does it hold up

- **10-seed split sweep** — scorer 0.623 ± 0.081 vs baseline 0.536 ± 0.049; mean delta +0.086; scorer wins 8/10 splits.
- **Cold-start cohort** (n=100, people never seen in training) — scorer F1 0.792 vs baseline F1 0.693 (+0.100).

## What the scorer learned

| feature | standardised weight |
|---|---|
| `directional_fit` | +0.190 |
| `mutual_fit` | +0.171 |
| `cos_bio` | +0.118 |
| `same_arc` | +0.118 |
| `same_city` | +0.112 |
| `topic_jaccard` | +0.089 |
| `reciprocal_fit` | +0.065 |
| `stage_gap` | -0.043 |
| `seniority_gap` | -0.035 |
| `same_domain` | +0.006 |

## Pioneer fine-tune

Not run for these numbers — no `PIONEER_API_KEY` in this environment, so the table
above comes from the local logistic head. The Pioneer path is wired and runnable
(`--backend pioneer`): it uploads the same pairs as a classification dataset, LoRA-
fine-tunes `fastino/gliner2-base-v1`, and scores the same held-out split through
`POST /inference`. `data/pioneer_train.preview.jsonl` is the exact upload format.

## Caveats

- 50 test rows is small. Read the CI, not just the point estimate.
- Labels are mock. The generator encodes the product thesis — ask/offer fit and shared
  trajectory dominate, shared topic is a weak positive — calibrated so the base landing
  rate is ~40%, matching the 41% the rest of the repo quotes, with the best decile of
  pairs at ~96% and the worst at ~8%. So this measures whether the scorer *recovers that
  structure* from 200 noisy labels, not whether the thesis is true of real users. Re-run
  against the loop's real landings to answer that.
- The scorer reads structured profile fields; the baseline reads profile text only. That
  gap is the finding, not a handicap: `cos_bio` is feature 0, so the scorer strictly
  contains the baseline's signal and adds direction (who asks vs who offers) and gaps
  (stage, seniority) that a similarity score cannot represent.
