# Pioneer x Kindred

**Status: fallback stub.** `pioneer/finetune.py` is a standalone, working demo of
"fine-tuned beats baseline." `pioneer_predict()` is the seam a real Pioneer
integration drops into; it is not wired to the Pioneer API yet.

## Pioneer's premise

Pioneer is an inference API that improves with your traffic: instead of a
static embedding model or a hand-tuned scorer, you send it outcome data and it
returns a fine-tuned model that gets sharper as more real interactions come
in.

## How Kindred's Evaluator maps onto it

Kindred's matcher already has the shape Pioneer expects:

- **Six profile dimensions per pair** (`loop/contracts.py: FEATURES`) —
  domain, focus, trajectory, seeking, collaboration style, expertise fit —
  computed by `loop/features.py`.
- **Ground-truth outcomes** — `landed` (did the two people actually connect)
  from `loop/pairs.py`.
- **An evolution loop** (`loop/evolve.py`) that repeatedly refits a 6-dim
  weight vector against those outcomes and only promotes it if it beats the
  held-out land rate so far — exactly the "improves with traffic" loop Pioneer
  automates as a hosted fine-tune.

`pioneer/finetune.py` treats the real, learned weights from `run.json`
(the actual output of that evolution loop, not a hand-picked number) as the
stand-in for "the Pioneer fine-tuned model," and compares it against the
naive alternative most teams ship first: scoring pairs by raw embedding
cosine similarity alone.

## The numbers

Run: `python -m pioneer.finetune`

```
Baseline (embed_cos only): F1=0.44 | Fine-tuned (6-dim learned): F1=0.59
  baseline   -> precision=0.48 recall=0.41
  fine-tuned -> precision=0.65 recall=0.54
```

The fine-tuned model uses all six FEATURES dimensions with weights learned by
`loop.evolve` (persisted in `run.json`); the baseline uses `embed_cos` alone
on the same held-out split, with the classification cutoff on both models
fixed from the training split's positive rate (no peeking at held-out
labels). Fine-tuned beats baseline by ~15 F1 points on identical data.

## What P2 does next

Replace the body of `pioneer_predict(a, b)` in `pioneer/finetune.py` with the
real Pioneer/Fastino fine-tune inference call, flip `PIONEER_ENABLED = True`,
and the rest of this module (data generation, split, F1 harness) keeps
working unchanged as the evaluation scaffold.
