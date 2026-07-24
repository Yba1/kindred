"""Pioneer/Fastino fine-tune stub for the Kindred Evaluator.

This module is a FALLBACK. It is intentionally standalone (it does not modify
`loop/scorer.py` or anything else in `loop/`) so it can be dropped in for the
demo and later swapped out by the real Pioneer integration without touching
the rest of the pipeline.

Two things live here:

1. `PIONEER_ENABLED` / `pioneer_predict(a, b)` — the real seam. Off by default.
   P2 replaces the body of `pioneer_predict` with an actual call to the
   Pioneer/Fastino fine-tuned inference API and flips `PIONEER_ENABLED`.

2. A working local stand-in for "the fine-tuned scorer beats naive baseline":
   the fine-tuned model is the REAL learned 6-dim weight vector produced by
   `loop.evolve` (persisted in `run.json`'s final generation) -- i.e. the
   actual output of the evolution loop, not a toy. The naive baseline is a
   single-feature classifier on embedding cosine similarity alone
   (`embed_cos`), which is the "obvious" approach Pioneer's fine-tuning is
   meant to beat. `evaluate()` measures F1 for both on a held-out split of
   freshly generated personas/pairs and returns the numbers.

Pure standard library. No numpy, no sklearn.
"""
from __future__ import annotations

import json
import os
import random

from loop.contracts import FEATURES, Persona
from loop.features import pair_features
from loop.pairs import generate_pairs
from loop.personas import generate_personas
from loop.scorer import score_features

# ---- Pioneer/Fastino seam --------------------------------------------------
# OFF by default. P2: flip this once `pioneer_predict` calls the real API.
PIONEER_ENABLED: bool = False


def pioneer_predict(a: Persona, b: Persona) -> float:
    """Placeholder for the real Pioneer/Fastino fine-tuned inference call.

    Must return a connection probability in [0, 1] for the pair (a, b) once
    wired. Never called while PIONEER_ENABLED is False.
    """
    # wire the real Pioneer/Fastino fine-tune API call here
    raise NotImplementedError(
        "wire the real Pioneer/Fastino fine-tune API call here"
    )


# ---- "fine-tuned" stand-in: the real learned weights from loop.evolve -----
_RUN_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run.json"
)

# Used only if run.json is missing/unreadable -- demo-truth weighting from
# loop/scorer.py's own __main__ example (trajectory/seeking/collab/expertise
# heavy, domain/focus light).
_FALLBACK_WEIGHTS: list[float] = [0.2, 0.2, 1.3, 1.6, 1.1, 0.9]


def _load_finetuned_weights() -> list[float]:
    """Load the length-6 weight vector from run.json's final generation.

    run.json is written by the real `loop.evolve` run -- these are actual
    learned weights, not hand-picked numbers, so this doubles as "the
    fine-tuned model" for the demo.
    """
    try:
        with open(_RUN_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        weights = data["weights"][-1]
        if isinstance(weights, list) and len(weights) == len(FEATURES):
            return [float(w) for w in weights]
    except (FileNotFoundError, KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return list(_FALLBACK_WEIGHTS)


# ---- pure-Python metrics ----------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two already-unit-normalized embedding vectors."""
    return sum(x * y for x, y in zip(a, b))


def _precision_recall_f1(preds: list[int], labels: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _split(n: int, seed: int, train_fraction: float = 0.7) -> tuple[list[int], list[int]]:
    """Deterministic train/held-out index split."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(train_fraction * n)
    return idx[:cut], idx[cut:]


def _topk_classify_f1(
    train_labels: list[int],
    held_scores: list[float],
    held_labels: list[int],
) -> tuple[float, float, float, float]:
    """Rank-based classifier: predict the top-K highest-scoring held-out pairs
    as positive, where K is the positive rate measured on TRAIN (no peeking at
    held-out labels to pick the cutoff). This mirrors the calibrated,
    rank-driven evaluation `loop.evolve` itself uses (top-K held-out land
    rate) but reports it as a standard precision/recall/F1 triple so the two
    models -- baseline and fine-tuned -- are directly comparable.
    """
    k_fraction = sum(train_labels) / len(train_labels) if train_labels else 0.5
    n = len(held_scores)
    k = max(1, int(round(k_fraction * n)))
    order = sorted(range(n), key=lambda i: held_scores[i], reverse=True)
    preds = [0] * n
    for i in order[:k]:
        preds[i] = 1
    precision, recall, f1 = _precision_recall_f1(preds, held_labels)
    return precision, recall, f1, k_fraction


def evaluate(
    n_personas: int = 300,
    n_pairs: int = 250,
    persona_seed: int = 42,
    pair_seed: int = 43,
    split_seed: int = 44 + 25,  # same personas/pairs seeds and split stream loop.run/evolve use
) -> dict:
    """Generate personas+pairs, score them with both models, and return F1s.

    Baseline: rank-classifier on `embed_cos` alone (cosine similarity of the
    two personas' embeddings) -- ignores the other 5 dimensions entirely.
    Fine-tuned: rank-classifier on `score_features` using the learned 6-dim
    weights from run.json (all six FEATURES dimensions).
    The positive-rate cutoff (K) is measured on TRAIN only; both models are
    then classified on the same HELD-OUT split for a fair F1 comparison.
    """
    personas = generate_personas(n_personas, seed=persona_seed)
    pairs = generate_pairs(personas, n=n_pairs, seed=pair_seed)
    by_id = {p.id: p for p in personas}

    labels = [pr.landed for pr in pairs]
    embed_cos = [_cosine(by_id[pr.a].embedding, by_id[pr.b].embedding) for pr in pairs]

    weights = _load_finetuned_weights()
    six_dim_feats = [pair_features(by_id[pr.a], by_id[pr.b]) for pr in pairs]
    finetuned_scores = [score_features(f, weights) for f in six_dim_feats]

    train_idx, held_idx = _split(len(pairs), seed=split_seed)

    def subset(values: list, idxs: list[int]) -> list:
        return [values[i] for i in idxs]

    train_labels = subset(labels, train_idx)
    held_labels = subset(labels, held_idx)

    base_p, base_r, base_f1, base_k = _topk_classify_f1(
        train_labels, subset(embed_cos, held_idx), held_labels,
    )
    ft_p, ft_r, ft_f1, ft_k = _topk_classify_f1(
        train_labels, subset(finetuned_scores, held_idx), held_labels,
    )

    return {
        "baseline_f1": round(base_f1, 4),
        "finetuned_f1": round(ft_f1, 4),
        "baseline_features": "embed_cos only",
        "finetuned_features": "all six dimensions (learned weights)",
        "baseline_precision": round(base_p, 4),
        "baseline_recall": round(base_r, 4),
        "finetuned_precision": round(ft_p, 4),
        "finetuned_recall": round(ft_r, 4),
        "positive_rate_cutoff": round(ft_k, 4),
        "finetuned_weights": weights,
        "n_held_out": len(held_idx),
    }


if __name__ == "__main__":
    result = evaluate()
    print(
        f"Baseline (embed_cos only): F1={result['baseline_f1']:.2f} | "
        f"Fine-tuned (6-dim learned): F1={result['finetuned_f1']:.2f}"
    )
    print(
        f"  baseline   -> precision={result['baseline_precision']:.2f} "
        f"recall={result['baseline_recall']:.2f}"
    )
    print(
        f"  fine-tuned -> precision={result['finetuned_precision']:.2f} "
        f"recall={result['finetuned_recall']:.2f}"
    )
    print(f"  (n_held_out={result['n_held_out']}, positive-rate cutoff={result['positive_rate_cutoff']:.3f})")
    print(f"  fine-tuned weights (order={FEATURES}): {result['finetuned_weights']}")
    if result["finetuned_f1"] > result["baseline_f1"]:
        print("Fine-tuned model beats the naive embedding-cosine baseline.")
    else:
        print("WARNING: fine-tuned did NOT beat baseline -- retune threshold/weights.")
