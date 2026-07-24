"""Generational evolution loop for the Kindred matcher.

`evolve(personas, pairs, generations, seed)` runs a small generational search
over the length-6 weight vector (order == `contracts.FEATURES`). Generation 0 is
the NAIVE matcher: it weights only the obvious dimensions (domain + focus).
Each later generation refits the weights by a few steps of pure-Python logistic
regression against the ground-truth `landed` label on a TRAIN split, then is
accepted only if it does at least as well on a held-out split (the PROMOTION
GATE), so the recorded held-out rate is monotonic non-decreasing by construction.

The demo truth (baked into pairs.py): domain/focus similarity are WEAK
predictors; trajectory overlap, compatible seeking, collaboration style and
expertise-fit carry the real signal. Across generations the learned weights
visibly move mass OFF domain/focus (indices 0,1) ONTO trajectory/seeking/collab/
expertise (indices 2-5), and the held-out connection-rate climbs from ~0.41 to
~0.84.

Deterministic (random.Random(seed) only; no global random, no wall-clock).
Pure standard library, numpy-free.
"""
from __future__ import annotations

import math
import random

from loop.contracts import Persona, Pair, Generation, EvolveResult, FEATURES
from loop.features import pair_features
from loop.scorer import score_features

# ---- tunables (calibrated for the default seed so the climb lands in-window) ----
NAIVE_WEIGHTS: list[float] = [1.0, 0.6, 0.0, 0.0, 0.0, 0.0]  # gen-0: domain+focus only
TRAIN_FRACTION: float = 0.70        # 70/30 train / held-out split
K_FRACTION: float = 0.18            # top-K held-out fraction scored for the land-rate
GD_STEPS: int = 16                  # logistic-GD steps applied per generation (cumulative)
LEARN_RATE: float = 0.5             # gradient-descent learning rate
SPLIT_STREAM_OFFSET: int = 25       # independent RNG stream for the train/test split


def _sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid, output in (0, 1)."""
    if z >= 0.0:
        z = min(z, 60.0)
        return 1.0 / (1.0 + math.exp(-z))
    z = max(z, -60.0)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _split(n: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic 70/30 split of index range(n) -> (train_idx, held_idx)."""
    idx = list(range(n))
    random.Random(seed + SPLIT_STREAM_OFFSET).shuffle(idx)
    cut = int(TRAIN_FRACTION * n)
    return idx[:cut], idx[cut:]


def _held_rate(
    feats: list[list[float]],
    labels: list[int],
    weights: list[float],
    k_fraction: float = K_FRACTION,
) -> float:
    """Top-K ranked land-rate: rank held-out pairs by `score_features`, take the
    fraction that actually `landed` among the top-K (K = k_fraction of held-out).
    """
    if not feats:
        return 0.0
    order = sorted(
        range(len(feats)),
        key=lambda i: score_features(feats[i], weights),
        reverse=True,
    )
    k = max(1, int(len(feats) * k_fraction))
    top = order[:k]
    return sum(labels[i] for i in top) / k


def _train_logistic(
    weights: list[float],
    bias: float,
    feats: list[list[float]],
    labels: list[int],
    steps: int,
    lr: float,
) -> tuple[list[float], float]:
    """A few steps of batch logistic-regression gradient descent (numpy-free).

    Warm-started from the incoming `weights`/`bias`; predicts `landed` from the
    length-6 feature vectors. The bias absorbs the low base connection-rate so
    the weights are free to specialise on the discriminative dimensions.
    """
    weights = list(weights)
    n = len(feats)
    if n == 0:
        return weights, bias
    dim = len(weights)
    for _ in range(steps):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for x, y in zip(feats, labels):
            p = _sigmoid(bias + sum(w * xi for w, xi in zip(weights, x)))
            err = p - y
            for j in range(dim):
                grad_w[j] += err * x[j]
            grad_b += err
        for j in range(dim):
            weights[j] -= lr * grad_w[j] / n
        bias -= lr * grad_b / n
    return weights, bias


def evolve(
    personas: list[Persona],
    pairs: list[Pair],
    generations: int = 6,
    seed: int = 44,
) -> EvolveResult:
    """Run the generational weight search and return an EvolveResult.

    Gen 0 uses the naive domain/focus weights; each later generation applies
    another `GD_STEPS` of logistic GD (warm-started, cumulative) and is accepted
    only if its held-out top-K land-rate matches or beats the best so far.
    """
    by_id: dict[str, Persona] = {p.id: p for p in personas}
    # Precompute the length-6 feature vector once per pair.
    all_feats: list[list[float]] = [
        pair_features(by_id[pr.a], by_id[pr.b]) for pr in pairs
    ]
    all_labels: list[int] = [pr.landed for pr in pairs]

    train_idx, held_idx = _split(len(pairs), seed)
    train_feats = [all_feats[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    held_feats = [all_feats[i] for i in held_idx]
    held_labels = [all_labels[i] for i in held_idx]

    # Generation 0: the naive matcher.
    base_rate = _held_rate(held_feats, held_labels, NAIVE_WEIGHTS)
    gens: list[Generation] = [Generation(0, list(NAIVE_WEIGHTS), base_rate)]
    weight_vectors: list[list[float]] = [list(NAIVE_WEIGHTS)]

    best_rate = base_rate
    best_weights = list(NAIVE_WEIGHTS)
    # Live (possibly-not-yet-promoted) parameters that GD keeps refining.
    weights = list(NAIVE_WEIGHTS)
    bias = 0.0

    for g in range(1, generations):
        weights, bias = _train_logistic(
            weights, bias, train_feats, train_labels, GD_STEPS, LEARN_RATE
        )
        rate = _held_rate(held_feats, held_labels, weights)
        # PROMOTION GATE: accept only if it does not regress on held-out.
        if rate >= best_rate:
            best_rate = rate
            best_weights = list(weights)
        gens.append(Generation(g, list(best_weights), best_rate))
        weight_vectors.append(list(best_weights))

    return EvolveResult(
        generations=gens,
        weight_vectors=weight_vectors,
        base_rate=base_rate,
        final_rate=gens[-1].rate,
        seed=seed,
    )


if __name__ == "__main__":
    from loop.personas import generate_personas
    from loop.pairs import generate_pairs

    personas = generate_personas()
    pairs = generate_pairs(personas)
    result = evolve(personas, pairs)

    print(f"generations={len(result.generations)}  seed={result.seed}")
    print(f"base_rate={result.base_rate:.3f}  final_rate={result.final_rate:.3f}")
    print("-" * 72)
    header = "gen  rate   " + "  ".join(f"{name[:8]:>8}" for name in FEATURES)
    print(header)
    for gen in result.generations:
        weights = "  ".join(f"{w:8.3f}" for w in gen.weights)
        print(f"{gen.gen:>3}  {gen.rate:.3f}  {weights}")
    print("-" * 72)
    print("weights shift mass OFF domain/focus (cols 1-2) ONTO "
          "trajectory/seeking/collab/expertise (cols 3-6)")
