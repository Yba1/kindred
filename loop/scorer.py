"""Match scorer for the Kindred evolution loop.

Turns a length-6 feature vector (order == `contracts.FEATURES`) plus a length-6
weight vector into a connection probability in [0, 1] via logistic regression.

Two paths:
  * a numpy-free, pure-stdlib logistic fallback (always available), and
  * a Pioneer/Fastino hook (`pioneer_score_pair`) that is OFF by default.

Pure, deterministic, standard-library only.
"""
from __future__ import annotations

import math

from loop.contracts import Persona, FEATURES
from loop.features import pair_features

# ---- Pioneer/Fastino seam -------------------------------------------------
# OFF by default. When wired, this must return a probability in [0, 1] for the
# pair (a, b). It must NEVER be reached while USE_PIONEER is False, so the
# logistic fallback below stays the sole code path in the default build.
USE_PIONEER: bool = False


def pioneer_score_pair(a: Persona, b: Persona) -> float:
    """Placeholder for a learned Pioneer/Fastino match model."""
    raise NotImplementedError("wire Pioneer/Fastino here")


# ---- logistic fallback ----------------------------------------------------
# Clamp the logit before exp() so extreme inputs cannot overflow math.exp.
_CLAMP: float = 60.0


def _sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid, output in (0, 1)."""
    if z >= 0.0:
        z = min(z, _CLAMP)
        return 1.0 / (1.0 + math.exp(-z))
    z = max(z, -_CLAMP)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def score_features(feats: list[float], weights: list[float], bias: float = 0.0) -> float:
    """logistic(dot(weights, feats) + bias) -> probability in [0, 1].

    `feats` and `weights` are both length-6, aligned to `contracts.FEATURES`.
    """
    assert len(weights) == len(FEATURES), (
        f"weights must be length {len(FEATURES)} (got {len(weights)})"
    )
    assert len(feats) == len(FEATURES), (
        f"feats must be length {len(FEATURES)} (got {len(feats)})"
    )
    z = bias + sum(w * f for w, f in zip(weights, feats))
    return _sigmoid(z)


def score_pair(a: Persona, b: Persona, weights: list[float], bias: float = 0.0) -> float:
    """Score the pair (a, b): probability they connect, in [0, 1]."""
    if USE_PIONEER:
        return pioneer_score_pair(a, b)
    return score_features(pair_features(a, b), weights, bias)


if __name__ == "__main__":
    from loop.personas import generate_personas

    # Demo-truth weighting: domain/focus similarity are weak, while trajectory,
    # seeking, collaboration and expertise-fit carry the real signal.
    sample_weights: list[float] = [0.2, 0.2, 1.3, 1.6, 1.1, 0.9]
    sample_bias: float = -1.0

    personas = generate_personas()
    a, b = personas[0], personas[1]
    score = score_pair(a, b, sample_weights, sample_bias)
    print(f"pair: {a.id} x {b.id}")
    print(score)
