"""Pairwise feature extraction for the Kindred evolution loop.

`pair_features(a, b)` returns a length-6 vector aligned to `contracts.FEATURES`:
    ["domain_sim", "focus_sim", "trajectory_sim",
     "seeking_match", "collab_sim", "expertise_fit"]

Pure and deterministic. Every value lies in [0, 1]. Standard library only.
"""
from __future__ import annotations

import math

from loop.contracts import Persona, FEATURES, EXPERTISE

# ---- seeking compatibility (symmetric) ----
# Each unordered pair below is a compatible intent match; membership is checked
# order-independently via frozenset.
_SEEKING_PAIRS: set[frozenset] = {
    frozenset({"cofounder"}),            # cofounder <-> cofounder
    frozenset({"hire", "peer"}),
    frozenset({"investor", "founder"}),
    frozenset({"mentor", "student"}),
    frozenset({"customer", "operator"}),
}

# ---- expertise_fit: reward "one step ahead" ----
# Keyed by absolute EXPERTISE index distance. Peaks at exactly 1 level apart,
# 0.6 at the same level, decaying for larger gaps. All values in [0, 1].
_EXPERTISE_FIT: dict[int, float] = {0: 0.6, 1: 1.0, 2: 0.4, 3: 0.2}


def _domain_sim(a: Persona, b: Persona) -> float:
    return 1.0 if a.topic == b.topic else 0.0


def _focus_sim(a: Persona, b: Persona) -> float:
    return 1.0 if a.focus == b.focus else 0.0


def _trajectory_sim(a: Persona, b: Persona) -> float:
    """Jaccard overlap of the two trajectories' stage sets, in [0, 1]."""
    sa, sb = set(a.trajectory), set(b.trajectory)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _seeking_match(a: Persona, b: Persona) -> float:
    return 1.0 if frozenset({a.seeking, b.seeking}) in _SEEKING_PAIRS else 0.0


def _collab_sim(a: Persona, b: Persona) -> float:
    return 1.0 if a.style == b.style else 0.0


def _expertise_fit(a: Persona, b: Persona) -> float:
    """Peaks when the pair is ~one EXPERTISE level apart (one step ahead)."""
    try:
        dist = abs(EXPERTISE.index(a.expertise) - EXPERTISE.index(b.expertise))
    except ValueError:
        return 0.0
    if dist in _EXPERTISE_FIT:
        return _EXPERTISE_FIT[dist]
    # Fallback for any larger index space: smooth decay peaking at dist == 1.
    return max(0.0, math.exp(-((dist - 1) ** 2) / 2.0))


def pair_features(a: Persona, b: Persona) -> list[float]:
    """Return the length-6 feature vector for (a, b), ordered per FEATURES."""
    vec = [
        _domain_sim(a, b),
        _focus_sim(a, b),
        _trajectory_sim(a, b),
        _seeking_match(a, b),
        _collab_sim(a, b),
        _expertise_fit(a, b),
    ]
    assert len(vec) == len(FEATURES), "feature vector must align to FEATURES"
    # Clamp defensively so every value is guaranteed in [0, 1].
    return [min(1.0, max(0.0, float(v))) for v in vec]


if __name__ == "__main__":
    from loop.personas import generate_personas

    personas = generate_personas(4, seed=42)
    a, b = personas[0], personas[1]
    features = pair_features(a, b)
    print(f"a = {a.name}: topic={a.topic} focus={a.focus} "
          f"seeking={a.seeking} style={a.style} expertise={a.expertise}")
    print(f"b = {b.name}: topic={b.topic} focus={b.focus} "
          f"seeking={b.seeking} style={b.style} expertise={b.expertise}")
    for name, val in zip(FEATURES, features):
        print(f"  {name:<14} {val:.3f}")
    print(f"vector (len {len(features)}): {features}")
