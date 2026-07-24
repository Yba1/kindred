"""Deterministic, label-calibrated pair generation for the Kindred loop.

`generate_pairs(personas, n, seed)` samples `n` distinct-persona pairs (biased
toward SAME-DOMAIN, which is what a naive matcher would chase) and assigns each
a ground-truth `landed` label drawn from a HIDDEN true model.

The hidden truth (the whole point of the demo): whether two people actually
connect is driven by their TRAJECTORY overlap, compatible SEEKING/intent, a
little shared collaboration STYLE, and being ~one expertise step apart. Sharing
a DOMAIN alone does NOT help (it carries a tiny penalty). The Evaluator has to
discover this from outcomes rather than from the obvious topic-similarity.

Calibration targets (measured over generated pairs):
  * same-domain pairs land ~41%  (must fall in [0.38, 0.44])
  * pairs sharing trajectory AND seeking land ~80-88%
"""
from __future__ import annotations

import math
import random

from loop.contracts import EXPERTISE, Pair, Persona

# ---- hidden true-model constants (tuned so same-domain land-rate ~= 0.41) ----
A0: float = -1.05          # baseline intercept
A_TRAJ: float = 1.05       # shared trajectory (real signal)
A_SEEK: float = 1.05       # compatible intent (real signal)
A_STYLE: float = 0.55      # matching collaboration style (mild signal)
A_EXPERT: float = 0.45     # expertise ~one step apart (mild signal)
DOMAIN_PENALTY: float = 0.25   # same-domain alone slightly HURTS

SAME_DOMAIN_BIAS: float = 0.50  # prob. a sampled pair is forced same-topic

# ---- seeking/intent compatibility (symmetric) ----
# Listed truth: cofounder<->cofounder, hire<->peer, investor<->founder,
# mentor<->student, customer<->operator. The founder/student/operator tokens are
# stage-flavoured, so they are mapped onto the nearest SEEKING value here.
_SEEKING_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"cofounder"}),            # cofounder <-> cofounder
    frozenset({"hire", "peer"}),         # hire <-> peer
    frozenset({"investor", "cofounder"}),  # investor <-> founder(=cofounder)
    frozenset({"mentor", "hire"}),       # mentor <-> student(=hire)
    frozenset({"customer", "peer"}),     # customer <-> operator(=peer)
})


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _trajectory_match(a: tuple, b: tuple) -> bool:
    """True if trajectories share their final two stages, or overlap heavily."""
    ta, tb = tuple(a), tuple(b)
    if len(ta) >= 2 and len(tb) >= 2 and ta[-2:] == tb[-2:]:
        return True
    if ta and tb and ta[-1] == tb[-1] and (len(ta) == 1 or len(tb) == 1):
        return True
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return False
    jaccard = len(sa & sb) / len(sa | sb)
    return jaccard >= 0.6


def _seeking_match(sa: str, sb: str) -> bool:
    return frozenset({sa, sb}) in _SEEKING_PAIRS


def _expertise_one_step(ea: str, eb: str) -> bool:
    try:
        return abs(EXPERTISE.index(ea) - EXPERTISE.index(eb)) == 1
    except ValueError:
        return False


def _land_probability(
    same_topic: bool,
    same_trajectory: bool,
    seeking_match: bool,
    same_style: bool,
    expertise_one_step: bool,
) -> float:
    """Hidden true model: trajectory + seeking (+ style, expertise) drive landing."""
    z = A0
    z += A_TRAJ * (1.0 if same_trajectory else 0.0)
    z += A_SEEK * (1.0 if seeking_match else 0.0)
    z += A_STYLE * (1.0 if same_style else 0.0)
    z += A_EXPERT * (1.0 if expertise_one_step else 0.0)
    z -= DOMAIN_PENALTY * (1.0 if same_topic else 0.0)
    return _sigmoid(z)


def _by_topic(personas: list[Persona]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for idx, p in enumerate(personas):
        buckets.setdefault(p.topic, []).append(idx)
    return buckets


def generate_pairs(personas: list[Persona], n: int = 250, seed: int = 43) -> list[Pair]:
    """Sample `n` distinct-persona pairs and label them via the hidden true model."""
    rng = random.Random(seed)
    total = len(personas)
    if total < 2:
        return []

    by_topic = _by_topic(personas)
    multi_topics = [t for t, ids in by_topic.items() if len(ids) >= 2]

    seen: set[tuple[int, int]] = set()
    pairs: list[Pair] = []
    attempts = 0
    max_attempts = n * 200

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1

        if multi_topics and rng.random() < SAME_DOMAIN_BIAS:
            ids = by_topic[rng.choice(multi_topics)]
            i, j = rng.sample(ids, 2)
        else:
            i, j = rng.sample(range(total), 2)

        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)

        a, b = personas[key[0]], personas[key[1]]

        same_topic = a.topic == b.topic
        same_trajectory = _trajectory_match(a.trajectory, b.trajectory)
        seeking_match = _seeking_match(a.seeking, b.seeking)
        same_style = a.style == b.style
        expertise_one_step = _expertise_one_step(a.expertise, b.expertise)

        prob = _land_probability(
            same_topic, same_trajectory, seeking_match, same_style, expertise_one_step
        )
        landed = 1 if rng.random() < prob else 0

        pairs.append(Pair(
            a=a.id,
            b=b.id,
            same_topic=same_topic,
            same_trajectory=same_trajectory,
            seeking_match=seeking_match,
            same_style=same_style,
            landed=landed,
        ))

    return pairs


def _rate(pairs: list[Pair], predicate) -> tuple[float, int]:
    subset = [p for p in pairs if predicate(p)]
    if not subset:
        return 0.0, 0
    return sum(p.landed for p in subset) / len(subset), len(subset)


if __name__ == "__main__":
    from loop.personas import generate_personas

    people = generate_personas()
    pairs = generate_pairs(people)

    same_domain_rate, n_domain = _rate(pairs, lambda p: p.same_topic)
    traj_seek_rate, n_ts = _rate(
        pairs, lambda p: p.same_trajectory and p.seeking_match
    )
    overall_rate = sum(p.landed for p in pairs) / len(pairs) if pairs else 0.0

    print(f"n pairs                     = {len(pairs)}")
    print(f"same-domain land-rate       = {same_domain_rate:.3f}  (n={n_domain}, target 0.38-0.44)")
    print(f"trajectory&seeking land-rate= {traj_seek_rate:.3f}  (n={n_ts}, target 0.80-0.88)")
    print(f"overall land-rate           = {overall_rate:.3f}")

    assert 0.38 <= same_domain_rate <= 0.44, (
        f"same-domain land-rate {same_domain_rate:.3f} outside [0.38, 0.44] - retune A0"
    )
    print("calibration OK: same-domain land-rate in [0.38, 0.44]")
