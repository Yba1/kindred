"""Deterministic persona generator for the Kindred evolution loop.

`generate_personas(n, seed)` builds ~n Persona records spread evenly across
TOPICS. Every field is drawn from `random.Random(seed)` ONLY, so a given seed
always yields byte-identical output (critical for replay).

The six profile dimensions (topic, focus, stage, trajectory, seeking, style,
expertise) are assigned largely independently so the Evaluator can later learn
separate weights. Embeddings are the exception: they are correlated with topic
(per-topic base vector + small noise, unit-normalized).
"""
from __future__ import annotations

import math
import random

from loop.contracts import (
    Persona, TOPICS, FOCI, STAGES, SEEKING, STYLES, EXPERTISE, EMBED_DIM,
)

# ---- name fragments (deterministic pools, stdlib only) ----
_FIRST = [
    "Ada", "Ben", "Cara", "Devi", "Ezra", "Fay", "Gio", "Hana", "Ivo", "Jun",
    "Kai", "Lena", "Milo", "Nia", "Omar", "Priya", "Quinn", "Rafe", "Sana",
    "Theo", "Uma", "Vik", "Wren", "Xena", "Yara", "Zane",
]
_LAST = [
    "Adeyemi", "Bianchi", "Chen", "Dubois", "Eriksen", "Farah", "Gupta",
    "Haas", "Ibarra", "Jansen", "Kowalski", "Lindqvist", "Mensah", "Novak",
    "Okafor", "Pereira", "Qureshi", "Rossi", "Silva", "Tanaka", "Ueda",
    "Vasquez", "Wong", "Xu", "Yilmaz", "Zhao",
]


def _unit(vec: list[float]) -> list[float]:
    """Return `vec` scaled to unit length (safe against zero vectors)."""
    norm = math.sqrt(sum(c * c for c in vec))
    if norm == 0.0:
        return vec
    return [c / norm for c in vec]


def _topic_bases(rng: random.Random) -> dict[str, list[float]]:
    """One fixed base vector per topic; personas cluster around these."""
    bases: dict[str, list[float]] = {}
    for topic in TOPICS:
        bases[topic] = _unit([rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)])
    return bases


def _trajectory_for(stage: str, rng: random.Random) -> tuple[str, ...]:
    """Ordered slice of STAGES ending at the persona's current stage."""
    end = STAGES.index(stage)
    length = rng.randint(1, min(end + 1, 4))
    start = end - length + 1
    return tuple(STAGES[start:end + 1])


def generate_personas(n: int = 300, seed: int = 42) -> list[Persona]:
    """Generate `n` deterministic personas seeded by `seed`.

    Six profile dimensions are drawn independently. Embeddings are a per-topic
    base vector plus small per-persona noise, normalized to unit length, so
    within-topic cosine > cross-topic cosine.
    """
    rng = random.Random(seed)
    bases = _topic_bases(rng)
    noise_scale = 0.35

    personas: list[Persona] = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]  # even round-robin distribution

        # Six dimensions, each its own independent draw.
        name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        focus = rng.choice(FOCI)
        stage = rng.choice(STAGES)
        trajectory = _trajectory_for(stage, rng)
        seeking = rng.choice(SEEKING)
        style = rng.choice(STYLES)
        expertise = rng.choice(EXPERTISE)

        # Topic-correlated embedding.
        base = bases[topic]
        embedding = _unit([
            base[d] + rng.uniform(-noise_scale, noise_scale)
            for d in range(EMBED_DIM)
        ])

        personas.append(Persona(
            id=f"p{i + 1:04d}",
            name=name,
            topic=topic,
            focus=focus,
            stage=stage,
            trajectory=trajectory,
            seeking=seeking,
            style=style,
            expertise=expertise,
            embedding=embedding,
        ))
    return personas


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both unit-normalized


def _mean_cosines(personas: list[Persona]) -> tuple[float, float]:
    """Return (mean within-topic cosine, mean cross-topic cosine)."""
    within: list[float] = []
    cross: list[float] = []
    for i, a in enumerate(personas):
        for b in personas[i + 1:]:
            cos = _cosine(a.embedding, b.embedding)
            (within if a.topic == b.topic else cross).append(cos)
    mean_within = sum(within) / len(within) if within else 0.0
    mean_cross = sum(cross) / len(cross) if cross else 0.0
    return mean_within, mean_cross


if __name__ == "__main__":
    people = generate_personas()
    print(f"generated {len(people)} personas")
    for p in people[:2]:
        print(f"  {p.id} {p.name!r} topic={p.topic} focus={p.focus} "
              f"stage={p.stage} trajectory={p.trajectory} seeking={p.seeking} "
              f"style={p.style} expertise={p.expertise}")
        print(f"    embedding[:4]={[round(x, 3) for x in p.embedding[:4]]}")
    mean_within, mean_cross = _mean_cosines(people)
    print(f"mean within-topic cosine = {mean_within:.4f}")
    print(f"mean cross-topic cosine  = {mean_cross:.4f}")
    assert mean_within > mean_cross, "within-topic cosine must exceed cross-topic"
    print("sanity check OK: within-topic > cross-topic")
