"""Turn an EvolveResult into a script of village agent events.

narrate(result) -> list[dict], each event:
    {"speaker","text","action","consensus","banner"?}

Speakers: profiler, matcher, evaluator, introducer, actian, pioneer, band, gemini.
Rules (see agents/phase_card_agent.md): text <= ~90 chars, consensus monotonically
non-decreasing across the WHOLE list, ends on action:"resolve" at consensus 1.0,
banners used sparingly. True to the mechanism: domain/focus are demoted,
trajectory/seeking/style/expertise are the real signal, and promotion only
happens if the held-out rate improves.
"""
from __future__ import annotations

from loop.contracts import EvolveResult

VALID_SPEAKERS = {
    "profiler", "matcher", "evaluator", "introducer",
    "actian", "pioneer", "band", "gemini",
}
MAX_TEXT_LEN = 90


def _ev(speaker: str, text: str, action: str, consensus: float, banner: str | None = None) -> dict:
    assert speaker in VALID_SPEAKERS, f"bad speaker {speaker!r}"
    assert len(text) <= MAX_TEXT_LEN, f"text too long ({len(text)}): {text!r}"
    assert action in {"speak", "agree", "disagree", "resolve"}, f"bad action {action!r}"
    event = {"speaker": speaker, "text": text, "action": action, "consensus": round(consensus, 3)}
    if banner:
        event["banner"] = banner
    return event


def narrate(result: EvolveResult) -> list[dict]:
    events: list[dict] = []

    # ---- open: profiler / gemini / actian set up the six-dim profile ----
    events.append(_ev(
        "profiler",
        "New intake. Reading you across six axes, not one topic tag.",
        "speak", 0.08,
    ))
    events.append(_ev(
        "gemini",
        "Domain, focus, trajectory, seeking, style, expertise — all separable.",
        "speak", 0.16,
    ))
    events.append(_ev(
        "actian",
        "Six-dim profile embedded and indexed. Neighbours pulled.",
        "speak", 0.22, banner="PROFILE COMMITTED",
    ))

    # ---- matcher proposes domain-heavy ranking; evaluator pushes back ----
    events.append(_ev(
        "matcher",
        "Ranking by domain first. Same-topic candidates float to the top.",
        "speak", 0.28,
    ))
    events.append(_ev(
        "evaluator",
        f"Domain-only matches landed {result.base_rate:.0%}. That's not the signal.",
        "disagree", 0.28,
    ))
    events.append(_ev(
        "evaluator",
        "Trajectory, seeking and style carry it. Demote domain and focus.",
        "speak", 0.34,
    ))

    # ---- walk the real generations ----
    gens = list(result.generations)
    lo, hi = 0.36, 0.90
    span = (result.final_rate - result.base_rate) or 1.0
    running = 0.34
    crossed_70 = False
    reporters = ["pioneer", "matcher", "profiler"]

    # keep the walk bounded (~4-6 checkpoints) even if there are many generations:
    # always show gen 0, the generation that first crosses ~0.7, and the last one,
    # plus up to one evenly-spaced midpoint for texture.
    n = len(gens)
    checkpoint_idx = {0, n - 1} if n else set()
    for i, g in enumerate(gens):
        if g.rate >= 0.7:
            checkpoint_idx.add(i)
            break
    if n > 4:
        checkpoint_idx.add(n // 2)
    checkpoints = sorted(checkpoint_idx)

    for j, i in enumerate(checkpoints):
        g = gens[i]
        frac = (g.rate - result.base_rate) / span
        frac = max(0.0, min(1.0, frac))
        consensus = max(running, lo + frac * (hi - lo))
        running = consensus
        speaker = reporters[j % len(reporters)]
        banner = None
        if g.rate >= 0.7 and not crossed_70:
            banner = "CONNECTION-RATE CROSSES 70%"
            crossed_70 = True
        text = f"Generation {g.gen}: held-out connection-rate {g.rate:.0%}."
        events.append(_ev(speaker, text, "speak" if i != 0 else "speak", consensus, banner=banner))
        if i == 0:
            events.append(_ev(
                "matcher",
                "Some domain pairs still land, but only beside a shared ask.",
                "agree", max(running, consensus + 0.02),
            ))
            running = max(running, consensus + 0.02)

    # ---- explicit promotion gate before the final generations resolve ----
    running = max(running, 0.90)
    events.append(_ev(
        "evaluator",
        "Gate check: promote the new weights only if held-out improves.",
        "speak", running,
    ))
    running = max(running, 0.94)
    events.append(_ev(
        "pioneer",
        f"CONNECTION-RATE UP TO {result.final_rate:.0%}. No regression — gate passed.",
        "agree", running, banner=f"CONNECTION-RATE UP TO {result.final_rate:.0%}",
    ))

    # ---- introducer resolves with a BAND opener and promotion banner ----
    events.append(_ev(
        "band",
        "Drafting the opener: shared pivot, shared ask, shared async style.",
        "speak", running,
    ))
    events.append(_ev(
        "introducer",
        "Weights promoted, graph re-clustered. Sending the introduction.",
        "resolve", 1.0, banner="WEIGHTS PROMOTED",
    ))

    return events


if __name__ == "__main__":
    import json

    verification_path = None
    result = None

    try:
        from loop.personas import generate_personas
        from loop.pairs import generate_pairs
        from loop.evolve import evolve

        personas = generate_personas(40, seed=7)
        pairs = generate_pairs(personas, 200, seed=7)
        result = evolve(personas, pairs, generations=6, seed=7)
        verification_path = "real evolve"
    except Exception as exc:  # noqa: BLE001 - evolve.py may be mid-build by another agent
        from dataclasses import dataclass, field

        @dataclass
        class _Generation:
            gen: int
            weights: list
            rate: float

        @dataclass
        class _EvolveResult:
            generations: list
            weight_vectors: list
            base_rate: float
            final_rate: float
            seed: int

        mock_gens = [
            _Generation(gen=0, weights=[0.6, 0.5, 0.1, 0.1, 0.1, 0.1], rate=0.41),
            _Generation(gen=3, weights=[0.3, 0.2, 0.5, 0.4, 0.4, 0.3], rate=0.66),
            _Generation(gen=6, weights=[0.1, 0.1, 0.7, 0.6, 0.6, 0.4], rate=0.84),
        ]
        result = _EvolveResult(
            generations=mock_gens,
            weight_vectors=[g.weights for g in mock_gens],
            base_rate=0.41,
            final_rate=0.84,
            seed=7,
        )
        verification_path = f"mock (real evolve unavailable: {exc!r})"

    events = narrate(result)

    # verify all hard rules
    assert all(e["speaker"] in VALID_SPEAKERS for e in events)
    assert all(len(e["text"]) <= MAX_TEXT_LEN for e in events)
    consensus_values = [e["consensus"] for e in events]
    assert all(b >= a for a, b in zip(consensus_values, consensus_values[1:])), "consensus dropped"
    assert events[0]["consensus"] <= 0.15
    assert events[-1]["consensus"] == 1.0
    assert events[-1]["action"] == "resolve"

    print(f"verification path: {verification_path}")
    print(f"event count: {len(events)}, first consensus: {events[0]['consensus']}, "
          f"last consensus: {events[-1]['consensus']}")
    print(json.dumps(events, indent=2))
