"""Kindred match scorer — workstream C (Pioneer/Fastino fine-tune + wiring).

The one thing the rest of the repo needs:

    from kindred_pioneer import score_pair
    score_pair(person_a, person_b)  # -> float in [0, 1]

Everything else (data generation, training, evaluation, the HTTP shim) hangs
off that.
"""

from .schema import LabeledPair, Person
from .scorer import (
    ScorerNotTrained,
    backend,
    decide,
    explain,
    info,
    score_pair,
    score_pairs,
    threshold,
)

__all__ = [
    "LabeledPair",
    "Person",
    "ScorerNotTrained",
    "backend",
    "decide",
    "explain",
    "info",
    "score_pair",
    "score_pairs",
    "threshold",
]

__version__ = "0.1.0"
