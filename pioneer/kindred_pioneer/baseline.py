"""The embedding-cosine baseline the fine-tuned scorer has to beat.

This is what "semantic match" means before you learn anything from outcomes:
embed both profiles, take the cosine, call the similar ones a match. It gets
every fairness affordance the scorer gets — the same profile text, and a
decision threshold tuned to maximise its own F1 on the validation split.

`TrivialBaseline` is also reported, because a cosine baseline that lands near
"predict every pair connects" should be visible as such rather than quietly
flattering the comparison.
"""

from __future__ import annotations

import numpy as np

from . import embeddings, metrics
from .schema import LabeledPair


class CosineBaseline:
    """score = cosine(embed(bio_a), embed(bio_b)), threshold tuned on validation."""

    name = "cosine baseline"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def scores(self, pairs: list[LabeledPair]) -> np.ndarray:
        return np.array(
            [embeddings.cosine(p.a.bio, p.b.bio) for p in pairs],
            dtype=np.float64,
        )

    def fit_threshold(self, pairs: list[LabeledPair]) -> "CosineBaseline":
        y = np.array([p.label for p in pairs], dtype=np.float64)
        self.threshold = metrics.best_threshold(y, self.scores(pairs))
        return self

    def score_pair(self, a, b) -> float:
        return embeddings.cosine(a.bio, b.bio)


class TrivialBaseline:
    """Predicts "they'll connect" for everyone — the floor any model must clear."""

    name = "always-connect floor"

    def scores(self, pairs: list[LabeledPair]) -> np.ndarray:
        return np.ones(len(pairs), dtype=np.float64)

    def fit_threshold(self, pairs: list[LabeledPair]) -> "TrivialBaseline":
        self.threshold = 0.5
        return self
