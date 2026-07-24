"""`score_pair(person_a, person_b) -> float` — the judge the evolution loop calls.

    from kindred_pioneer import score_pair
    score_pair(alice, bob)          # 0.0 .. 1.0, probability the intro lands

Accepts `Person` objects or plain dicts, so the loop can hand over whatever the
Profiler produced without a conversion step. `score_pairs` batches, which is how
a whole generation gets scored in one pass.

Backend selection (`KINDRED_SCORER_BACKEND`):
  * `local`   — the fitted logistic head in artifacts/model.json. Default. No network.
  * `pioneer` — the fine-tuned Pioneer model, via POST /inference. Needs
                PIONEER_API_KEY and KINDRED_PIONEER_MODEL_ID.

The scorer is frozen once trained: it loads the artifact and never refits. If the
judge moved every generation, the loop would be chasing a moving target.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

from . import features, paths
from .model import LogisticScorer
from .schema import Person

PersonLike = Person | dict[str, Any]


class ScorerNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _local_model() -> LogisticScorer:
    if not paths.MODEL_PATH.exists():
        raise ScorerNotTrained(
            f"no trained scorer at {paths.MODEL_PATH}. Run: python -m kindred_pioneer.train"
        )
    return LogisticScorer.load(paths.MODEL_PATH)


@lru_cache(maxsize=1)
def _pioneer_handle() -> tuple[Any, str]:
    from . import pioneer_client

    model_id = os.environ.get("KINDRED_PIONEER_MODEL_ID")
    if not model_id:
        raise ScorerNotTrained(
            "KINDRED_PIONEER_MODEL_ID is not set — it is the training job id from "
            "`python -m kindred_pioneer.train --backend pioneer`."
        )
    return pioneer_client.PioneerClient(), model_id


def backend() -> str:
    return os.environ.get("KINDRED_SCORER_BACKEND", "local").lower()


def score_pairs(pairs: Sequence[tuple[PersonLike, PersonLike]]) -> list[float]:
    """Score many pairs at once. Same numbers as score_pair, fewer round trips."""
    people = [(Person.from_any(a), Person.from_any(b)) for a, b in pairs]
    if not people:
        return []

    if backend() == "pioneer":
        client, model_id = _pioneer_handle()
        return [
            float(client.score_text(model_id, features.pair_to_text(a, b))) for a, b in people
        ]

    model = _local_model()
    X = np.vstack([features.pair_features(a, b) for a, b in people])
    return [float(v) for v in model.predict_proba(X)]


def score_pair(person_a: PersonLike, person_b: PersonLike) -> float:
    """Probability in [0, 1] that these two people actually connect."""
    return score_pairs([(person_a, person_b)])[0]


def decide(person_a: PersonLike, person_b: PersonLike) -> bool:
    """Boolean call at the model's calibrated threshold — for loops that want a verdict."""
    return score_pair(person_a, person_b) >= threshold()


def threshold() -> float:
    """The decision threshold fitted on validation, not a hardcoded 0.5."""
    if backend() == "pioneer":
        return float(os.environ.get("KINDRED_PIONEER_THRESHOLD", "0.5"))
    return float(_local_model().threshold)


def explain(person_a: PersonLike, person_b: PersonLike, top_k: int = 4) -> dict[str, Any]:
    """Score plus the features that drove it — feeds the graph's per-edge reasoning.

    Contribution is weight x standardised feature value, so it reads as
    "this pushed the score up/down by this much in logit space".
    """
    a, b = Person.from_any(person_a), Person.from_any(person_b)
    model = _local_model()
    x = features.pair_features(a, b)
    z = (x - model.mean) / model.scale
    contributions = sorted(
        zip(model.feature_names, (model.weights * z).tolist(), x.tolist()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    score = float(model.predict_proba(x.reshape(1, -1))[0])
    return {
        "score": score,
        "threshold": float(model.threshold),
        "verdict": "connect" if score >= model.threshold else "pass",
        "drivers": [
            {"feature": name, "value": round(value, 4), "contribution": round(contrib, 4)}
            for name, contrib, value in contributions[:top_k]
        ],
    }


def info() -> dict[str, Any]:
    """Backend + artifact provenance, for /health and the demo's 'which model is this'."""
    payload: dict[str, Any] = {"backend": backend()}
    if backend() == "pioneer":
        payload["model_id"] = os.environ.get("KINDRED_PIONEER_MODEL_ID")
        payload["ready"] = bool(payload["model_id"]) and bool(os.environ.get("PIONEER_API_KEY"))
        return payload
    try:
        model = _local_model()
    except ScorerNotTrained as exc:
        return {**payload, "ready": False, "error": str(exc)}
    return {
        **payload,
        "ready": True,
        "artifact": str(paths.MODEL_PATH),
        "trained_backend": model.backend,
        "threshold": float(model.threshold),
        "features": list(model.feature_names),
    }
