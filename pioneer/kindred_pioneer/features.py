"""Pair featurisation — the view of a pair the fine-tuned scorer gets.

Two hard rules:

1. Every feature is computable from two `Person` records at inference time.
   Nothing here can see the label or the generative probability.
2. `cos_bio` — the baseline's entire signal — is feature 0. The scorer is
   strictly better informed than the baseline rather than differently informed,
   so "scorer beats baseline" is a statement about using more of the profile,
   not about handing the two models different data.

Order matters: `FEATURE_NAMES` is the wire format of the saved model.
"""

from __future__ import annotations

import numpy as np

from . import embeddings
from .schema import DIRECTIONAL_ASKS, RECIPROCAL_ASKS, Person

FEATURE_NAMES = (
    "cos_bio",            # what the embedding baseline sees, verbatim
    "reciprocal_fit",     # both sides want the same peer relationship
    "directional_fit",    # mean of A-asks/B-offers and B-asks/A-offers
    "mutual_fit",         # both directions non-empty
    "topic_jaccard",      # shared interest tags
    "same_domain",
    "same_arc",           # same prior domain — "same trajectory"
    "stage_gap",
    "seniority_gap",
    "same_city",
)
N_FEATURES = len(FEATURE_NAMES)


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pair_features(a: Person, b: Person) -> np.ndarray:
    """Symmetric feature vector for an unordered pair."""
    seek_a, seek_b = set(a.seeking), set(b.seeking)
    offer_a, offer_b = set(a.offering), set(b.offering)

    reciprocal = len(seek_a & seek_b & RECIPROCAL_ASKS) / max(1, len(RECIPROCAL_ASKS))
    dir_ab = len(seek_a & offer_b) / max(1, len(seek_a & DIRECTIONAL_ASKS))
    dir_ba = len(seek_b & offer_a) / max(1, len(seek_b & DIRECTIONAL_ASKS))

    same_domain = float(a.domain == b.domain)
    same_arc = float(a.prior_domain == b.prior_domain)

    return np.array(
        [
            embeddings.cosine(a.bio, b.bio),
            reciprocal,
            (dir_ab + dir_ba) / 2.0,
            float(dir_ab > 0 and dir_ba > 0),
            _jaccard(a.interests, b.interests),
            same_domain,
            same_arc,
            abs(a.stage_index - b.stage_index) / 2.0,
            min(abs(a.seniority - b.seniority), 15) / 15.0,
            float(a.city == b.city and a.city != "remote"),
        ],
        dtype=np.float64,
    )


def feature_matrix(pairs) -> np.ndarray:
    """Stack features for a list of LabeledPair (or (a, b) tuples)."""
    rows = []
    for item in pairs:
        a, b = (item.a, item.b) if hasattr(item, "a") else item
        rows.append(pair_features(a, b))
    return np.vstack(rows) if rows else np.zeros((0, N_FEATURES))


def labels(pairs) -> np.ndarray:
    return np.array([p.label for p in pairs], dtype=np.float64)


def pair_to_text(a: Person, b: Person) -> str:
    """Serialise a pair for Pioneer's text classifier.

    The SLM reads the same facts the feature vector encodes, just in prose —
    so the two backends are trained on the same information.
    """
    return (
        "PERSON A: " + a.bio + "\n"
        "PERSON B: " + b.bio + "\n"
        f"SHARED INTERESTS: {', '.join(sorted(set(a.interests) & set(b.interests))) or 'none'}\n"
        f"A ASKS: {', '.join(a.seeking) or 'none'} | B OFFERS: {', '.join(b.offering) or 'none'}\n"
        f"B ASKS: {', '.join(b.seeking) or 'none'} | A OFFERS: {', '.join(a.offering) or 'none'}\n"
        f"TRAJECTORY: A {a.prior_domain} -> {a.domain} | B {b.prior_domain} -> {b.domain}\n"
        f"STAGE: A {a.stage} | B {b.stage}"
    )
