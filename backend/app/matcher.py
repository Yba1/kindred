"""Matcher — turns retrieved neighbours into scored, reasoned matches.

Score blends three axes with a fixed weighting (no learning loop — that's
workstream D):

    score = 0.34 * sim_domain          # shared topic
          + 0.36 * sim_trajectory      # shared arc  (weighted a touch higher:
          + 0.30 * seeking_fit         #   "closest in meaning" > "closest in topic")

`seeking_fit` is mutual complementarity: how well YOUR ask matches what THEY do,
and theirs matches what you do. `reasons` are the 1-3 short strings the frontend
shows on node click.

Those same numbers ALSO ship per edge as a feature vector, ordered over

    FEATURE_NAMES = ["topic", "trajectory", "seeking", "stage"]

so the frontend can re-score every edge under a learned weight vector `w` with
no round trip. An edge's weight is exactly `weighted_mean(features)` under
DEFAULT_WEIGHTS — the same formula the frontend applies in
`frontend/src/weights/rescore.js` — so a client that never re-scores sees the
identical graph this service returned before features existed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np

from .actian import Neighbour
from .embeddings import cosine, embed
from .profiler import apply_gemini_reasons
from .schemas import Profile

BLEND = {"domain": 0.34, "trajectory": 0.36, "seeking": 0.30}

# --------------------------------------------------------------------------- #
#  The A->B feature contract. Order matters — it is index-aligned with
#  FEATURE_NAMES in frontend/src/weights/rescore.js. Do not reorder.
# --------------------------------------------------------------------------- #
FEATURE_NAMES: tuple[str, ...] = ("topic", "trajectory", "seeking", "stage")

# The fixed blend above, expressed over FEATURE_NAMES. `stage` carries 0.0 on
# purpose: it is NOT part of the similarity model this service already ships, so
# giving it mass here would move scores the frontend renders today. It is emitted
# as a feature anyway, to hand the Evaluator a fourth, near-orthogonal dimension
# to put mass on when its learned `w` arrives.
DEFAULT_WEIGHTS: tuple[float, ...] = (
    BLEND["domain"], BLEND["trajectory"], BLEND["seeking"], 0.0,
)

ROUND = 4          # score/weight precision — unchanged from before features existed
FEATURE_ROUND = 6  # finer, so deriving the score FROM the features moves nothing


@dataclass
class Match:
    id: str
    name: str
    score: float
    reasons: list[str]
    components: dict[str, float]
    features: list[float] = field(default_factory=list)
    profile: Optional[Profile] = None   # kept so the graph can wire person<->person


# --------------------------------------------------------------------------- #
#  Scoring primitives (mirrors of the frontend's rescore.js)
# --------------------------------------------------------------------------- #
def clamp01(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def weighted_mean(features: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """sum(w_i * f_i) / sum(w_i) — `scoreEdge` from rescore.js, ported verbatim.

    This is the ONE place an edge weight is computed, for both the user's edges
    and the person<->person ones, so the payload weight and anything the frontend
    recomputes under the same `w` cannot drift.
    """
    w = DEFAULT_WEIGHTS if weights is None else tuple(weights)
    acc = mass = 0.0
    for i in range(min(len(features), len(w))):
        wi = max(0.0, float(w[i]))
        acc += wi * clamp01(features[i])
        mass += wi
    return clamp01(acc / mass) if mass > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Embedding helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4096)
def _vec(text: str) -> np.ndarray:
    """Memoized `embed`. The same profile text is compared many times per /graph
    call once person<->person edges exist; embedding is the expensive bit."""
    return embed(text)


def _text_sim(a: str, b: str) -> float:
    """Cosine under the same embedding the vector store indexes on, so a
    person<->person similarity is numerically the same thing the Actian client
    returns for user<->person."""
    return cosine(_vec(a), _vec(b))


def as_profile(meta) -> Profile:
    """Rebuild a Profile from the meta dict the vector store carries."""
    if isinstance(meta, Profile):
        return meta
    data = {k: v for k, v in (meta or {}).items() if k in Profile.model_fields}
    data.setdefault("id", "unknown")
    data.setdefault("name", data["id"])
    return Profile(**data)


# --------------------------------------------------------------------------- #
#  seeking — mutual want<->offer complementarity
# --------------------------------------------------------------------------- #
def _offer_text(p: Profile) -> str:
    return " ".join(filter(None, [p.domain, " ".join(p.roles), " ".join(p.tags)]))


def seeking_fit(a: Profile, b: Profile) -> float:
    """Mutual want<->offer complementarity, in [0,1]. Symmetric in a/b."""
    a_ask, b_ask = (a.seeking or "").strip(), (b.seeking or "").strip()
    if not a_ask and not b_ask:
        return 0.0
    a_offer, b_offer = _offer_text(a), _offer_text(b)
    x = _text_sim(a_ask, b_offer) if a_ask and b_offer else 0.0
    y = _text_sim(b_ask, a_offer) if b_ask and a_offer else 0.0
    both = _text_sim(a_ask, b_ask) if a_ask and b_ask else 0.0
    return max(x, y, 0.6 * both)


# --------------------------------------------------------------------------- #
#  stage — the fourth feature dim
# --------------------------------------------------------------------------- #
# Where someone sits on the build arc, read off fields the profile ALREADY holds
# (`seeking`, nudged by `trajectory`/`summary`/`tags`). No new similarity model —
# a re-reading of existing data. The ask someone leads with is the most reliable
# stage tell we have: you look for a cofounder before you look for a first
# engineer, and for a first engineer before a growth lead.
_ASK_LADDER: tuple[tuple[str, float], ...] = (
    # most specific phrases first — first hit wins
    ("technical cofounder", 0.15),
    ("cofounder", 0.15),
    ("co-founder", 0.15),
    ("founding partner", 0.15),
    ("mentor", 0.25),
    ("advisor", 0.25),
    ("design partner", 0.40),
    ("early user", 0.40),
    ("beta user", 0.40),
    ("pilot customer", 0.40),   # NOT bare "pilot" — it is a substring of "copilot"
    ("pilot user", 0.40),
    ("pre-seed", 0.55),
    ("seed round", 0.55),
    ("funding", 0.55),
    ("investor", 0.55),
    ("angel", 0.55),
    ("founding engineer", 0.70),
    ("first engineer", 0.70),
    ("first hire", 0.70),
    ("head of", 0.90),
    ("growth lead", 0.90),
    ("customers", 0.90),
    ("engineer", 0.80),
    ("scientist", 0.80),
    ("designer", 0.80),
    ("researcher", 0.80),
    ("manager", 0.80),
    ("lead", 0.80),
)
_STAGE_NEUTRAL = 0.5

# Secondary cues over the prose, so stage is not a pure function of the ask string.
_STAGE_NUDGES: tuple[tuple[float, tuple[str, ...]], ...] = (
    (-0.10, ("idea", "exploring", "figuring out", "prototype", "just left", "nights and weekends")),
    (+0.10, ("hiring", "scaling", "revenue", "series a", "our team", "customers")),
)


def stage_position(p: Profile) -> float:
    """A person's position on the build arc, in [0,1]. 0 = pre-formation."""
    ask = (p.seeking or "").lower()
    pos = _STAGE_NEUTRAL
    for phrase, value in _ASK_LADDER:
        if phrase in ask:
            pos = value
            break
    context = " ".join(filter(None, [p.trajectory, p.summary, " ".join(p.tags)])).lower()
    for delta, cues in _STAGE_NUDGES:
        if any(cue in context for cue in cues):
            pos += delta
    return clamp01(pos)


def stage_affinity(a: Profile, b: Profile) -> float:
    """1.0 when two people are at the same point on the arc, 0.0 at opposite ends.

    Same semantics the frontend already renders for this dim in its stub payload:
    same-stage pairs score 1.0 and it falls off with the gap.
    """
    return clamp01(1.0 - abs(stage_position(a) - stage_position(b)))


# --------------------------------------------------------------------------- #
#  Feature vectors
# --------------------------------------------------------------------------- #
def edge_features(sim_domain: float, sim_trajectory: float, fit: float, stage: float) -> list[float]:
    """The A->B feature vector, clamped to 0..1 and index-aligned to FEATURE_NAMES."""
    return [round(clamp01(v), FEATURE_ROUND) for v in (sim_domain, sim_trajectory, fit, stage)]


def pair_features(a: Profile, b: Profile) -> list[float]:
    """Features for ANY two people — the model /graph already runs user<->neighbour,
    applied person<->person so the graph has ties that can re-cluster."""
    return edge_features(
        _text_sim(a.domain_text(), b.domain_text()),
        _text_sim(a.trajectory_text(), b.trajectory_text()),
        seeking_fit(a, b),
        stage_affinity(a, b),
    )


# --------------------------------------------------------------------------- #
#  Reasons + ranking
# --------------------------------------------------------------------------- #
def _reasons(user: Profile, cand: Profile, comp: dict[str, float]) -> list[str]:
    out: list[str] = []
    cand_traj = (cand.trajectory or "").strip()
    if comp["trajectory"] >= comp["domain"] and cand_traj:
        out.append(f"same trajectory: {user.trajectory or user.domain} ↔ {cand_traj}")
    dom = (cand.domain or "").strip()
    if dom and dom == user.domain:
        out.append(f"shared domain: {dom}")
    user_ask = user.seeking.strip()
    cand_ask = (cand.seeking or "").strip()
    if comp["seeking"] > 0.2:
        if user_ask and cand_ask and _text_sim(user_ask, cand_ask) > 0.6:
            out.append(f"both seeking {user_ask}")
        elif user_ask and dom:
            out.append(f"you're seeking {user_ask}; they're building in {dom}")
        elif cand_ask:
            out.append(f"they're seeking {cand_ask}")
    if not out and cand_traj:
        out.append(f"nearest arc to yours: {cand_traj}")
    seen, uniq = set(), []
    for r in out:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq[:3]


def rank(user: Profile, neighbours: Iterable[Neighbour], limit: int) -> list[Match]:
    matches: list[Match] = []
    meta_by_id: dict[str, dict] = {}  # kept for the Gemini reason pass below
    for n in neighbours:
        cand = as_profile(n.meta)
        meta_by_id[n.id] = n.meta
        fit = seeking_fit(user, cand)
        features = edge_features(n.sim_domain, n.sim_trajectory, fit, stage_affinity(user, cand))
        comp = {
            "domain": BLEND["domain"] * n.sim_domain,
            "trajectory": BLEND["trajectory"] * n.sim_trajectory,
            "seeking": BLEND["seeking"] * fit,
        }
        # Identical to comp's sum under DEFAULT_WEIGHTS — but computed through the
        # SAME function the frontend uses, off the SAME rounded features it will
        # receive, so weight and re-scored weight agree to the last decimal.
        score = weighted_mean(features)
        matches.append(
            Match(
                id=n.id,
                name=n.meta.get("name", n.id) if isinstance(n.meta, dict) else cand.name,
                score=round(score, ROUND),
                reasons=_reasons(user, cand, comp),
                components={k: round(v, ROUND) for k, v in comp.items()},
                features=features,
                profile=cand,
            )
        )
    matches.sort(key=lambda m: m.score, reverse=True)
    top = matches[:limit]
    # Upgrade the template reasons to Gemini's when a key is configured. One
    # call for the whole set, and a no-op without a key — so the panel still
    # reads sensibly on the fallback path.
    apply_gemini_reasons(user, top, meta_by_id)
    return top
