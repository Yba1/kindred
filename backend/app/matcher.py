"""Matcher — turns retrieved neighbours into scored, reasoned matches.

Score blends three axes with a fixed weighting (no learning loop — that's
workstream D):

    score = 0.34 * sim_domain          # shared topic
          + 0.36 * sim_trajectory      # shared arc  (weighted a touch higher:
          + 0.30 * seeking_fit         #   "closest in meaning" > "closest in topic")

`seeking_fit` is mutual complementarity: how well YOUR ask matches what THEY do,
and theirs matches what you do. `reasons` are the 1-3 short strings the frontend
shows on node click.
"""
from __future__ import annotations

from dataclasses import dataclass

from .actian import Neighbour
from .embeddings import cosine, embed
from .schemas import Profile

BLEND = {"domain": 0.34, "trajectory": 0.36, "seeking": 0.30}


@dataclass
class Match:
    id: str
    name: str
    score: float
    reasons: list[str]
    components: dict[str, float]


def _offer_text(meta: dict) -> str:
    return " ".join(filter(None, [
        meta.get("domain", ""),
        " ".join(meta.get("roles", []) or []),
        " ".join(meta.get("tags", []) or []),
    ]))


def _seeking_fit(user: Profile, cand: dict) -> float:
    """Mutual want<->offer complementarity, in [0,1]."""
    user_ask = user.seeking.strip()
    cand_ask = (cand.get("seeking") or "").strip()
    cand_offer = _offer_text(cand)
    user_offer = " ".join(filter(None, [user.domain, " ".join(user.roles), " ".join(user.tags)]))
    if not user_ask and not cand_ask:
        return 0.0
    a = cosine(embed(user_ask), embed(cand_offer)) if user_ask and cand_offer else 0.0
    b = cosine(embed(cand_ask), embed(user_offer)) if cand_ask and user_offer else 0.0
    both = cosine(embed(user_ask), embed(cand_ask)) if user_ask and cand_ask else 0.0
    return max(a, b, 0.6 * both)


def _reasons(user: Profile, cand: dict, comp: dict[str, float]) -> list[str]:
    out: list[str] = []
    cand_traj = (cand.get("trajectory") or "").strip()
    if comp["trajectory"] >= comp["domain"] and cand_traj:
        out.append(f"same trajectory: {user.trajectory or user.domain} ↔ {cand_traj}")
    dom = (cand.get("domain") or "").strip()
    if dom and dom == user.domain:
        out.append(f"shared domain: {dom}")
    user_ask = user.seeking.strip()
    cand_ask = (cand.get("seeking") or "").strip()
    if comp["seeking"] > 0.2:
        if user_ask and cand_ask and cosine(embed(user_ask), embed(cand_ask)) > 0.6:
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


def rank(user: Profile, neighbours: list[Neighbour], limit: int) -> list[Match]:
    matches: list[Match] = []
    for n in neighbours:
        fit = _seeking_fit(user, n.meta)
        comp = {
            "domain": BLEND["domain"] * n.sim_domain,
            "trajectory": BLEND["trajectory"] * n.sim_trajectory,
            "seeking": BLEND["seeking"] * fit,
        }
        score = comp["domain"] + comp["trajectory"] + comp["seeking"]
        matches.append(
            Match(
                id=n.id,
                name=n.meta.get("name", n.id),
                score=round(score, 4),
                reasons=_reasons(user, n.meta, comp),
                components={k: round(v, 4) for k, v in comp.items()},
            )
        )
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:limit]
