"""Assemble the graph payload — EXACTLY the A->B contract.

  {center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight}],
   reasons:{id:[str]}}

x,y seed a radial layout (higher score -> smaller radius) so the frontend has
positions before its force sim runs. No other keys — the frontend renders this
verbatim.
"""
from __future__ import annotations

import math

from .matcher import BLEND, Match
from .schemas import GraphEdge, GraphNode, GraphResponse, Profile

_GOLDEN = math.pi * (3 - math.sqrt(5))  # golden angle -> even angular spread


def _raw_features(m: Match) -> list[float]:
    """Undo the Matcher's fixed BLEND weighting to recover raw per-dim
    similarities in [0,1], ordered [domain, trajectory, seeking, stage] to
    match the frontend's FEATURE_NAMES. This is what lets window.applyWeights
    actually rescore edges instead of leaving them pinned to the Matcher's
    own fixed formula.

    No live "stage"/expertise signal exists on Profile yet, so it's a
    placeholder mirroring seeking-fit until that field is added — flagged so
    it isn't mistaken for a real independent dimension.
    """
    domain = m.components.get("domain", 0.0) / BLEND["domain"]
    trajectory = m.components.get("trajectory", 0.0) / BLEND["trajectory"]
    seeking = m.components.get("seeking", 0.0) / BLEND["seeking"]
    stage = seeking  # placeholder — no expertise/stage signal on Profile yet
    return [round(min(1.0, max(0.0, x)), 4) for x in (domain, trajectory, seeking, stage)]


def build_graph(user: Profile, matches: list[Match]) -> GraphResponse:
    nodes: list[GraphNode] = [
        GraphNode(id="user", name=user.name or "You", score=1.0, x=0.0, y=0.0)
    ]
    edges: list[GraphEdge] = []
    reasons: dict[str, list[str]] = {}

    for i, m in enumerate(matches):
        radius = 90.0 + (1.0 - max(0.0, min(1.0, m.score))) * 320.0
        angle = i * _GOLDEN
        nodes.append(
            GraphNode(
                id=m.id,
                name=m.name,
                score=m.score,
                x=round(radius * math.cos(angle), 2),
                y=round(radius * math.sin(angle), 2),
            )
        )
        edges.append(GraphEdge(source="user", target=m.id, weight=m.score, features=_raw_features(m)))
        reasons[m.id] = m.reasons

    return GraphResponse(center="user", nodes=nodes, edges=edges, reasons=reasons)
