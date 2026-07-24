"""Assemble the graph payload — EXACTLY the A->B contract.

  {center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight}],
   reasons:{id:[str]}}

x,y seed a radial layout (higher score -> smaller radius) so the frontend has
positions before its force sim runs. No other keys — the frontend renders this
verbatim.
"""
from __future__ import annotations

import math

from .matcher import Match
from .schemas import GraphEdge, GraphNode, GraphResponse, Profile

_GOLDEN = math.pi * (3 - math.sqrt(5))  # golden angle -> even angular spread


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
        edges.append(GraphEdge(source="user", target=m.id, weight=m.score))
        reasons[m.id] = m.reasons

    return GraphResponse(center="user", nodes=nodes, edges=edges, reasons=reasons)
