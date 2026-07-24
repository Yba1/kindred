"""Assemble the graph payload — the A->B contract plus the one additive extension.

  {center, nodes:[{id,name,score,x,y}], edges:[{source,target,weight,features}],
   reasons:{id:[str]}, meta:{featureNames, weights}}

The original four keys are unchanged and still rendered verbatim. `features` is
the per-dim vector a weight vector multiplies, and every edge's `weight` IS
`weighted_mean(features, meta.weights)` — so a client that never re-scores sees
exactly the graph this service returned before features existed, and a client
that does re-scores without another round trip.

Edges are no longer just a star out of the centre. A star has nothing to
re-cluster: pull on any `w` and every node still hangs off the same single hub.
So matched people are also wired to each other, which is what gives a new `w`
clusters to melt and re-form.

x,y seed a radial layout (higher score -> smaller radius) so the frontend has
positions before its force sim runs.
"""
from __future__ import annotations

import math

from .matcher import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    ROUND,
    Match,
    pair_features,
    weighted_mean,
)
from .schemas import GraphEdge, GraphMeta, GraphNode, GraphResponse, Profile

_GOLDEN = math.pi * (3 - math.sqrt(5))  # golden angle -> even angular spread

# Inter-person wiring. A kNN union, not the full clique: the clique is O(n^2)
# (1225 edges at top_k=50) and mostly noise, whereas each person's strongest few
# ties are what actually form the clumps the re-cluster animation melts.
PERSON_EDGE_KNN = 6        # strongest partners kept per person — the real budget
PERSON_EDGE_FLOOR = 0.35   # sanity floor; rarely bites, since cosine() maps an
                           # unrelated pair to 0.5 rather than 0
MAX_PERSON_EDGES = 300     # hard cap, strongest first — 147 edges at top_k=50


def _person_edges(
    matches: list[Match],
    knn: int = PERSON_EDGE_KNN,
    floor: float = PERSON_EDGE_FLOOR,
    cap: int = MAX_PERSON_EDGES,
) -> list[GraphEdge]:
    """Person<->person ties: each person's strongest `knn` partners, unioned."""
    people = [m for m in matches if m.profile is not None]
    scored: dict[tuple[str, str], tuple[float, list[float]]] = {}
    per_node: dict[str, list[tuple[float, tuple[str, str]]]] = {m.id: [] for m in people}

    for i, a in enumerate(people):
        for b in people[i + 1:]:
            features = pair_features(a.profile, b.profile)
            weight = round(weighted_mean(features), ROUND)
            if weight < floor:
                continue
            key = (a.id, b.id)
            scored[key] = (weight, features)
            per_node[a.id].append((weight, key))
            per_node[b.id].append((weight, key))

    keep: set[tuple[str, str]] = set()
    for ties in per_node.values():
        ties.sort(key=lambda t: t[0], reverse=True)
        keep.update(key for _, key in ties[:knn])

    ranked = sorted(((scored[k][0], k) for k in keep), key=lambda t: (-t[0], t[1]))[:cap]
    return [
        GraphEdge(source=k[0], target=k[1], weight=w, features=scored[k][1])
        for w, k in ranked
    ]


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
        edges.append(
            GraphEdge(source="user", target=m.id, weight=m.score, features=list(m.features))
        )
        reasons[m.id] = m.reasons

    edges.extend(_person_edges(matches))

    return GraphResponse(
        center="user",
        nodes=nodes,
        edges=edges,
        reasons=reasons,
        meta=GraphMeta(featureNames=list(FEATURE_NAMES), weights=list(DEFAULT_WEIGHTS)),
    )
