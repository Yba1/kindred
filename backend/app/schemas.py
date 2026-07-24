"""Request/response schemas.

Two contracts must not drift:

  /profile out — {roles, trajectory, seeking, tags, embedding}. The profile is
                 reasoning (roles + trajectory + the ask), never bare topic tags.

  /graph out   — EXACTLY {center, nodes:[{id,name,score,x,y}],
                 edges:[{source,target,weight}], reasons:{id:[str]}}.
                 No extra keys — the frontend renders this verbatim.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

_WORD = re.compile(r"[A-Za-z']+")


# --------------------------------------------------------------------------- #
#  /profile
# --------------------------------------------------------------------------- #
class ProfileRequest(BaseModel):
    """Raw intake context in. Rejects tag-only payloads — /profile reasons over
    prose, it does not accept a bag of topics."""

    context: str = Field(..., min_length=1)
    name: Optional[str] = None
    id: Optional[str] = None

    @field_validator("context")
    @classmethod
    def not_tag_only(cls, v: str) -> str:
        v = v.strip()
        if len(_WORD.findall(v)) < 6:
            raise ValueError(
                "context is too thin to profile — give a sentence or two about your "
                "path and what you're looking for, not a tag list"
            )
        chunks = [c.strip() for c in re.split(r"[,|/]", v) if c.strip()]
        if len(chunks) >= 3 and all(len(_WORD.findall(c)) <= 2 for c in chunks):
            raise ValueError(
                "looks like a tag list — /profile needs prose describing your "
                "trajectory and your ask, not comma-separated topics"
            )
        return v


class ProfileOut(BaseModel):
    """The /profile response contract."""

    id: str
    name: str
    roles: list[str] = Field(default_factory=list)
    trajectory: str = ""
    seeking: str = ""
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


class Profile(BaseModel):
    """Internal semantic profile (stored + matched on). Superset of ProfileOut."""

    id: str
    name: str
    roles: list[str] = Field(default_factory=list)
    trajectory: str = ""       # e.g. "finance -> agent infra"
    seeking: str = ""          # the explicit ask, e.g. "technical cofounder"
    tags: list[str] = Field(default_factory=list)
    domain: str = ""           # topical area — feeds the DOMAIN vector view
    summary: str = ""          # one-liner — shown nowhere in /graph, used internally
    source: str = "gemini"     # "gemini" | "heuristic" | "seed"

    def domain_text(self) -> str:
        """Text behind the DOMAIN vector view (topic clumps)."""
        return " ".join(filter(None, [self.domain, " ".join(self.tags), " ".join(self.roles)]))

    def trajectory_text(self) -> str:
        """Text behind the TRAJECTORY vector view (shared arc + ask)."""
        return " ".join(filter(None, [self.trajectory, self.seeking, self.summary]))


# --------------------------------------------------------------------------- #
#  /graph
# --------------------------------------------------------------------------- #
class GraphRequest(BaseModel):
    """Match a person against the store. `context` is profiled on the fly; pass a
    pre-built `profile` to skip re-profiling."""

    context: Optional[str] = None
    profile: Optional[Profile] = None
    name: Optional[str] = None
    top_k: int = Field(default=12, ge=1, le=50)

    @field_validator("profile")
    @classmethod
    def _need_one(cls, v, info):
        if v is None and not info.data.get("context"):
            raise ValueError("provide either `context` or `profile`")
        return v


class GraphNode(BaseModel):
    id: str
    name: str
    score: float
    x: float = 0.0
    y: float = 0.0


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float


class GraphResponse(BaseModel):
    center: str = "user"
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    reasons: dict[str, list[str]] = Field(default_factory=dict)
