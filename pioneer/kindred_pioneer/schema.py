"""Person and LabeledPair records — shared by the generator, trainer and scorer.

This is the only place the pair vocabulary lives. The Profiler (workstream A)
produces the same shape from raw user context; everything downstream of here
treats a Person as already-profiled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Asks that only land when *both* sides want the same thing — peer relationships.
RECIPROCAL_ASKS = frozenset({
    "cofounder",
    "accountability partner",
    "peer group",
})

# Asks that land when one side wants it and the other side can supply it.
DIRECTIONAL_ASKS = frozenset({
    "first customers",
    "seed capital",
    "intro to investors",
    "ml hiring",
    "design help",
    "go-to-market advice",
    "technical mentorship",
    "regulatory guidance",
    "distribution partner",
})

ALL_ASKS = RECIPROCAL_ASKS | DIRECTIONAL_ASKS

STAGES = ("exploring", "building", "scaling")
STAGE_INDEX = {name: i for i, name in enumerate(STAGES)}


@dataclass
class Person:
    """A profiled person. `bio` is the rendered text the embedding baseline sees."""

    id: str
    name: str
    domain: str
    prior_domain: str
    stage: str
    seniority: int
    city: str
    interests: list[str] = field(default_factory=list)
    seeking: list[str] = field(default_factory=list)
    offering: list[str] = field(default_factory=list)
    bio: str = ""

    def __post_init__(self) -> None:
        if self.stage not in STAGE_INDEX:
            raise ValueError(f"unknown stage {self.stage!r}; expected one of {STAGES}")
        if not self.bio:
            self.bio = render_bio(self)

    @property
    def stage_index(self) -> int:
        return STAGE_INDEX[self.stage]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, value: "Person | dict[str, Any]") -> "Person":
        """Accept a Person or a plain dict — callers across the repo send both."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            known = {f for f in cls.__dataclass_fields__}
            missing = {"id", "name", "domain", "prior_domain", "stage", "seniority", "city"} - value.keys()
            if missing:
                raise ValueError(f"person is missing required fields: {sorted(missing)}")
            return cls(**{k: v for k, v in value.items() if k in known})
        raise TypeError(f"cannot read a Person from {type(value).__name__}")


@dataclass
class LabeledPair:
    """One training row: two people and whether the intro actually landed."""

    a: Person
    b: Person
    label: int  # 1 = they connected, 0 = the intro went nowhere
    p_true: float = 0.0  # generative probability; diagnostics only, never a feature

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "label": self.label,
            "p_true": round(self.p_true, 6),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "LabeledPair":
        return cls(
            a=Person.from_any(row["a"]),
            b=Person.from_any(row["b"]),
            label=int(row["label"]),
            p_true=float(row.get("p_true", 0.0)),
        )


def render_bio(p: Person) -> str:
    """Flatten a profile into the blurb the embedding baseline is scored on.

    Everything the feature extractor uses is present in this text, so the
    baseline is not being starved of information it could in principle read.
    """
    parts = [
        f"{p.name}, {p.seniority} years in, currently {p.stage}.",
        f"Came out of {p.prior_domain} and now works in {p.domain}.",
    ]
    if p.interests:
        parts.append(f"Focused on {', '.join(p.interests)}.")
    if p.seeking:
        parts.append(f"Looking for {', '.join(p.seeking)}.")
    if p.offering:
        parts.append(f"Can help with {', '.join(p.offering)}.")
    parts.append(f"Based in {p.city}.")
    return " ".join(parts)
