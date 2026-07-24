"""People store — profile registry + vector index, wired to the Actian client.

`register()` embeds a profile's two views (domain + trajectory) and upserts them,
so every profile write lands in the vector store. It returns the concatenated
[domain || trajectory] embedding, which /profile hands back to the caller.
Seeded people load on import.
"""
from __future__ import annotations

import numpy as np

from .actian import client as actian_client
from .embeddings import embed
from .profiler import derive_roles
from .schemas import Profile
from .seed import PERSONAS


class PeopleStore:
    def __init__(self) -> None:
        self._profiles: dict[str, Profile] = {}
        self._vecs = actian_client

    def register(self, profile: Profile) -> list[float]:
        """Store the profile + its two vector views. Returns the [domain||traj] embedding."""
        domain_vec = embed(profile.domain_text())
        traj_vec = embed(profile.trajectory_text())
        self._profiles[profile.id] = profile
        self._vecs.upsert(
            id=profile.id, domain_vec=domain_vec, traj_vec=traj_vec, meta=profile.model_dump()
        )
        return np.concatenate([domain_vec, traj_vec]).round(6).tolist()

    def get(self, id: str) -> Profile | None:
        return self._profiles.get(id)

    def all(self) -> list[Profile]:
        return list(self._profiles.values())

    def count(self) -> int:
        return len(self._profiles)

    def seed(self) -> int:
        for p in dict_personas():
            self.register(p)
        return self.count()


def dict_personas() -> list[Profile]:
    out: list[Profile] = []
    for d in PERSONAS:
        roles = d.get("roles") or derive_roles(d.get("trajectory", ""), d.get("seeking", ""),
                                                d.get("domain", ""))
        out.append(Profile(source="seed", **{**d, "roles": roles}))
    return out


store = PeopleStore()
store.seed()
