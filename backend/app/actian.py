"""Actian vector-store client with a numpy cosine fallback.

Design (from the phase cards): ONE client, TWO vector views per record — a DOMAIN
index (topic clumps) and a TRAJECTORY index (shared arc + ask). Upsert on profile
write, top-k pull on match. If Actian is unreachable, everything runs against an
in-process numpy cosine index instead, so the demo never hard-depends on the DB.

The live Actian wiring is isolated to `_ActianBackend`: fill in `_connect` /
`_upsert` / `_query` against your Actian Vector deployment. Until then (or on any
connection error) the client transparently uses `_NumpyBackend`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import settings
from .embeddings import cosine


@dataclass
class Record:
    id: str
    domain_vec: np.ndarray
    traj_vec: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class Neighbour:
    id: str
    sim_domain: float
    sim_trajectory: float
    meta: dict

    @property
    def retrieval_score(self) -> float:
        # untrimmed pull ranks on both views equally; the Matcher re-scores with weights
        return 0.5 * self.sim_domain + 0.5 * self.sim_trajectory


class _NumpyBackend:
    """Deterministic in-memory cosine index. Always available."""

    mode = "numpy"

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def upsert(self, rec: Record) -> None:
        self._records[rec.id] = rec

    def query(
        self,
        domain_vec: np.ndarray,
        traj_vec: np.ndarray,
        top_k: int,
        exclude_id: Optional[str] = None,
    ) -> list[Neighbour]:
        out: list[Neighbour] = []
        for rid, rec in self._records.items():
            if rid == exclude_id:
                continue
            out.append(
                Neighbour(
                    id=rid,
                    sim_domain=cosine(domain_vec, rec.domain_vec),
                    sim_trajectory=cosine(traj_vec, rec.traj_vec),
                    meta=rec.meta,
                )
            )
        out.sort(key=lambda n: n.retrieval_score, reverse=True)
        return out[:top_k]

    def count(self) -> int:
        return len(self._records)


class _ActianBackend:  # pragma: no cover - requires a live Actian deployment
    """Best-effort live backend. Raises on any failure so the client falls back."""

    mode = "actian"

    def __init__(self) -> None:
        self._client = self._connect()
        self._fallback = _NumpyBackend()  # mirror writes so retrieval works pre-index

    def _connect(self):
        # Fill in against your Actian Vector deployment, e.g. a psycopg/HTTP client
        # using settings.actian_host / _db / _user / _password. Kept as an explicit
        # NotImplementedError so an unconfigured deploy cleanly uses numpy instead.
        raise NotImplementedError("Actian live client not wired; using numpy fallback")

    def upsert(self, rec: Record) -> None:
        self._fallback.upsert(rec)
        # self._client.upsert("domain_idx", rec.id, rec.domain_vec, rec.meta)
        # self._client.upsert("traj_idx",   rec.id, rec.traj_vec,   rec.meta)

    def query(self, domain_vec, traj_vec, top_k, exclude_id=None) -> list[Neighbour]:
        # dv = self._client.knn("domain_idx", domain_vec, top_k)
        # tv = self._client.knn("traj_idx",   traj_vec,   top_k)
        # ... merge by id ...
        return self._fallback.query(domain_vec, traj_vec, top_k, exclude_id)

    def count(self) -> int:
        return self._fallback.count()


class ActianClient:
    """Public store facade. Picks Actian when configured & reachable, else numpy."""

    def __init__(self) -> None:
        self._backend = self._select_backend()

    def _select_backend(self):
        if settings.actian_enabled:
            try:
                return _ActianBackend()
            except Exception:
                pass  # any connection/wiring failure -> numpy
        return _NumpyBackend()

    @property
    def mode(self) -> str:
        return self._backend.mode

    def upsert(
        self, id: str, domain_vec: np.ndarray, traj_vec: np.ndarray, meta: dict
    ) -> None:
        self._backend.upsert(Record(id=id, domain_vec=domain_vec, traj_vec=traj_vec, meta=meta))

    def query(
        self,
        domain_vec: np.ndarray,
        traj_vec: np.ndarray,
        top_k: int = 30,
        exclude_id: Optional[str] = None,
    ) -> list[Neighbour]:
        return self._backend.query(domain_vec, traj_vec, top_k, exclude_id)

    def count(self) -> int:
        return self._backend.count()


# module-level singleton (one client for the process)
client = ActianClient()
