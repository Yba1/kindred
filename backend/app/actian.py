"""Actian VectorAI DB client with a numpy cosine fallback.

Design (from the phase cards): ONE client, TWO vector views per record — a DOMAIN
index (topic clumps) and a TRAJECTORY index (shared arc + ask). Upsert on profile
write, top-k pull on match. If Actian is unreachable, everything runs against an
in-process numpy cosine index instead, so the demo never hard-depends on the DB.

Live path (`_ActianBackend`): the gRPC SDK from `actian-vectorai-client` talking
to Actian VectorAI DB on ACTIAN_HOST:ACTIAN_PORT (6574 = gRPC). The two views are
two collections, one point per person in each:

    <db>_domain        vector = the DOMAIN view
    <db>_trajectory    vector = the TRAJECTORY view

A match is two real ANN pulls — one per collection — merged by person id. An id
that only surfaced in one view has its missing similarity completed by fetching
that point's stored vector back out of the other collection, so every candidate
is scored on both axes before the Matcher re-weights them. Actian point ids must
be int/UUID, so the app id ("p_maya") is mapped through uuid5 and carried in the
payload under `kindred_id`; that keeps re-seeds idempotent across restarts.

Collections persist across server restarts, but a restarted server lists them
closed — points calls 404 until `vde.open_collection` reopens them, so the client
does that (and rebuilds an off-Green index) before it touches a collection it
did not just create.

The numpy index stays a hot mirror of every write. If the DB is unreachable at
boot the client never leaves numpy; if it drops out mid-demo the client degrades
onto the mirror at the first failed call and /health flips back to "numpy". That
degrade is deliberately one-way for the life of the process — mid-demo is no
place for reconnect stalls — so restart the backend to pick Actian back up.
"""
from __future__ import annotations

import atexit
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import settings
from .embeddings import cosine

log = logging.getLogger(__name__)

# Actian point ids must be int or UUID; ours are strings like "p_maya". uuid5
# gives a stable, collision-free mapping so re-seeding overwrites the same points
# instead of duplicating them. The app id rides along in the payload.
_ID_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "kindred.actian.vectorai")
_ID_KEY = "kindred_id"


def _point_id(record_id: str) -> str:
    return str(uuid.uuid5(_ID_NS, record_id))


def _collection_names() -> tuple[str, str]:
    """<db>_domain / <db>_trajectory. ACTIAN_DB is just a namespace prefix here."""
    prefix = (settings.actian_db or "kindred").strip() or "kindred"
    return f"{prefix}_domain", f"{prefix}_trajectory"


def _to_list(vec: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(vec, dtype=np.float32).ravel()]


def _score_to_unit(score: float) -> float:
    """Actian Cosine similarity is [-1,1]; embeddings.cosine reports [0,1].

    Map identically so a live score and a fallback score mean the same thing to
    the Matcher's fixed weights.
    """
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


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


class _ActianBackend:
    """Live Actian VectorAI DB backend. Raises on any failure so the client falls back."""

    mode = "actian"

    def __init__(self) -> None:
        self.mirror = _NumpyBackend()          # hot mirror; the degrade target
        self.domain_collection, self.trajectory_collection = _collection_names()
        self._dim: int | None = None           # set once the real embedding width is known
        self._closed = False
        self._client = self._connect()
        # The SDK runs a background event loop; without an explicit shutdown the
        # process lingers ~8s on exit (ugly on a demo Ctrl-C).
        atexit.register(self.close)

    # --- connection --------------------------------------------------------- #
    def _connect(self):
        """Open the gRPC channel and prove the server answers. Raises if it doesn't."""
        from actian_vectorai import VectorAIClient  # optional dep; guarded by caller

        host = settings.actian_host or "localhost"
        port = settings.actian_port or "6574"
        kwargs: dict = {"timeout": settings.actian_timeout}
        if settings.actian_password:               # local container runs auth-disabled
            kwargs["api_key"] = settings.actian_password

        client = VectorAIClient(f"{host}:{port}", **kwargs)
        client.connect()
        info = client.health_check(timeout=settings.actian_timeout)
        log.info(
            "Actian VectorAI DB connected at %s:%s — %s", host, port, info.get("version", info)
        )
        return client

    def ping(self) -> None:
        """Cheap liveness round-trip so /health reports what is actually true."""
        self._client.health_check(timeout=settings.actian_timeout)

    def close(self) -> None:
        if self._closed:                       # atexit may fire after an explicit close
            return
        self._closed = True
        try:
            self._client.shutdown()
        except Exception:  # pragma: no cover - best effort on the way out
            pass

    # --- collections -------------------------------------------------------- #
    def _ensure_collections(self, dim: int) -> None:
        if self._dim == dim:
            return
        for name in (self.domain_collection, self.trajectory_collection):
            self._ensure_collection(name, dim)
        self._dim = dim

    def _ensure_collection(self, name: str, dim: int) -> None:
        from actian_vectorai import DimensionMismatchError, Distance, VectorParams
        from actian_vectorai.models import CollectionStatus

        if self._client.collections.exists(name):
            # A collection that survived a server restart is listed but closed —
            # points calls 404 until it is explicitly reopened.
            self._client.vde.open_collection(name)
            try:
                # Confirm the persisted collection is the width we now embed at
                # (the hash fallback is 256-d, Gemini is 768-d — switching keys
                # between runs must not wedge the store).
                self._client.points.search(
                    name, vector=[1.0] + [0.0] * (dim - 1), limit=1, with_payload=False
                )
            except DimensionMismatchError:
                log.warning("Actian collection %s has a stale vector width; recreating", name)
                self._client.collections.delete(name)
            else:
                # Re-seeding every boot leaves tombstones behind; once the index
                # goes off-Green its recall starts dropping, so rebuild it.
                if self._client.collections.get_info(name).status != CollectionStatus.Green:
                    log.info("Actian collection %s index is stale; rebuilding", name)
                    self._client.vde.rebuild_index(name)
                return

        self._client.collections.create(
            name, vectors_config=VectorParams(size=dim, distance=Distance.Cosine)
        )
        log.info("Actian collection %s created (size=%d, distance=Cosine)", name, dim)

    # --- writes ------------------------------------------------------------- #
    def upsert(self, rec: Record) -> None:
        from actian_vectorai import PointStruct

        self.mirror.upsert(rec)                    # keep the safety net warm
        self._ensure_collections(int(np.asarray(rec.domain_vec).size))

        payload = {**rec.meta, _ID_KEY: rec.id}
        pid = _point_id(rec.id)
        self._client.points.upsert(
            self.domain_collection,
            [PointStruct(id=pid, vector=_to_list(rec.domain_vec), payload=payload)],
        )
        self._client.points.upsert(
            self.trajectory_collection,
            [PointStruct(id=pid, vector=_to_list(rec.traj_vec), payload=payload)],
        )

    # --- reads -------------------------------------------------------------- #
    def query(
        self,
        domain_vec: np.ndarray,
        traj_vec: np.ndarray,
        top_k: int,
        exclude_id: Optional[str] = None,
    ) -> list[Neighbour]:
        self._ensure_collections(int(np.asarray(domain_vec).size))

        # Over-pull: the two views rank differently, and the union is what the
        # Matcher gets to re-score.
        pull = max(top_k * 2, top_k + 1)
        domain_hits = self._search(self.domain_collection, domain_vec, pull)
        traj_hits = self._search(self.trajectory_collection, traj_vec, pull)

        ids = (set(domain_hits) | set(traj_hits)) - {exclude_id}
        # Complete the half-scored candidates against their stored vectors so
        # every neighbour carries both views.
        domain_hits.update(
            self._complete(self.domain_collection, ids - set(domain_hits), domain_vec)
        )
        traj_hits.update(
            self._complete(self.trajectory_collection, ids - set(traj_hits), traj_vec)
        )

        out: list[Neighbour] = []
        for rid in ids:
            d = domain_hits.get(rid)
            t = traj_hits.get(rid)
            meta = (d or t or (0.0, {}))[1]
            out.append(
                Neighbour(
                    id=rid,
                    sim_domain=d[0] if d else 0.0,
                    sim_trajectory=t[0] if t else 0.0,
                    meta=meta,
                )
            )
        out.sort(key=lambda n: n.retrieval_score, reverse=True)
        return out[:top_k]

    def _search(
        self, collection: str, vec: np.ndarray, limit: int
    ) -> dict[str, tuple[float, dict]]:
        hits = self._client.points.search(
            collection, vector=_to_list(vec), limit=limit, with_payload=True
        )
        out: dict[str, tuple[float, dict]] = {}
        for h in hits:
            payload = dict(h.payload or {})
            rid = payload.pop(_ID_KEY, None) or str(h.id)
            out[rid] = (_score_to_unit(h.score), payload)
        return out

    def _complete(
        self, collection: str, ids: set[str], query_vec: np.ndarray
    ) -> dict[str, tuple[float, dict]]:
        """Score ids that only surfaced in the other view, from their stored vectors."""
        if not ids:
            return {}
        points = self._client.points.get(
            collection,
            ids=[_point_id(i) for i in ids],
            with_payload=True,
            with_vectors=True,
        )
        out: dict[str, tuple[float, dict]] = {}
        for p in points:
            payload = dict(p.payload or {})
            rid = payload.pop(_ID_KEY, None)
            raw = getattr(p, "vectors", None)
            if rid is None or raw is None:
                continue
            vec = np.asarray(raw, dtype=np.float32).ravel()
            if vec.size != np.asarray(query_vec).size:
                continue
            out[rid] = (cosine(np.asarray(query_vec, dtype=np.float32), vec), payload)
        return out

    def count(self) -> int:
        """Points in the DOMAIN collection — i.e. people actually indexed in Actian."""
        if self._dim is None:                      # nothing written yet this process
            return self.mirror.count()
        return int(self._client.points.count(self.domain_collection))


class ActianClient:
    """Public store facade. Picks Actian when configured & reachable, else numpy."""

    def __init__(self) -> None:
        self._backend = self._select_backend()

    def _select_backend(self):
        if settings.actian_enabled:
            try:
                return _ActianBackend()
            except Exception as exc:               # any connection/wiring failure -> numpy
                log.warning("Actian unavailable (%s: %s); using numpy index",
                            type(exc).__name__, exc)
        return _NumpyBackend()

    def _degrade(self, exc: Exception):
        """A live call failed mid-flight — fall onto the mirrored numpy index.

        The mirror has every write this process made, so the demo keeps serving
        matches and /health honestly reports "numpy" from here on.
        """
        mirror = getattr(self._backend, "mirror", None)
        if mirror is None:
            return self._backend                   # already on numpy — never swap in an empty one
        log.warning("Actian call failed (%s: %s); degrading to numpy index",
                    type(exc).__name__, exc)
        self._backend = mirror
        return self._backend

    @property
    def mode(self) -> str:
        ping = getattr(self._backend, "ping", None)
        if ping is not None:
            try:
                ping()
            except Exception as exc:
                self._degrade(exc)
        return self._backend.mode

    def upsert(
        self, id: str, domain_vec: np.ndarray, traj_vec: np.ndarray, meta: dict
    ) -> None:
        rec = Record(id=id, domain_vec=domain_vec, traj_vec=traj_vec, meta=meta)
        try:
            self._backend.upsert(rec)
        except Exception as exc:
            self._degrade(exc).upsert(rec)

    def query(
        self,
        domain_vec: np.ndarray,
        traj_vec: np.ndarray,
        top_k: int = 30,
        exclude_id: Optional[str] = None,
    ) -> list[Neighbour]:
        try:
            return self._backend.query(domain_vec, traj_vec, top_k, exclude_id)
        except Exception as exc:
            return self._degrade(exc).query(domain_vec, traj_vec, top_k, exclude_id)

    def count(self) -> int:
        try:
            return self._backend.count()
        except Exception as exc:
            return self._degrade(exc).count()


# module-level singleton (one client for the process)
client = ActianClient()
