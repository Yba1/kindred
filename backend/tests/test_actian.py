"""Actian VectorAI DB client tests.

Two halves:
  * the fallback contract — numpy engages when the DB is unconfigured, unreachable,
    or dies mid-flight. Runs everywhere, offline.
  * the live round-trip — skipped unless Actian VectorAI DB is actually up on
    ACTIAN_HOST:ACTIAN_PORT (start it with the docker command in .env.example).
    Uses its own collection prefix so it never touches the demo collections.
"""
from dataclasses import replace

import numpy as np
import pytest

from app import actian as actian_mod
from app.actian import ActianClient, _NumpyBackend, _point_id, _score_to_unit
from app.config import settings
from app.embeddings import cosine, embed


def _vec(text: str) -> np.ndarray:
    return embed(text)


# --------------------------------------------------------------------------- #
#  fallback contract (always runs)
# --------------------------------------------------------------------------- #
def test_disabled_config_uses_numpy(monkeypatch):
    monkeypatch.setattr(actian_mod, "settings", replace(settings, actian_host=""))
    assert ActianClient().mode == "numpy"


def test_unreachable_host_falls_back_to_numpy(monkeypatch):
    """A dead port must degrade cleanly, not raise — this is the demo safety net."""
    monkeypatch.setattr(
        actian_mod, "settings",
        replace(settings, actian_host="127.0.0.1", actian_port="6599", actian_timeout=2.0),
    )
    c = ActianClient()
    assert c.mode == "numpy"
    c.upsert("p_a", _vec("agent infra evals"), _vec("quant -> agents"), {"name": "A"})
    assert c.count() == 1
    hits = c.query(_vec("agent infra"), _vec("quant -> agents"), top_k=5)
    assert [n.id for n in hits] == ["p_a"]


def test_degrades_onto_the_mirror_when_a_live_call_fails():
    """If Actian drops mid-demo, the mirrored numpy index keeps serving matches
    and `mode` starts telling the truth."""

    class Exploding:
        mode = "actian"

        def __init__(self):
            self.mirror = _NumpyBackend()

        def upsert(self, rec):
            self.mirror.upsert(rec)          # mirror stays warm, then the DB blows up
            raise ConnectionError("actian went away")

        def query(self, *a, **kw):
            raise ConnectionError("actian went away")

        def count(self):
            raise ConnectionError("actian went away")

    c = ActianClient.__new__(ActianClient)        # no real connection needed here
    c._backend = Exploding()
    assert c.mode == "actian"                    # nothing has failed yet

    c.upsert("p_a", _vec("agent infra"), _vec("quant -> agents"), {"name": "A"})
    assert c.mode == "numpy"                  # degraded on the failed write
    assert c.count() == 1                     # the mirror kept the record
    hits = c.query(_vec("agent infra"), _vec("quant -> agents"), top_k=5)
    assert [n.id for n in hits] == ["p_a"]


def test_point_ids_are_stable_uuids():
    import uuid

    assert _point_id("p_maya") == _point_id("p_maya")     # idempotent re-seeds
    assert _point_id("p_maya") != _point_id("p_devan")
    uuid.UUID(_point_id("p_maya"))                        # Actian only accepts int/UUID ids


def test_live_scores_map_onto_the_fallback_scale():
    """Actian Cosine is [-1,1]; embeddings.cosine is [0,1]. The Matcher's fixed
    weights only make sense if both land on the same scale."""
    a, b = _vec("agent infra evals"), _vec("agent infra evals")
    raw = float(np.dot(a, b))
    assert _score_to_unit(raw) == pytest.approx(cosine(a, b), abs=1e-6)
    assert _score_to_unit(-1.0) == 0.0 and _score_to_unit(1.0) == 1.0


# --------------------------------------------------------------------------- #
#  live round-trip (skipped when the DB is down)
# --------------------------------------------------------------------------- #
TEST_PREFIX = "kindred_pytest"


def _live_settings():
    return replace(
        settings,
        actian_host=settings.actian_host or "localhost",
        actian_port=settings.actian_port or "6574",
        actian_db=TEST_PREFIX,
        actian_timeout=5.0,
    )


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(actian_mod, "settings", _live_settings())
    c = ActianClient()
    if c.mode != "actian":
        pytest.skip("Actian VectorAI DB not reachable — see backend/.env.example")
    backend = c._backend
    yield c
    for name in (backend.domain_collection, backend.trajectory_collection):
        try:
            backend._client.collections.delete(name, strict=False)
        except Exception:
            pass
    backend.close()


def test_live_upsert_and_query_round_trip(live):
    people = {
        "p_quant": ("agent infra evals quant python", "quant trading -> agent infra"),
        "p_bio":   ("biotech protein folding wet lab", "pharma -> computational biology"),
        "p_climate": ("climate carbon grid batteries", "energy trading -> climate tech"),
    }
    for pid, (domain, traj) in people.items():
        live.upsert(pid, _vec(domain), _vec(traj), {"name": pid.upper(), "domain": domain})

    assert live.count() == 3                       # counted in the DB, not in memory

    hits = live.query(_vec("agent infra evals quant python"),
                      _vec("quant trading -> agent infra"), top_k=3)
    assert [h.id for h in hits][0] == "p_quant"    # nearest on both views
    assert hits[0].meta["name"] == "P_QUANT"       # payload survived the round trip
    # every candidate is scored on BOTH views, even ones that only ranked in one
    assert all(0.0 < h.sim_domain <= 1.0 and 0.0 < h.sim_trajectory <= 1.0 for h in hits)


def test_live_query_honours_exclude_id(live):
    for pid in ("p_a", "p_b"):
        live.upsert(pid, _vec("agent infra"), _vec("quant -> agents"), {"name": pid})
    got = live.query(_vec("agent infra"), _vec("quant -> agents"), top_k=5, exclude_id="p_a")
    assert [h.id for h in got] == ["p_b"]


def test_live_upsert_is_idempotent(live):
    for _ in range(3):
        live.upsert("p_a", _vec("agent infra"), _vec("quant -> agents"), {"name": "A"})
    assert live.count() == 1                       # uuid5 ids overwrite, never duplicate


def test_live_reopens_collections_left_closed_by_a_server_restart(live, monkeypatch):
    """A collection that outlived a server restart is listed but closed — points
    calls 404 until it is reopened. Restarting the container mid-demo must not
    knock the backend onto numpy."""
    live.upsert("p_a", _vec("agent infra"), _vec("quant -> agents"), {"name": "A"})
    backend = live._backend
    for name in (backend.domain_collection, backend.trajectory_collection):
        backend._client.vde.close_collection(name)

    monkeypatch.setattr(actian_mod, "settings", _live_settings())
    reconnected = ActianClient()                   # a fresh boot against persisted data
    try:
        hits = reconnected.query(_vec("agent infra"), _vec("quant -> agents"), top_k=5)
        assert reconnected.mode == "actian"        # did NOT have to fall back
        assert [h.id for h in hits] == ["p_a"]     # data survived, index reopened
    finally:
        getattr(reconnected._backend, "close", lambda: None)()
