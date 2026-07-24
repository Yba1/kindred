"""Contract tests — run with: ./.venv/bin/python -m pytest backend/tests -q

Assert the /profile output shape, the /graph payload contract (the four original
keys frozen, `features` + `meta` additive), and the tag-only rejection. All run on
the fallback path (no Gemini / no Actian), offline.
"""
from fastapi.testclient import TestClient

from app.main import app
from app.matcher import FEATURE_NAMES

client = TestClient(app)

CTX = ("Ex-quant trader now building an evaluation harness for tool-using "
       "agents. Seeking a technical cofounder.")


def graph(**over):
    body = {"context": CTX, "name": "You", "top_k": 8, **over}
    r = client.post("/graph", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def weighted_mean(features, w):
    """Independent re-implementation of `scoreEdge` in
    frontend/src/weights/rescore.js. Deliberately NOT imported from app.matcher —
    the point is to check the payload against the CONSUMER's formula."""
    acc = mass = 0.0
    for f, wi in zip(features, w):
        wi = max(0.0, wi)
        acc += wi * min(1.0, max(0.0, f))
        mass += wi
    return acc / mass if mass > 0 else 0.0


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["people"] >= 30
    assert body["actian"] in ("numpy", "actian")


def test_profile_contract():
    r = client.post("/profile", json={
        "context": "I spent six years on a derivatives desk and left to build LLM "
                   "orchestration infra. Right now I'm building a model router. "
                   "I'm looking for a technical cofounder.",
        "name": "Test User",
    })
    assert r.status_code == 200
    p = r.json()
    assert set(p) >= {"roles", "trajectory", "seeking", "tags", "embedding"}
    assert p["roles"] and isinstance(p["roles"], list)
    assert p["trajectory"] and p["seeking"]
    assert isinstance(p["embedding"], list) and len(p["embedding"]) > 0
    assert all(isinstance(x, (int, float)) for x in p["embedding"])


def test_profile_rejects_tag_only():
    assert client.post("/profile", json={"context": "agents, infra, python, llm"}).status_code == 422


def test_profile_rejects_too_thin():
    assert client.post("/profile", json={"context": "hi there"}).status_code == 422


def test_graph_exact_contract():
    g = graph()
    # Top-level keys: the four originals are frozen; `meta` is the ONE additive
    # extension. Anything else appearing here is drift.
    assert set(g) == {"center", "nodes", "edges", "reasons", "meta"}
    assert g["center"] == "user"
    assert g["nodes"][0]["id"] == "user"
    assert len(g["nodes"]) == 9  # user + top_k
    # EXACT node keys — unchanged
    for n in g["nodes"]:
        assert set(n) == {"id", "name", "score", "x", "y"}
    # Edge keys: the three originals frozen, `features` additive
    for e in g["edges"]:
        assert set(e) == {"source", "target", "weight", "features"}
    # every node still hangs off the centre by exactly one edge
    star = [e for e in g["edges"] if "user" in (e["source"], e["target"])]
    assert all(e["source"] == "user" for e in star)
    assert sorted(e["target"] for e in star) == sorted(n["id"] for n in g["nodes"][1:])
    # reasons: 1-3 strings per non-center node
    for n in g["nodes"][1:]:
        rs = g["reasons"].get(n["id"])
        assert rs and 1 <= len(rs) <= 3
        assert all(isinstance(s, str) and s for s in rs)


def test_graph_ranks_by_score_desc():
    g = client.post("/graph", json={
        "context": "Ex-investment-banker moving into fintech, building an underwriting "
                   "copilot for lenders. Looking for a technical cofounder.",
        "top_k": 10,
    }).json()
    scores = [n["score"] for n in g["nodes"][1:]]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
#  The additive features contract (A->B) — this is what the re-cluster runs on.
#  Without it `window.applyWeights()` is a silent no-op: no per-edge features
#  means there is nothing for the weight vector to multiply.
# --------------------------------------------------------------------------- #
def test_graph_meta_declares_the_feature_contract():
    meta = graph()["meta"]
    assert meta["featureNames"] == ["topic", "trajectory", "seeking", "stage"]
    assert list(FEATURE_NAMES) == meta["featureNames"]      # backend agrees with itself
    assert len(meta["weights"]) == len(meta["featureNames"])
    assert all(isinstance(w, (int, float)) and w >= 0 for w in meta["weights"])
    assert sum(meta["weights"]) > 0


def test_every_edge_carries_four_clamped_features():
    g = graph(top_k=12)
    assert g["edges"], "no edges to check"
    for e in g["edges"]:
        f = e["features"]
        assert isinstance(f, list) and len(f) == len(g["meta"]["featureNames"]) == 4
        assert all(isinstance(v, (int, float)) for v in f)
        assert all(0.0 <= v <= 1.0 for v in f), f"feature out of 0..1 on {e}"


def test_weight_is_the_weighted_mean_of_features():
    """The consistency invariant: a client that re-scores with the backend's own
    `w` must land back on the weight it was handed, or the graph jumps the moment
    anyone calls applyWeights."""
    g = graph(top_k=12)
    w = g["meta"]["weights"]
    worst = max(abs(weighted_mean(e["features"], w) - e["weight"]) for e in g["edges"])
    assert worst < 1e-3, f"weight drifts from weighted_mean(features) by {worst}"


def test_graph_has_inter_person_edges():
    """A star has nothing to re-cluster — every node hangs off the same hub no
    matter what `w` says. The re-cluster is only visible with person<->person ties."""
    g = graph(top_k=12)
    inter = [e for e in g["edges"] if "user" not in (e["source"], e["target"])]
    assert len(inter) >= 10, f"only {len(inter)} inter-person edges"
    assert len(g["edges"]) <= 400, "edge budget blown"
    ids = {n["id"] for n in g["nodes"]}
    for e in inter:
        assert e["source"] in ids and e["target"] in ids and e["source"] != e["target"]


def test_reweighting_actually_reshuffles_the_graph():
    """Feature vectors that all say the same thing would render the animation
    inert. Two very different `w` must produce two very different rankings."""
    g = graph(top_k=12)
    edges = g["edges"]
    topic = sorted(range(len(edges)), key=lambda i: -weighted_mean(edges[i]["features"], [0.7, 0.1, 0.1, 0.1]))
    traj = sorted(range(len(edges)), key=lambda i: -weighted_mean(edges[i]["features"], [0.1, 0.6, 0.25, 0.05]))
    n = max(5, len(edges) // 4)
    overlap = len(set(topic[:n]) & set(traj[:n])) / n
    assert overlap < 0.7, f"top-{n} barely moves ({overlap:.0%} overlap) — features carry no signal"


def test_each_feature_dim_carries_its_own_signal():
    """A dim that is constant, or a clone of its neighbour, is dead weight in the
    weight vector — the Evaluator would have nothing to learn on it."""
    g = graph(top_k=12)
    cols = list(zip(*[e["features"] for e in g["edges"]]))
    for i, name in enumerate(g["meta"]["featureNames"]):
        assert len(set(cols[i])) > 1, f"feature '{name}' is constant across every edge"
    basis = [[1 if j == i else 0 for j in range(4)] for i in range(4)]
    rankings = [
        sorted(range(len(g["edges"])), key=lambda k: -weighted_mean(g["edges"][k]["features"], b))
        for b in basis
    ]
    n = max(5, len(g["edges"]) // 4)
    for i in range(4):
        for j in range(i + 1, 4):
            shared = len(set(rankings[i][:n]) & set(rankings[j][:n])) / n
            assert shared < 0.9, (
                f"'{FEATURE_NAMES[i]}' and '{FEATURE_NAMES[j]}' rank edges near-identically"
            )


# --------------------------------------------------------------------------- #
#  The scoring primitives, unit-level
# --------------------------------------------------------------------------- #
def test_weighted_mean_matches_the_frontend_formula():
    from app.matcher import weighted_mean as backend_wm

    f = [0.2, 0.9, 0.4, 0.75]
    for w in ([0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.25, 0.05], [0.34, 0.36, 0.3, 0.0], [3, 1, 1, 1]):
        assert abs(backend_wm(f, w) - weighted_mean(f, w)) < 1e-12
    assert backend_wm(f, [0, 0, 0, 0]) == 0.0          # zero mass -> 0, like rescore.js
    assert 0.0 <= backend_wm([5, -3, 0.5, 0.5]) <= 1.0  # garbage in still clamps


def test_features_are_symmetric_and_bounded():
    from app.matcher import pair_features, stage_affinity, stage_position
    from app.store import store

    a, b = store.all()[0], store.all()[5]
    assert pair_features(a, b) == pair_features(b, a)
    assert all(0.0 <= v <= 1.0 for v in pair_features(a, b))
    assert stage_affinity(a, a) == 1.0
    for p in store.all():
        assert 0.0 <= stage_position(p) <= 1.0
    # the ask a person leads with places them on the arc: cofounder < first
    # engineer < growth lead
    positions = {p.seeking: stage_position(p) for p in store.all()}
    assert positions["technical cofounder"] < positions["first engineer"]
    assert positions["first engineer"] < positions["growth lead"]


def test_features_survive_a_profile_with_nothing_in_it():
    from app.matcher import pair_features
    from app.schemas import Profile

    empty = Profile(id="x", name="X")
    out = pair_features(empty, empty)
    assert len(out) == 4 and all(0.0 <= v <= 1.0 for v in out)
