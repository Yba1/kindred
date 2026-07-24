"""Contract tests — run with: ./.venv/bin/python -m pytest backend/tests -q

Assert the /profile output shape, the EXACT /graph payload contract, and the
tag-only rejection. All run on the fallback path (no Gemini / no Actian), offline.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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
    r = client.post("/graph", json={
        "context": "Ex-quant trader now building an evaluation harness for tool-using "
                   "agents. Seeking a technical cofounder.",
        "name": "You",
        "top_k": 8,
    })
    assert r.status_code == 200
    g = r.json()
    # EXACT top-level keys
    assert set(g) == {"center", "nodes", "edges", "reasons"}
    assert g["center"] == "user"
    assert g["nodes"][0]["id"] == "user"
    assert len(g["nodes"]) == 9  # user + top_k
    # EXACT node keys
    for n in g["nodes"]:
        assert set(n) == {"id", "name", "score", "x", "y"}
    # EXACT edge keys, all from center
    for e in g["edges"]:
        assert set(e) == {"source", "target", "weight"}
        assert e["source"] == "user"
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
