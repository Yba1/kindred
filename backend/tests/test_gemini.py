"""Gemini wiring tests — run with: cd backend && ./.venv/bin/python -m pytest tests -q

No live API key exists, so every test here MOCKS the `google-genai` client and
asserts on the calls we make. What they prove:

  * with no key, nothing ever reaches the SDK and /health tells the truth;
  * with a key, profiling, embedding and reason-writing all call through;
  * every failure mode (exception, 429, malformed JSON, missing SDK) degrades to
    the deterministic path instead of erroring the request;
  * a 429 trips the shared breaker, so we stop hammering an exhausted quota;
  * we never fan out one API call per node.

The live path itself is UNVERIFIED against the real API — no key exists.
"""
from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, embeddings, profiler
from app.actian import Neighbour
from app.main import app
from app.matcher import Match, rank
from app.schemas import Profile
from conftest import use_settings

client = TestClient(app)

CONTEXT = (
    "I spent six years on a derivatives desk and left to build LLM orchestration "
    "infra. Right now I'm building a model router. I'm looking for a technical cofounder."
)


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #
def text_response(payload) -> SimpleNamespace:
    return SimpleNamespace(text=payload if isinstance(payload, str) else json.dumps(payload))


def embed_response(vectors) -> SimpleNamespace:
    return SimpleNamespace(embeddings=[SimpleNamespace(values=list(v)) for v in vectors])


def as_list(contents) -> list[str]:
    return list(contents) if isinstance(contents, list) else [contents]


def fake_vector(text: str, dim: int | None = None) -> list[float]:
    """Deterministic unit vector that is clearly NOT the hash fallback."""
    dim = dim or embeddings._DIM
    rng = np.random.default_rng(embeddings._stable_hash("gemini:" + text) % (2**32))
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def rate_limit_error() -> Exception:
    """A real google-genai 429 when the SDK is installed, else a look-alike."""
    try:
        from google.genai import errors

        return errors.ClientError(
            429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}}
        )
    except Exception:  # pragma: no cover - only when the SDK is absent

        class _RateLimited(Exception):
            code = 429

        return _RateLimited("429 RESOURCE_EXHAUSTED")


class _FakeModels:
    def __init__(self, owner: "FakeGemini") -> None:
        self._owner = owner

    def generate_content(self, *, model, contents, config=None):
        self._owner.generate_calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return self._owner._generate(contents)

    def embed_content(self, *, model, contents, config=None):
        self._owner.embed_calls.append({"model": model, "contents": contents, "config": config})
        return self._owner._embed(contents)


class FakeGemini:
    """Stand-in for `genai.Client`: records every call, plays a programmed answer."""

    def __init__(self) -> None:
        self.generate_calls: list[dict] = []
        self.embed_calls: list[dict] = []
        self._generate = lambda contents: text_response({})
        self._embed = lambda contents: embed_response([fake_vector(t) for t in as_list(contents)])
        self.models = _FakeModels(self)

    # -- programming helpers ------------------------------------------------ #
    def generates(self, payload) -> "FakeGemini":
        self._generate = lambda contents: text_response(payload)
        return self

    def generate_raises(self, exc: Exception) -> "FakeGemini":
        def _boom(contents):
            raise exc

        self._generate = _boom
        return self

    def embeds(self, fn) -> "FakeGemini":
        self._embed = lambda contents: embed_response([fn(t) for t in as_list(contents)])
        return self

    def embed_raises(self, exc: Exception) -> "FakeGemini":
        def _boom(contents):
            raise exc

        self._embed = _boom
        return self

    # -- call shape --------------------------------------------------------- #
    @property
    def embed_batches(self) -> list[dict]:
        return [c for c in self.embed_calls if isinstance(c["contents"], list)]

    @property
    def embed_singles(self) -> list[dict]:
        return [c for c in self.embed_calls if not isinstance(c["contents"], list)]


@pytest.fixture
def gemini(monkeypatch) -> FakeGemini:
    """A configured key + a mocked SDK client. This is the 'live' path."""
    fake = FakeGemini()
    use_settings(monkeypatch, gemini_api_key="test-key")
    monkeypatch.setattr(config, "gemini_sdk_available", lambda: True)
    monkeypatch.setattr(config, "gemini_client", lambda: fake)
    config.reset_gemini_state()
    embeddings.reset_cache()
    return fake


def a_profile(**over) -> Profile:
    base = dict(
        id="u1", name="You", roles=["ex-quant", "founder"],
        trajectory="quant trading -> agent infra", seeking="technical cofounder",
        tags=["agents", "evals"], domain="agent infra",
        summary="Ex-quant building an eval harness.",
    )
    base.update(over)
    return Profile(**base)


def some_matches(n: int = 3) -> tuple[list[Match], dict[str, dict]]:
    matches, metas = [], {}
    for i in range(n):
        mid = f"p_{i}"
        matches.append(
            Match(id=mid, name=f"Person {i}", score=0.9 - i * 0.1,
                  reasons=["shared domain: agent infra"], components={})
        )
        metas[mid] = {
            "name": f"Person {i}", "domain": "agent infra", "roles": ["founder"],
            "trajectory": "trading -> agent infra", "seeking": "technical cofounder",
            "tags": ["agents"], "summary": f"Person {i} summary",
        }
    return matches, metas


# --------------------------------------------------------------------------- #
#  1. No key: the default path, untouched
# --------------------------------------------------------------------------- #
def test_health_reports_fallback_modes_without_key():
    body = client.get("/health").json()
    assert body["profiler"] == "heuristic"
    assert body["embeddings"] == "hash-fallback"
    assert body["gemini"] is False


def test_no_key_never_constructs_a_client(monkeypatch):
    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("Gemini must not be touched without a key")

    monkeypatch.setattr(config, "_build_client", explode)
    monkeypatch.setattr(config, "gemini_client", explode)

    assert client.post("/profile", json={"context": CONTEXT}).status_code == 200
    g = client.post("/graph", json={"context": CONTEXT, "top_k": 5})
    assert g.status_code == 200
    assert len(g.json()["nodes"]) == 6


def test_embed_without_key_is_the_hash_fallback():
    assert np.allclose(embeddings.embed("agent infra"), embeddings._hash_embed("agent infra"))
    assert embeddings.backend_mode() == "hash-fallback"
    assert profiler.profiler_mode() == "heuristic"


def test_reasons_are_a_noop_without_key(monkeypatch):
    monkeypatch.setattr(config, "gemini_client", lambda: pytest.fail("no key -> no call"))
    matches, metas = some_matches()
    profiler.apply_gemini_reasons(a_profile(), matches, metas)
    assert matches[0].reasons == ["shared domain: agent infra"]


# --------------------------------------------------------------------------- #
#  2. Profiling goes through Gemini when a key is present
# --------------------------------------------------------------------------- #
def test_profile_calls_gemini_and_uses_its_answer(gemini):
    gemini.generates({
        "name": "Ada", "roles": ["ex-quant", "founder"],
        "trajectory": "derivatives desk -> agent infra", "seeking": "a technical cofounder",
        "domain": "agent infra", "tags": ["routing", "llm"], "summary": "Ex-quant, now infra.",
    })

    p = profiler.build_profile(CONTEXT)

    assert len(gemini.generate_calls) == 1, "profiling must call Gemini exactly once"
    call = gemini.generate_calls[0]
    assert call["model"] == config.settings.gemini_model
    assert CONTEXT in call["contents"], "the intake context must reach the model"
    assert call["config"]["response_mime_type"] == "application/json"
    assert p.source == "gemini"
    assert p.name == "Ada"
    assert p.trajectory == "derivatives desk -> agent infra"
    assert p.seeking == "a technical cofounder"
    assert p.tags == ["routing", "llm"]
    assert profiler.profiler_mode() == "gemini"


def test_profile_route_serves_the_gemini_answer(gemini):
    gemini.generates({
        "roles": ["ex-quant"], "trajectory": "desk -> agents", "seeking": "a cofounder",
        "domain": "agent infra", "tags": ["llm"], "summary": "s",
    })
    body = client.post("/profile", json={"context": CONTEXT, "name": "Ada"}).json()
    assert body["trajectory"] == "desk -> agents"
    assert body["seeking"] == "a cofounder"
    assert gemini.generate_calls, "the route must exercise the live profiler"


def test_profile_gemini_gaps_are_filled_by_the_heuristic(gemini):
    """A thin answer must not produce a hollow profile — /profile's contract says
    trajectory and seeking are always populated."""
    gemini.generates({"domain": "agent infra"})
    p = profiler.build_profile(CONTEXT)
    assert p.source == "gemini"
    assert p.domain == "agent infra"
    assert p.trajectory and p.seeking, "missing fields must come from the heuristic"
    assert p.roles


def test_profile_falls_back_on_exception(gemini):
    gemini.generate_raises(RuntimeError("connection reset"))
    p = profiler.build_profile(CONTEXT)
    assert p.source == "heuristic"
    assert p.trajectory and p.seeking
    assert len(gemini.generate_calls) == 1


def test_profile_falls_back_on_malformed_response(gemini):
    gemini.generates("I'm afraid I can't do that, Dave.")
    p = profiler.build_profile(CONTEXT)
    assert p.source == "heuristic"
    assert p.seeking


def test_profile_survives_a_response_with_no_text(gemini):
    gemini._generate = lambda contents: SimpleNamespace(text=None)
    assert profiler.build_profile(CONTEXT).source == "heuristic"


def test_profile_falls_back_on_429_and_stops_calling(gemini):
    gemini.generate_raises(rate_limit_error())

    first = profiler.build_profile(CONTEXT)
    assert first.source == "heuristic"
    assert len(gemini.generate_calls) == 1

    second = profiler.build_profile(CONTEXT)
    assert second.source == "heuristic"
    assert len(gemini.generate_calls) == 1, "a 429 must open the breaker, not retry"
    assert config.gemini_degraded() is True
    assert profiler.profiler_mode() == "heuristic", "/health must admit we are on fallbacks"


def test_route_still_serves_when_gemini_is_down(gemini):
    gemini.generate_raises(RuntimeError("boom"))
    gemini.embed_raises(RuntimeError("boom"))
    r = client.post("/graph", json={"context": CONTEXT, "top_k": 6})
    assert r.status_code == 200
    assert len(r.json()["nodes"]) == 7


# --------------------------------------------------------------------------- #
#  3. Embeddings go through Gemini when a key is present
# --------------------------------------------------------------------------- #
def test_embed_calls_gemini_and_caches_the_result(gemini):
    vec = embeddings.embed("agent infra orchestration")

    assert gemini.embed_calls, "embedding must call Gemini"
    assert gemini.embed_calls[0]["model"] == config.settings.gemini_embed_model
    assert gemini.embed_calls[0]["config"]["output_dimensionality"] == embeddings._DIM
    assert vec.shape == (embeddings._DIM,)
    assert not np.allclose(vec, embeddings._hash_embed("agent infra orchestration"))
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
    assert embeddings.backend_mode() == "gemini"

    before = len(gemini.embed_calls)
    for _ in range(5):
        embeddings.embed("agent infra orchestration")
    assert len(gemini.embed_calls) == before, "repeat embeds must be served from cache"


def test_embed_warms_the_corpus_in_batches_not_per_node(gemini):
    embeddings.embed("something new")

    assert gemini.embed_batches, "the seed corpus must be embedded in batched calls"
    warmed = sum(len(c["contents"]) for c in gemini.embed_batches)
    assert warmed > 40, f"expected the corpus warm-up, only got {warmed} texts"
    assert len(gemini.embed_batches) <= 3

    before = len(gemini.embed_calls)
    for persona_text in ("agent infra agents evals quant python ex-quant trading",
                         "technical cofounder", "design partner"):
        embeddings.embed(persona_text)
    assert len(gemini.embed_calls) - before <= 3, "warmed texts must not re-hit the API"


def test_embed_falls_back_to_hash_on_exception(gemini):
    gemini.embed_raises(RuntimeError("connection reset"))
    assert np.allclose(embeddings.embed("agent infra"), embeddings._hash_embed("agent infra"))


def test_embed_falls_back_on_429_and_stops_calling(gemini):
    gemini.embed_raises(rate_limit_error())

    assert np.allclose(embeddings.embed("agent infra"), embeddings._hash_embed("agent infra"))
    calls = len(gemini.embed_calls)
    embeddings.embed("a different string")
    assert len(gemini.embed_calls) == calls, "a 429 must open the breaker, not retry"
    assert config.gemini_degraded() is True
    assert embeddings.backend_mode() == "hash-fallback"


def test_embed_coerces_a_surprising_dimension(gemini):
    gemini.embeds(lambda t: fake_vector(t, dim=768))
    vec = embeddings.embed("agent infra")
    assert vec.shape == (embeddings._DIM,), "live and fallback vectors must stay comparable"
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_embed_falls_back_on_empty_or_broken_vectors(gemini):
    gemini.embeds(lambda t: [])
    assert np.allclose(embeddings.embed("agent infra"), embeddings._hash_embed("agent infra"))

    embeddings.reset_cache()
    config.reset_gemini_state()
    gemini.embeds(lambda t: [float("nan")] * embeddings._DIM)
    assert np.allclose(embeddings.embed("agent infra"), embeddings._hash_embed("agent infra"))


def test_legacy_embed_model_skips_the_dimension_flag(gemini, monkeypatch):
    use_settings(monkeypatch, gemini_api_key="test-key",
                 gemini_embed_model="text-embedding-004")
    embeddings.embed("agent infra")
    assert "output_dimensionality" not in gemini.embed_calls[0]["config"]


def test_older_chat_model_skips_the_thinking_flag(gemini, monkeypatch):
    """An overridden GEMINI_MODEL must not 400 on a knob it doesn't have."""
    assert "thinking_config" in profiler._generate_config(600)  # default is 2.5
    use_settings(monkeypatch, gemini_api_key="test-key", gemini_model="gemini-1.5-flash")
    assert "thinking_config" not in profiler._generate_config(600)


def test_cosine_survives_mismatched_dimensions():
    assert embeddings.cosine(np.ones(256, dtype=np.float32), np.ones(768, dtype=np.float32)) == 0.0


# --------------------------------------------------------------------------- #
#  4. Reasons
# --------------------------------------------------------------------------- #
def test_reasons_use_one_gemini_call_for_the_whole_batch(gemini):
    matches, metas = some_matches(12)
    gemini.generates({m.id: [f"you and {m.name} both left trading for agents"] for m in matches})

    profiler.apply_gemini_reasons(a_profile(), matches, metas)

    assert len(gemini.generate_calls) == 1, "one call for the batch, never one per node"
    prompt = gemini.generate_calls[0]["contents"]
    assert "technical cofounder" in prompt and "p_0" in prompt
    for m in matches:
        assert m.reasons == [f"you and {m.name} both left trading for agents"]


def test_reasons_are_cleaned_and_capped(gemini):
    matches, metas = some_matches(1)
    gemini.generates({
        "p_0": ["  - first reason.  ", "", "second", "third", "fourth (dropped)"],
        "p_unknown": ["ignored"],
    })
    profiler.apply_gemini_reasons(a_profile(), matches, metas)
    assert matches[0].reasons == ["first reason", "second", "third"]


def test_reasons_keep_templates_for_ids_gemini_skipped(gemini):
    matches, metas = some_matches(3)
    gemini.generates({"p_1": ["Gemini wrote this one"]})
    profiler.apply_gemini_reasons(a_profile(), matches, metas)
    assert matches[0].reasons == ["shared domain: agent infra"]
    assert matches[1].reasons == ["Gemini wrote this one"]
    assert matches[2].reasons == ["shared domain: agent infra"]


@pytest.mark.parametrize("failure", ["exception", "429", "garbage"])
def test_reasons_keep_templates_on_failure(gemini, failure):
    matches, metas = some_matches(3)
    if failure == "exception":
        gemini.generate_raises(RuntimeError("connection reset"))
    elif failure == "429":
        gemini.generate_raises(rate_limit_error())
    else:
        gemini.generates("sorry, no JSON here")

    profiler.apply_gemini_reasons(a_profile(), matches, metas)

    assert all(m.reasons == ["shared domain: agent infra"] for m in matches)
    if failure == "429":
        assert config.gemini_degraded() is True


def test_reasons_prompt_survives_braces_in_the_data(gemini):
    """The candidate payload is JSON, so the prompt must not go through str.format."""
    matches, metas = some_matches(1)
    metas["p_0"]["summary"] = "wrote {a template} with {braces}"
    gemini.generates({"p_0": ["ok"]})
    profiler.apply_gemini_reasons(a_profile(summary="I use {curly} braces"), matches, metas)
    assert matches[0].reasons == ["ok"]


def test_apply_gemini_reasons_fits_the_real_matcher_output(gemini):
    """The adapter takes exactly what `matcher.rank` has on hand: its Match list
    plus the Neighbour metas. Wiring it is one line at the end of rank()."""
    user = a_profile()
    neighbours = [
        Neighbour(id="p_maya", sim_domain=0.9, sim_trajectory=0.9,
                  meta={"name": "Maya Okafor", "domain": "agent infra",
                        "trajectory": "quant trading -> agent infra",
                        "seeking": "technical cofounder", "tags": ["agents"], "roles": ["founder"]}),
        Neighbour(id="p_lena", sim_domain=0.5, sim_trajectory=0.4,
                  meta={"name": "Lena Fischer", "domain": "fintech",
                        "trajectory": "investment banking -> fintech",
                        "seeking": "technical cofounder", "tags": ["credit"], "roles": ["founder"]}),
    ]
    matches = rank(user, neighbours, limit=2)
    templates = {m.id: list(m.reasons) for m in matches}
    gemini.generates({"p_maya": ["Maya is hiring the cofounder you want to be"]})

    profiler.apply_gemini_reasons(user, matches, {n.id: n.meta for n in neighbours})

    by_id = {m.id: m.reasons for m in matches}
    assert by_id["p_maya"] == ["Maya is hiring the cofounder you want to be"]
    assert by_id["p_lena"] == templates["p_lena"], "unanswered ids keep their template"


# --------------------------------------------------------------------------- #
#  5. Capability reporting + quota discipline
# --------------------------------------------------------------------------- #
def test_health_reports_gemini_when_live(gemini):
    body = client.get("/health").json()
    assert body["profiler"] == "gemini"
    assert body["embeddings"] == "gemini"
    assert body["gemini"] is True


def test_key_without_the_sdk_stays_on_fallbacks(monkeypatch):
    """The package is optional: a key alone must not enable a path we can't run."""
    use_settings(monkeypatch, gemini_api_key="test-key")
    monkeypatch.setattr(config, "gemini_sdk_available", lambda: False)
    monkeypatch.setattr(config, "_build_client", lambda *a, **k: pytest.fail("no SDK"))

    assert config.settings.gemini_enabled is False
    assert profiler.build_profile(CONTEXT).source == "heuristic"
    assert np.allclose(embeddings.embed("agent infra"), embeddings._hash_embed("agent infra"))
    body = client.get("/health").json()
    assert (body["profiler"], body["embeddings"], body["gemini"]) == \
        ("heuristic", "hash-fallback", False)


def test_sdk_probe_reports_false_when_the_package_is_missing(monkeypatch):
    config.gemini_sdk_available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert config.gemini_sdk_available() is False
    config.gemini_sdk_available.cache_clear()


def test_local_rate_limit_degrades_instead_of_calling(gemini, monkeypatch):
    use_settings(monkeypatch, gemini_api_key="test-key", gemini_rpm=2)
    config.reset_gemini_state()
    gemini.generates({"trajectory": "a -> b", "seeking": "x"})

    sources = [profiler.build_profile(CONTEXT).source for _ in range(4)]

    assert sources[:2] == ["gemini", "gemini"]
    assert sources[2:] == ["heuristic", "heuristic"], "over-budget calls fall back"
    assert len(gemini.generate_calls) == 2, "we must not exceed our own RPM budget"


def test_graph_contract_holds_on_the_live_path_without_fanning_out(gemini):
    gemini.generates({
        "roles": ["ex-quant", "founder"], "trajectory": "derivatives -> agent infra",
        "seeking": "a technical cofounder", "domain": "agent infra",
        "tags": ["routing"], "summary": "Ex-quant building a router.",
    })

    g = client.post("/graph", json={"context": CONTEXT, "name": "You", "top_k": 8}).json()

    assert set(g) == {"center", "nodes", "edges", "reasons"}
    assert len(g["nodes"]) == 9
    for node in g["nodes"][1:]:
        assert 1 <= len(g["reasons"][node["id"]]) <= 3
    assert len(gemini.generate_calls) == 1, "one profile call per request"
    assert len(gemini.embed_singles) <= 8, (
        f"one /graph must not fan out per node "
        f"(got {len(gemini.embed_singles)} single embeds)"
    )
