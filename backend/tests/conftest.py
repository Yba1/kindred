"""Shared test setup for both sponsor paths.

Two jobs, and the ORDER matters:

1. Point the suite at a scratch Actian collection prefix. `test_api.py` drives
   the real app, which upserts every profile it builds — without this, repeated
   runs silently accumulate "Test User" rows in the collections the demo reads
   from. This must happen before `app.config` is imported, since `load_dotenv`
   never overrides an already-set env var.
2. Guarantee every test starts from a clean Gemini capability state: no key,
   empty vector cache, untripped circuit breaker.

If Actian isn't running, part 1 is a no-op and the suite runs offline against
the numpy fallback.
"""
from __future__ import annotations

import os

# MUST precede the `app.config` import below.
os.environ["ACTIAN_DB"] = "kindred_test"

import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import config, embeddings  # noqa: E402


def use_settings(monkeypatch, **overrides):
    """Swap in a modified `Settings` everywhere it is read.

    `main.py` binds `settings` at import (`from .config import settings`), so the
    module attribute has to be replaced there too or /health would report the
    old capability state.
    """
    patched = replace(config.settings, **overrides)
    monkeypatch.setattr(config, "settings", patched)
    main = sys.modules.get("app.main")
    if main is not None:
        monkeypatch.setattr(main, "settings", patched, raising=False)
    return patched


@pytest.fixture(autouse=True)
def _clean_capability_state(monkeypatch):
    """No key, no cached vectors, no breaker state — before AND after each test.

    The key is cleared on `settings` itself, not just the environment: a dev with
    a real key in .env must still get the fallback-path assertions.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    use_settings(monkeypatch, gemini_api_key="")
    config.reset_gemini_state()
    embeddings.reset_cache()
    yield
    config.reset_gemini_state()
    embeddings.reset_cache()


@pytest.fixture(scope="session", autouse=True)
def _drop_test_collections():
    """Leave no scratch collections behind after the session."""
    yield
    try:
        from actian_vectorai import VectorAIClient

        from app.config import settings

        if not settings.actian_enabled:
            return
        with VectorAIClient(
            f"{settings.actian_host}:{settings.actian_port or '6574'}",
            timeout=settings.actian_timeout,
        ) as c:
            for name in c.collections.list():
                if name.startswith("kindred_test") or name.startswith("kindred_pytest"):
                    c.collections.delete(name, strict=False)
    except Exception:
        pass
