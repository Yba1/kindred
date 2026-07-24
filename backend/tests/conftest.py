"""Shared test setup.

Puts `backend/` on sys.path so the suite runs from any cwd, and guarantees every
test starts from a clean capability state: no Gemini key, empty vector cache,
untripped circuit breaker.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

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
