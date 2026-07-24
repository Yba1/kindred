"""Environment config, capability detection, and the shared Gemini call guard.

Everything is optional. `settings` reports which sponsors are wired so /health
and the profiler/embeddings/Actian modules can pick their live-vs-fallback path.

The Gemini helpers below are the ONE place that talks to the Google GenAI SDK
(`google-genai`, imported as `from google import genai`). They live here rather
than in profiler.py / embeddings.py because profiling, embedding and
reason-writing all draw on a single API quota, so they must share a single rate
limiter and a single circuit breaker.

Activation is key-only: export `GEMINI_API_KEY` and every guarded path goes live.
No flags, no code edits. If the key is absent, the SDK isn't installed, the key
is bad, the quota is exhausted (429) or a response is malformed, `gemini_call`
returns None and the caller uses its deterministic fallback.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Optional, TypeVar

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

log = logging.getLogger("kindred.gemini")

T = TypeVar("T")


def _env_str(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name) or default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    # Model ids are overridable but never *required* — defaults are the current
    # free-tier friendly pair.
    gemini_model: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_embed_model: str = field(
        default_factory=lambda: _env_str("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    )
    gemini_timeout_s: float = field(default_factory=lambda: _env_float("GEMINI_TIMEOUT_S", 12.0))
    # Free tier is ~10-30 req/min for generate_content; embeddings get a much
    # higher ceiling. Buckets never sleep — they degrade to the fallback instead,
    # so a request is never held hostage by the quota.
    gemini_rpm: int = field(default_factory=lambda: _env_int("GEMINI_RPM", 10))
    gemini_embed_rpm: int = field(default_factory=lambda: _env_int("GEMINI_EMBED_RPM", 90))
    gemini_cooldown_s: float = field(default_factory=lambda: _env_float("GEMINI_COOLDOWN_S", 45.0))

    actian_host: str = field(default_factory=lambda: os.getenv("ACTIAN_HOST", "").strip())
    actian_port: str = field(default_factory=lambda: os.getenv("ACTIAN_PORT", "").strip())
    actian_db: str = field(default_factory=lambda: os.getenv("ACTIAN_DB", "").strip())
    actian_user: str = field(default_factory=lambda: os.getenv("ACTIAN_USER", "").strip())
    actian_password: str = field(default_factory=lambda: os.getenv("ACTIAN_PASSWORD", "").strip())

    embed_dim: int = field(default_factory=lambda: _env_int("KINDRED_EMBED_DIM", 256))

    @property
    def gemini_enabled(self) -> bool:
        """True only when a call could actually succeed: key present AND SDK importable."""
        return bool(self.gemini_api_key) and gemini_sdk_available()

    @property
    def actian_enabled(self) -> bool:
        return bool(self.actian_host and self.actian_db)


settings = Settings()


# --------------------------------------------------------------------------- #
#  Gemini capability — guarded SDK import
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def gemini_sdk_available() -> bool:
    """Is `google-genai` installed? Cached — the answer can't change mid-process.

    The SDK is listed in requirements.txt but the app must boot without it, so
    this is a spec lookup rather than an import at module scope.
    """
    try:
        return importlib.util.find_spec("google.genai") is not None
    except Exception:  # pragma: no cover - broken/partial install
        return False


@lru_cache(maxsize=4)
def _build_client(api_key: str, timeout_ms: int):
    from google import genai  # noqa: PLC0415 - guarded, optional dependency

    return genai.Client(api_key=api_key, http_options={"timeout": timeout_ms})


def gemini_client():
    """The shared `genai.Client`, or None when Gemini is not usable."""
    if not settings.gemini_enabled:
        return None
    try:
        return _build_client(settings.gemini_api_key, int(settings.gemini_timeout_s * 1000))
    except Exception as exc:  # pragma: no cover - only on a broken SDK install
        log.warning("Gemini client construction failed, using fallbacks: %r", exc)
        return None


# --------------------------------------------------------------------------- #
#  Quota guards — one token bucket per call kind, one breaker for the whole key
# --------------------------------------------------------------------------- #
class _TokenBucket:
    """Requests-per-minute limiter that refuses instead of sleeping."""

    def __init__(self, rpm: int) -> None:
        self.capacity = float(max(1, rpm))
        self._tokens = self.capacity
        self._rate = self.capacity / 60.0
        self._at = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._at) * self._rate)
            self._at = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class _CircuitBreaker:
    """Stops calling Gemini after repeated failures; reopens after a cooldown.

    A 429 opens it immediately — hammering an exhausted free-tier quota only makes
    the ban longer, and every caller has a fallback that works.
    """

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until

    def allow(self) -> bool:
        return not self.is_open

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self, cooldown_s: Optional[float] = None) -> None:
        with self._lock:
            self._failures += 1
            if cooldown_s is not None or self._failures >= self.threshold:
                self._open_until = time.monotonic() + (
                    cooldown_s if cooldown_s is not None else settings.gemini_cooldown_s
                )


_state_lock = threading.Lock()
_breaker = _CircuitBreaker()
_buckets: dict[str, _TokenBucket] = {}
_warned: set[str] = set()


def _bucket(kind: str) -> _TokenBucket:
    with _state_lock:
        bucket = _buckets.get(kind)
        if bucket is None:
            rpm = settings.gemini_embed_rpm if kind == "embed" else settings.gemini_rpm
            bucket = _buckets[kind] = _TokenBucket(rpm)
        return bucket


def reset_gemini_state() -> None:
    """Drop breaker/bucket/client/probe state. For tests and post-config reloads."""
    global _breaker
    with _state_lock:
        _breaker = _CircuitBreaker()
        _buckets.clear()
        _warned.clear()
    for cached in (_build_client, gemini_sdk_available):
        clear = getattr(cached, "cache_clear", None)  # may be stubbed out in tests
        if clear is not None:
            clear()


def is_rate_limited(exc: BaseException) -> bool:
    """Does this exception mean 'you have used your quota'? (google.genai raises
    ClientError with .code == 429; be liberal in case a transport wraps it.)"""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    if str(getattr(exc, "status", "")).upper() == "RESOURCE_EXHAUSTED":
        return True
    text = str(exc).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "QUOTA" in text


def gemini_degraded() -> bool:
    """True while the breaker is open — i.e. we are on fallbacks right now."""
    return _breaker.is_open


def gemini_live() -> bool:
    """True when Gemini is configured, importable, and not currently tripped."""
    return settings.gemini_enabled and not _breaker.is_open


def gemini_call(fn: Callable[[Any], T], *, kind: str = "generate", op: str = "") -> Optional[T]:
    """Run `fn(client)` behind the quota guards. Returns None on ANY failure.

    This is the single choke point for Gemini: it is the only place the SDK is
    invoked, so "fail safe to the fallback" is enforced once instead of at every
    call site. Callers treat None as "use the deterministic path".
    """
    if not settings.gemini_enabled:
        return None
    if not _breaker.allow():
        return None
    if not _bucket(kind).take():
        log.debug("Gemini %s skipped: local rate limit", op or kind)
        return None
    client = gemini_client()
    if client is None:
        return None
    try:
        result = fn(client)
    except Exception as exc:
        if is_rate_limited(exc):
            _breaker.record_failure(settings.gemini_cooldown_s)
            _warn_once(f"{kind}:429", "Gemini quota hit (429) on %s; falling back for %.0fs",
                       op or kind, settings.gemini_cooldown_s)
        else:
            _breaker.record_failure()
            _warn_once(f"{kind}:err", "Gemini %s failed (%r); falling back", op or kind, exc)
        return None
    _breaker.record_success()
    return result


def _warn_once(key: str, msg: str, *args: Any) -> None:
    with _state_lock:
        first = key not in _warned
        _warned.add(key)
    (log.warning if first else log.debug)(msg, *args)
