"""Embeddings with graceful degradation.

Live: Gemini (`gemini-embedding-001` by default) when GEMINI_API_KEY is set.
Fallback: a deterministic hashing bag-of-words embedding — no network, stable
across runs, and cosine-similar for texts that share vocabulary. Good enough to
make the demo graph cluster sensibly with zero external deps.

Every profile gets TWO views (the "one client, two vector views" from the phase
cards): a DOMAIN vector (topic clumps) and a TRAJECTORY vector (shared arc + ask).

Three properties keep the live path cheap and safe:

  * ONE dimensionality. Gemini vectors are requested at `settings.embed_dim` (the
    embedding model is Matryoshka-trained, so truncation is principled) and any
    surprise length is coerced. Live and fallback vectors are therefore always
    comparable — a mid-corpus fallback degrades match quality slightly instead of
    blowing up `np.dot` on mismatched shapes.
  * ONE embed per unique string, process-wide. `matcher._seeking_fit` embeds
    several strings per candidate; without the cache a single /graph would fan
    out ~90 API calls.
  * ONE batched warm-up. The seeded corpus is pre-embedded in a single batched
    request the first time the live path is used, so a cold /graph doesn't walk
    the corpus one call at a time.
"""
from __future__ import annotations

import hashlib
import re
import threading
from typing import Iterable, Optional

import numpy as np

from . import config

_TOKEN = re.compile(r"[a-z0-9']+")
_DIM = config.settings.embed_dim
_BATCH = 64            # texts per batched embed request
_CACHE_MAX = 8192      # unique strings held in the process-wide vector cache


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _stable_hash(s: str) -> int:
    """Process-independent hash (built-in hash() is salted per run)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def _hash_embed(text: str, dim: int = _DIM) -> np.ndarray:
    """Feature-hashing embedding: sum signed token hashes into `dim` buckets.

    Deterministic and dependency-free. Shared tokens -> higher cosine similarity.
    Includes unigrams and bigrams so word order carries a little signal.
    """
    vec = np.zeros(dim, dtype=np.float32)
    toks = _tokens(text)
    if not toks:
        return vec
    grams = list(toks)
    grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    for g in grams:
        h = _stable_hash(g)
        idx = h % dim
        sign = 1.0 if (h >> 32) & 1 else -1.0
        vec[idx] += sign
    return _normalize(vec)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _fit(values: Iterable[float], dim: int = _DIM) -> Optional[np.ndarray]:
    """Coerce a returned embedding to exactly `dim` unit-norm float32 components."""
    vec = np.asarray(list(values), dtype=np.float32)
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        return None
    if vec.size > dim:
        vec = vec[:dim]
    elif vec.size < dim:
        vec = np.pad(vec, (0, dim - vec.size))
    out = _normalize(vec)
    return out if float(np.linalg.norm(out)) > 0 else None


# --------------------------------------------------------------------------- #
#  Gemini path (guarded: config.gemini_call returns None on any failure)
# --------------------------------------------------------------------------- #
def _embed_config() -> dict:
    cfg: dict = {"task_type": "SEMANTIC_SIMILARITY"}
    # output_dimensionality is a Matryoshka feature of the gemini-embedding line;
    # older models (text-embedding-004) reject it, so only ask when supported.
    if "gemini-embedding" in config.settings.gemini_embed_model:
        cfg["output_dimensionality"] = _DIM
    return cfg


def _values_from(response, count: int) -> list[Optional[np.ndarray]]:
    out: list[Optional[np.ndarray]] = []
    embeddings = list(getattr(response, "embeddings", None) or [])
    for i in range(count):
        vec = None
        if i < len(embeddings):
            values = getattr(embeddings[i], "values", None)
            if values:
                vec = _fit(values)
        out.append(vec)
    return out


def _gemini_embed(text: str) -> Optional[np.ndarray]:
    """One text -> one vector, or None if Gemini is unavailable/failed."""
    resp = config.gemini_call(
        lambda client: client.models.embed_content(
            model=config.settings.gemini_embed_model, contents=text, config=_embed_config()
        ),
        kind="embed",
        op="embed_content",
    )
    if resp is None:
        return None
    try:
        return _values_from(resp, 1)[0]
    except Exception:
        return None


def _gemini_embed_many(texts: list[str]) -> dict[str, np.ndarray]:
    """Batched embed. Partial results are fine — misses fall through to `embed`."""
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(texts), _BATCH):
        chunk = texts[start : start + _BATCH]
        resp = config.gemini_call(
            lambda client, c=chunk: client.models.embed_content(
                model=config.settings.gemini_embed_model, contents=list(c), config=_embed_config()
            ),
            kind="embed",
            op="embed_content(batch)",
        )
        if resp is None:
            break  # breaker tripped or quota gone; stop pushing on it
        try:
            for text, vec in zip(chunk, _values_from(resp, len(chunk))):
                if vec is not None:
                    out[text] = vec
        except Exception:
            break
    return out


# --------------------------------------------------------------------------- #
#  Cache + one-shot corpus warm-up
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_cache: dict[str, np.ndarray] = {}
_warmed = False


def _store(text: str, vec: np.ndarray) -> np.ndarray:
    vec.flags.writeable = False          # cached vectors are shared; keep them immutable
    with _lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[text] = vec
    return vec


def _corpus_texts() -> list[str]:
    """Every string the seeded corpus will be embedded with.

    Mirrors `Profile.domain_text()` / `trajectory_text()` and `matcher._offer_text()`.
    Purely an optimisation: if those drift, a missed text is just one extra call.
    """
    from .profiler import derive_roles
    from .schemas import Profile
    from .seed import PERSONAS

    texts: list[str] = []
    for d in PERSONAS:
        roles = d.get("roles") or derive_roles(
            d.get("trajectory", ""), d.get("seeking", ""), d.get("domain", "")
        )
        p = Profile(source="seed", **{**d, "roles": roles})
        texts += [
            p.domain_text(),
            p.trajectory_text(),
            " ".join(filter(None, [p.domain, " ".join(p.roles), " ".join(p.tags)])),
            p.seeking,
        ]
    return [t for t in dict.fromkeys(texts) if t.strip()]


def _warm_corpus_once() -> None:
    """Pre-embed the seed corpus in one batched call, the first time we go live."""
    global _warmed
    with _lock:
        if _warmed:
            return
        _warmed = True                   # set first: a failure must not retry forever
    try:
        texts = [t for t in _corpus_texts() if t not in _cache]
        if not texts:
            return
        for text, vec in _gemini_embed_many(texts).items():
            _store(text, vec)
    except Exception:                    # pragma: no cover - warm-up is best-effort
        pass


def reset_cache() -> None:
    """Clear the vector cache and re-arm the warm-up (tests / config reloads)."""
    global _warmed
    with _lock:
        _cache.clear()
        _warmed = False


def embed(text: str) -> np.ndarray:
    """Return a unit vector for `text` (Gemini if available, else hashed).

    Cached per unique string for the life of the process, so repeated scoring
    passes cost nothing and the vector store stays internally consistent.
    """
    key = text or ""
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit

    if config.settings.gemini_enabled:
        _warm_corpus_once()
        with _lock:                      # the warm-up may have just filled it in
            hit = _cache.get(key)
        if hit is not None:
            return hit
        vec = _gemini_embed(key)
        if vec is not None:
            return _store(key, vec)
    return _store(key, _hash_embed(key))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two unit vectors, clamped to [0, 1]."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:               # never let a provider switch crash scoring
        return 0.0
    s = float(np.dot(a, b))
    # our vectors can share vocabulary or not; map [-1,1] -> [0,1] for scoring
    return max(0.0, min(1.0, (s + 1.0) / 2.0))


def backend_mode() -> str:
    """What /health reports — the mode actually in force right now."""
    return "gemini" if config.gemini_live() else "hash-fallback"
