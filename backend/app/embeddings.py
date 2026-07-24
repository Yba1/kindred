"""Embeddings with graceful degradation.

Live: Gemini `text-embedding-004` when GEMINI_API_KEY is set.
Fallback: a deterministic hashing bag-of-words embedding — no network, stable
across runs, and cosine-similar for texts that share vocabulary. Good enough to
make the demo graph cluster sensibly with zero external deps.

Every profile gets TWO views (the "one client, two vector views" from the phase
cards): a DOMAIN vector (topic clumps) and a TRAJECTORY vector (shared arc + ask).
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from .config import settings

_TOKEN = re.compile(r"[a-z0-9']+")
_DIM = settings.embed_dim


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


# --- Gemini path (guarded import; only used when a key is present) ---------- #
@lru_cache(maxsize=1)
def _gemini_model():  # pragma: no cover - requires network + key
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    return genai


def _gemini_embed(text: str) -> np.ndarray | None:  # pragma: no cover - network
    try:
        genai = _gemini_model()
        r = genai.embed_content(model="models/text-embedding-004", content=text)
        return _normalize(np.asarray(r["embedding"], dtype=np.float32))
    except Exception:
        return None


def embed(text: str) -> np.ndarray:
    """Return a unit vector for `text` (Gemini if available, else hashed)."""
    if settings.gemini_enabled:
        v = _gemini_embed(text)
        if v is not None:
            return v
    return _hash_embed(text)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two unit vectors, clamped to [0, 1]."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    s = float(np.dot(a, b))
    # our vectors can share vocabulary or not; map [-1,1] -> [0,1] for scoring
    return max(0.0, min(1.0, (s + 1.0) / 2.0))


def backend_mode() -> str:
    return "gemini" if settings.gemini_enabled else "hash-fallback"
