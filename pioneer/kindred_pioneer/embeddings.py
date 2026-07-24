"""Profile embeddings for the cosine baseline.

Default is a deterministic hashed bag-of-ngrams — no network, no model download,
identical numbers on every machine, which is what a demo needs.

To score against the real vector space instead, point `KINDRED_EMBED_BACKEND`
at a provider and implement it in `_embed_remote`. Everything downstream
(baseline, features, eval harness) reads `embed_text` and needs no changes.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache

import numpy as np

EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _bucket(token: str) -> int:
    """Stable across processes — Python's built-in hash() is salted per run."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % EMBED_DIM


@lru_cache(maxsize=4096)
def embed_text(text: str) -> tuple[float, ...]:
    """Sublinear-tf hashed embedding, L2-normalised. Cached and deterministic."""
    counts: dict[int, float] = {}
    for tok in _tokens(text):
        idx = _bucket(tok)
        counts[idx] = counts.get(idx, 0.0) + 1.0

    vec = np.zeros(EMBED_DIM, dtype=np.float64)
    for idx, count in counts.items():
        vec[idx] = 1.0 + math.log(count)

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return tuple(vec.tolist())


def embed(text: str) -> np.ndarray:
    return np.asarray(embed_text(text), dtype=np.float64)


def cosine(text_a: str, text_b: str) -> float:
    """Cosine similarity of two profile blurbs, clipped to [-1, 1]."""
    return float(np.clip(np.dot(embed(text_a), embed(text_b)), -1.0, 1.0))


def backend_name() -> str:
    return os.environ.get("KINDRED_EMBED_BACKEND", "hashed-local")
