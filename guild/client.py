"""Guild weight-versioning client — NOT IMPLEMENTED.

Design: every generation the Evaluator promotes (loop/evolve.py's
promotion-gate — only accepted if held-out connection-rate improves) is a
candidate for external versioning, so a degraded fine-tune could be rolled
back to a known-good weight vector without touching anything else in the
pipeline.

Status: no Guild account was wired up during the hackathon window — this was
scoped as optional/lower-priority from the start (see ../OWNERSHIP.md,
../SPONSORS.md). The data Guild would version already exists locally: every
run of `python -m loop.run` writes exactly this history to run.json's
`generations` array (`[{gen, rate, weights}, ...]`). This module is the real
integration point for pushing that array to Guild instead of just leaving it
on disk — `push_generation()` raises NotImplementedError; there is no
dry-run/mock push, on purpose.

To actually wire it: set GUILD_API_KEY, implement `_upload_version` against
Guild's real API, and call `push_generation()` from loop/run.py after each
accepted generation instead of (or in addition to) writing run.json.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class VersionRecord:
    generation: int
    weights: list[float]
    held_out_rate: float
    version_id: str | None = None


def guild_enabled() -> bool:
    return bool(os.getenv("GUILD_API_KEY", "").strip())


def push_generation(generation: int, weights: list[float], held_out_rate: float) -> VersionRecord:
    """Push one accepted generation's weights to Guild for versioning/rollback.

    NOT IMPLEMENTED. Raises unconditionally. `loop/replay.py`'s run.json
    already records this exact data locally (see loop/contracts.py's
    run.json schema comment) — this function is where that data would
    additionally go external, not a replacement for it.
    """
    raise NotImplementedError(
        "guild/client.py: no real Guild integration exists yet. "
        "Implement _upload_version() against Guild's API and wire it here."
    )


def _upload_version(generation: int, weights: list[float], held_out_rate: float) -> str:  # pragma: no cover
    """Where the real Guild API call goes. Never implemented — see module docstring."""
    raise NotImplementedError("wire the real Guild API call here")
