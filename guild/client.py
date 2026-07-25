"""Guild weight-versioning client — SIMULATED, not a real API connection.

Design: every generation the Evaluator promotes (loop/evolve.py's
promotion-gate — only accepted if held-out connection-rate improves) is a
candidate for external versioning, so a degraded fine-tune could be rolled
back to a known-good weight vector.

Status: no Guild account was wired up during the hackathon window, so
`push_generation()` does NOT call a real network endpoint. It builds a real
VersionRecord from the actual weights + rate you pass in (the same data
loop/replay.py already writes into run.json's `generations` array) and
tags every result `simulated=True`. No fabricated external state, no
hidden network call.

To make this real: set GUILD_API_KEY, implement `_upload_version` against
Guild's actual API, and call `push_generation()` from loop/run.py after each
accepted generation instead of `_simulate`.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


@dataclass
class VersionRecord:
    generation: int
    weights: list[float]
    held_out_rate: float
    version_id: str
    simulated: bool = True


def guild_enabled() -> bool:
    """True only once a real client is implemented AND a key is set. Today
    this is always False — push_generation() always simulates, key or not."""
    return False


def push_generation(generation: int, weights: list[float], held_out_rate: float) -> VersionRecord:
    """Build the version record Guild would store, without a real network call.

    Deterministic given the same inputs (version_id is a hash of the real
    weights, not random), so re-running the loop reproduces identical
    version ids — same property run.json's replay already guarantees.
    """
    return _simulate(generation, weights, held_out_rate)


def _simulate(generation: int, weights: list[float], held_out_rate: float) -> VersionRecord:
    payload = f"{generation}:{weights}:{held_out_rate}".encode()
    version_id = "sim_v" + hashlib.blake2b(payload, digest_size=6).hexdigest()
    return VersionRecord(
        generation=generation,
        weights=list(weights),
        held_out_rate=held_out_rate,
        version_id=version_id,
        simulated=True,
    )


def _upload_version(generation: int, weights: list[float], held_out_rate: float) -> str:  # pragma: no cover
    """Where the real Guild API call goes. Never implemented — see module docstring."""
    raise NotImplementedError("wire the real Guild API call here; nothing calls this yet")
