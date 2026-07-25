"""BAND intro-thread client — SIMULATED, not a real API connection.

Design: once the Matcher and Introducer agree on a match above threshold,
this is where Kindred opens a real intro thread on BAND, seeding the first
message with the Introducer's generated reasoning (not a generic "you two
should meet" notice). Whether that thread turns into sustained conversation
is the "landed" signal the Evaluator trains on next generation.

Status: no BAND account/API was wired up during the hackathon window, so
`send_intro()` does NOT call a real network endpoint. It builds a real,
deterministic IntroThread object from the actual reasoning you pass in —
same shape a live integration would return — and every result is clearly
tagged `simulated=True` so nothing downstream can mistake this for a real
send. There is no hidden network call and no fabricated external state.

To make this real: set BAND_API_KEY, implement `_post_thread` against
BAND's actual API, and have `send_intro` call it instead of `_simulate`.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


@dataclass
class IntroThread:
    thread_id: str
    match_id: str
    opener: str
    simulated: bool = True


def band_enabled() -> bool:
    """True only once a real client is implemented AND a key is set. Today
    this is always False — send_intro() always simulates, key or not."""
    return False


def send_intro(user_id: str, match_id: str, reasoning: list[str]) -> IntroThread:
    """Build the intro thread BAND would open, without a real network call.

    Deterministic given the same inputs (thread_id is a hash, not random) so
    a demo replay produces identical output. Real reasoning in, real opener
    text out — nothing about the CONTENT is fake, only the delivery.
    """
    return _simulate(user_id, match_id, reasoning)


def _simulate(user_id: str, match_id: str, reasoning: list[str]) -> IntroThread:
    thread_id = "sim_" + hashlib.blake2b(
        f"{user_id}:{match_id}".encode(), digest_size=6
    ).hexdigest()
    lead = reasoning[0] if reasoning else "you two matched"
    opener = f"Hey — connecting you two because {lead}. Worth a conversation?"
    return IntroThread(thread_id=thread_id, match_id=match_id, opener=opener, simulated=True)


def _post_thread(user_id: str, match_id: str, opener: str) -> str:  # pragma: no cover
    """Where the real BAND API call goes. Never implemented — see module docstring."""
    raise NotImplementedError("wire the real BAND API call here; nothing calls this yet")
