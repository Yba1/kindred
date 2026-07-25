"""BAND intro-thread client — NOT IMPLEMENTED.

Design: once the Matcher and Introducer agree on a match above threshold,
this is where Kindred would open a real intro thread on BAND, seeding the
first message with the Introducer's generated reasoning (not a generic
"you two should meet" notice). Whether that thread turns into sustained
conversation is the "landed" signal the Evaluator trains on next generation
— this is the one hop in the loop no other sponsor covers.

Status: no BAND account/API was wired up during the hackathon window. This
module exists so the intended integration point is a real file with a real
function signature, not just a paragraph in a README — but calling it does
nothing real. `send_intro()` raises NotImplementedError; there is no dry-run
mode and no mock network call, on purpose, so a caller can't mistake this for
working code.

To actually wire it: set BAND_API_KEY, implement `_post_thread` against
BAND's real API, and have `send_intro` call it instead of raising.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class IntroThread:
    thread_id: str
    match_id: str
    opener: str


def band_enabled() -> bool:
    return bool(os.getenv("BAND_API_KEY", "").strip())


def send_intro(user_id: str, match_id: str, reasoning: list[str]) -> IntroThread:
    """Open a BAND intro thread seeded with the match reasoning.

    NOT IMPLEMENTED. Raises unconditionally — there is no fallback path,
    unlike Actian/Gemini, because a fake "sent" thread would be actively
    misleading (nothing was actually sent to anyone).
    """
    raise NotImplementedError(
        "band/client.py: no real BAND integration exists yet. "
        "Implement _post_thread() against BAND's API and wire it here."
    )


def _post_thread(user_id: str, match_id: str, opener: str) -> str:  # pragma: no cover
    """Where the real BAND API call goes. Never implemented — see module docstring."""
    raise NotImplementedError("wire the real BAND API call here")
