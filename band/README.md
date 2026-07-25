# BAND — not implemented

This directory is the intended integration point for BAND (the intro-thread
sponsor), not a working client. `client.py`'s `send_intro()` raises
`NotImplementedError` unconditionally — there is no mock/dry-run mode, because
a fake "sent" response would be worse than an honest error.

**What BAND is designed to do in Kindred:** once the Matcher + Introducer agree
on a match, open a real intro thread seeded with the Introducer's generated
reasoning as the first message (not a generic notification), and report back
whether that thread turned into sustained conversation — that "landed" signal
is the next generation's training label for the Pioneer scorer.

**What actually exists today:** this file, a function signature, and the
narrated version of this exact handoff in the `/village` visualization (the
Introducer agent "acting through BAND" is dialogue, not a real API call).

**To wire it up:** set `BAND_API_KEY`, implement `_post_thread()` in
`client.py` against BAND's real API, and remove the `raise` in `send_intro()`.
