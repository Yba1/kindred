# BAND — simulated, not a real connection

`client.py`'s `send_intro()` does not call a real network endpoint. It builds
a real, deterministic `IntroThread` from the actual match reasoning you pass
in — same shape a live integration would return — and tags the result
`simulated=True`. There is no hidden network call and nothing fabricated
about the content; only the delivery isn't real.

**What BAND is designed to do in Kindred:** once the Matcher + Introducer
agree on a match, open a real intro thread seeded with the Introducer's
generated reasoning as the first message, and report back whether that
thread turned into sustained conversation — that "landed" signal is the next
generation's training label for the Pioneer scorer.

**What actually exists today:** a real function you can call right now
(`send_intro(user_id, match_id, reasoning)`), returning a real opener built
from real reasoning — plus the narrated version of this exact handoff in the
`/village` visualization.

**To make it real:** set `BAND_API_KEY`, implement `_post_thread()` in
`client.py` against BAND's actual API, and call it from `send_intro()`
instead of `_simulate()`.
