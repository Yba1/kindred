# Guild — not implemented

This directory is the intended integration point for Guild (weight-version
tracking), not a working client. `client.py`'s `push_generation()` raises
`NotImplementedError` unconditionally — there is no mock/dry-run mode.

**What Guild is designed to do in Kindred:** version every weight vector the
evolution loop's promotion gate accepts, so a degraded fine-tune can be rolled
back to a known-good state without touching the rest of the pipeline.

**What actually exists today:** this file, a function signature, and the real
local equivalent of the data Guild would version — `run.json`'s `generations`
array already records `{gen, rate, weights}` for every accepted generation
(see `../loop/replay.py` and `../loop/contracts.py`'s schema comment). Guild
would be pushing that same array externally, not replacing it.

**To wire it up:** set `GUILD_API_KEY`, implement `_upload_version()` in
`client.py` against Guild's real API, and call `push_generation()` from
`../loop/run.py` after each accepted generation.
