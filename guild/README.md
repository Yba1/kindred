# Guild — simulated, not a real connection

`client.py`'s `push_generation()` does not call a real network endpoint. It
builds a real `VersionRecord` from the actual weights + held-out rate the
loop just computed, with a deterministic `version_id` (a hash, not random),
tagged `simulated=True`.

**What Guild is designed to do in Kindred:** version every weight vector the
evolution loop's promotion gate accepts, so a degraded fine-tune can be
rolled back to a known-good state.

**What actually exists today:** `loop/run.py` calls `push_generation()` for
every accepted generation during a real loop run — check its stdout for
`[guild:simulated]` lines. The same data also lands in `run.json`'s
`generations` array (see `loop/replay.py`), which is the real local version
history Guild would additionally push externally.

**To make it real:** set `GUILD_API_KEY`, implement `_upload_version()` in
`client.py` against Guild's actual API, and call it from `push_generation()`
instead of `_simulate()`.
