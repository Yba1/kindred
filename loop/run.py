"""CLI entrypoint tying the Kindred evolution-loop package together.

Usage:
    python -m loop.run                      # generate + evolve + narrate + write run.json
    python -m loop.run --replay run.json    # load an existing run.json and replay its events

See loop/contracts.py for the shared dataclasses and the run.json schema.
"""
from __future__ import annotations

import argparse
import sys
import time


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m loop.run",
        description="Run (or replay) the Kindred persona/pair evolution loop.",
    )
    parser.add_argument(
        "--replay", metavar="PATH", default=None,
        help="path to an existing run.json to load and replay instead of regenerating",
    )
    parser.add_argument("--n-personas", type=int, default=300, help="number of personas (default: 300)")
    parser.add_argument("--n-pairs", type=int, default=250, help="number of pairs (default: 250)")
    parser.add_argument("--generations", type=int, default=6, help="number of evolve generations (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="base seed (default: 42)")
    parser.add_argument(
        "--interval", type=float, default=1.5,
        help="seconds to sleep between replayed events (default: 1.5, --replay only)",
    )
    return parser.parse_args(argv)


def _run_live(args: argparse.Namespace) -> int:
    try:
        from loop.personas import generate_personas
    except ImportError:
        print("loop.personas not ready yet - run again once it's built")
        return 1

    try:
        from loop.pairs import generate_pairs
    except ImportError:
        print("loop.pairs not ready yet - run again once it's built")
        return 1

    try:
        from loop.evolve import evolve
    except ImportError:
        print("loop.evolve not ready yet - run again once it's built")
        return 1

    try:
        from loop.narrate import narrate
    except ImportError:
        print("loop.narrate not ready yet - run again once it's built")
        return 1

    try:
        from loop.replay import write_run, summarize
    except ImportError:
        print("loop.replay not ready yet - run again once it's built")
        return 1

    personas = generate_personas(n=args.n_personas, seed=args.seed)
    pairs = generate_pairs(personas, n=args.n_pairs, seed=args.seed + 1)
    result = evolve(personas, pairs, generations=args.generations, seed=args.seed + 2)
    events = narrate(result)
    run = write_run(result, events, path="run.json")

    # Version every accepted generation. Simulated (see guild/README.md) —
    # real data in, no real network call out.
    try:
        from guild.client import push_generation
        for g in result.generations:
            record = push_generation(g.gen, list(g.weights), g.rate)
            print(f"[guild:simulated] gen {record.generation} -> {record.version_id} (rate {record.held_out_rate:.0%})")
    except ImportError:
        pass  # guild/ not on the path — versioning is optional, never blocks the loop

    print(summarize(run))

    gens = run.get("generations", [])
    if gens:
        first, last = gens[0], gens[-1]
        print(f"Gen {first['gen']}: {first['rate']:.0%} -> Gen {last['gen']}: {last['rate']:.0%}")

    return 0


def _run_replay(path: str, interval: float) -> int:
    try:
        from loop.replay import load_run, summarize
    except ImportError:
        print("loop.replay not ready yet - run again once it's built")
        return 1

    try:
        run = load_run(path)
    except (OSError, ValueError) as exc:
        print(f"could not load {path!r}: {exc}")
        return 1

    print(summarize(run))

    for event in run.get("events", []):
        speaker = event.get("speaker", "?")
        text = event.get("text", "")
        action = event.get("action", "?")
        consensus = event.get("consensus", "?")
        print(f"{speaker}: {text} [{action}, {consensus}]")
        time.sleep(interval)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.replay:
        return _run_replay(args.replay, args.interval)
    return _run_live(args)


if __name__ == "__main__":
    sys.exit(main())
