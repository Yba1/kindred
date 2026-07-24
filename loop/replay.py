"""run.json read/write for Kindred's deterministic replay.

Schema (see loop/contracts.py for the authoritative comment):
  {
    "meta": {"generations": int, "seed": int, "base_rate": float, "final_rate": float},
    "weights": [[...], ...],                 # ordered weight vectors, one per accepted generation
    "generations": [{"gen": 0, "rate": 0.41, "weights": [...]}, ...],
    "events": [{"speaker","text","action","consensus","banner"?}, ...]
  }
"""
from __future__ import annotations
import json
import os

from loop.contracts import EvolveResult


def _round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def write_run(result: EvolveResult, events: list, path: str = "run.json") -> dict:
    run = {
        "meta": {
            "generations": len(result.generations),
            "seed": result.seed,
            "base_rate": result.base_rate,
            "final_rate": result.final_rate,
        },
        "weights": [list(w) for w in result.weight_vectors],
        "generations": [
            {"gen": g.gen, "rate": g.rate, "weights": list(g.weights)}
            for g in result.generations
        ],
        "events": list(events),
    }
    run = _round_floats(run)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    return run


def load_run(path: str = "run.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(run: dict) -> str:
    meta = run.get("meta", {})
    gens = meta.get("generations", len(run.get("generations", [])))
    base = meta.get("base_rate")
    final = meta.get("final_rate")
    n_events = len(run.get("events", []))
    base_s = f"{base:.0%}" if isinstance(base, (int, float)) else "?"
    final_s = f"{final:.0%}" if isinstance(final, (int, float)) else "?"
    return f"{gens} gens, {base_s} -> {final_s}, {n_events} events"


if __name__ == "__main__":
    if os.path.exists("run.json"):
        print(summarize(load_run("run.json")))
    else:
        print("No run.json found. Run `python -m loop.run` first to generate one.")
