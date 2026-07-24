#!/usr/bin/env python3
"""Kindred H3 integration check.

Run from the repo root:

    python scripts/integration_check.py

Verifies every workstream's output against the contracts in OWNERSHIP.md,
backend/README.md, frontend/README.md, loop/contracts.py and
agents/phase_card_agent.md, so nothing silently drifts at merge time.

Each check runs independently and prints SKIP if its input file doesn't
exist yet (branch not merged), FAIL on a contract violation, PASS otherwise.
Exit code is 1 if any check FAILs, 0 otherwise (SKIPs are fine).

Standard library only.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGENT_WHITELIST = {
    "profiler", "matcher", "evaluator", "introducer",
    "actian", "pioneer", "band", "gemini",
}
VALID_ACTIONS = {"speak", "agree", "disagree", "resolve"}

Status = str  # "PASS" | "FAIL" | "SKIP"


class Result:
    def __init__(self, name: str, status: Status, detail: str) -> None:
        self.name = name
        self.status = status
        self.detail = detail

RESULTS: list[Result] = []


def record(name: str, status: Status, detail: str = "") -> None:
    RESULTS.append(Result(name, status, detail))


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Check 1: graph payload contract (A -> B)
# ---------------------------------------------------------------------------
def find_graph_files() -> list[str]:
    patterns = [
        os.path.join(ROOT, "backend", "sample_graph.json"),
        os.path.join(ROOT, "backend", "*graph*.json"),
        os.path.join(ROOT, "*graph*.json"),
    ]
    found: list[str] = []
    for pat in patterns:
        for p in glob.glob(pat):
            if os.path.isfile(p) and p not in found:
                found.append(p)
    return found


def check_graph_payload() -> None:
    files = find_graph_files()
    if not files:
        record("graph payload", "SKIP",
               "no backend/sample_graph.json or *graph*.json found -- workstream A/B not merged yet")
        return
    for path in files:
        rel = os.path.relpath(path, ROOT)
        try:
            data = load_json(path)
        except Exception as exc:
            record(f"graph payload ({rel})", "FAIL", f"invalid JSON: {exc}")
            continue

        expected_keys = {"center", "nodes", "edges", "reasons"}
        if not isinstance(data, dict) or set(data.keys()) != expected_keys:
            record(f"graph payload ({rel})", "FAIL",
                   f"expected exactly keys {sorted(expected_keys)}, got {sorted(data.keys()) if isinstance(data, dict) else type(data)}")
            continue

        problem = None
        if not isinstance(data["center"], str):
            problem = "center must be a string"
        elif not isinstance(data["nodes"], list) or not all(
            isinstance(n, dict) and set(n.keys()) >= {"id", "name", "score", "x", "y"}
            for n in data["nodes"]
        ):
            problem = "nodes must be a list of {id,name,score,x,y}"
        elif not isinstance(data["edges"], list) or not all(
            isinstance(e, dict) and set(e.keys()) >= {"source", "target", "weight"}
            for e in data["edges"]
        ):
            problem = "edges must be a list of {source,target,weight}"
        elif not isinstance(data["reasons"], dict) or not all(
            isinstance(v, list) and all(isinstance(s, str) for s in v)
            for v in data["reasons"].values()
        ):
            problem = "reasons must be a dict of id -> list[str]"

        if problem:
            record(f"graph payload ({rel})", "FAIL", problem)
        else:
            record(f"graph payload ({rel})", "PASS",
                   f"{len(data['nodes'])} nodes, {len(data['edges'])} edges")


# ---------------------------------------------------------------------------
# Check 2: run.json (loop -> viz bridge, loop/contracts.py schema)
# ---------------------------------------------------------------------------
def check_run_json() -> dict | None:
    path = os.path.join(ROOT, "run.json")
    if not os.path.isfile(path):
        record("run.json", "SKIP", "run.json not found -- workstream D not merged yet")
        return None

    try:
        data = load_json(path)
    except Exception as exc:
        record("run.json", "FAIL", f"invalid JSON: {exc}")
        return None

    top_keys = {"meta", "weights", "generations", "events"}
    if not isinstance(data, dict) or not top_keys.issubset(data.keys()):
        record("run.json", "FAIL", f"missing top-level keys, need {sorted(top_keys)}")
        return None

    meta = data.get("meta")
    meta_keys = {"generations", "seed", "base_rate", "final_rate"}
    if not isinstance(meta, dict) or not meta_keys.issubset(meta.keys()):
        record("run.json", "FAIL", f"meta must contain {sorted(meta_keys)}")
        return None

    if not isinstance(data["weights"], list) or not all(isinstance(w, list) for w in data["weights"]):
        record("run.json", "FAIL", "weights must be a list of lists")
        return None

    gens = data["generations"]
    gen_keys = {"gen", "rate", "weights"}
    if not isinstance(gens, list) or not all(
        isinstance(g, dict) and gen_keys.issubset(g.keys()) for g in gens
    ):
        record("run.json", "FAIL", "generations must be a list of {gen,rate,weights}")
        return None

    if not isinstance(data["events"], list):
        record("run.json", "FAIL", "events must be a list")
        return None

    base_rate = meta["base_rate"]
    final_rate = meta["final_rate"]
    if not (0.30 <= base_rate <= 0.50):
        record("run.json", "FAIL", f"meta.base_rate {base_rate} outside [0.30, 0.50]")
        return None
    if not (0.75 <= final_rate <= 0.95):
        record("run.json", "FAIL", f"meta.final_rate {final_rate} outside [0.75, 0.95]")
        return None

    rates = [g["rate"] for g in gens]
    for i in range(1, len(rates)):
        if rates[i] < rates[i - 1]:
            record("run.json", "FAIL",
                    f"generations[].rate not non-decreasing at index {i}: {rates[i - 1]} -> {rates[i]}")
            return None

    record("run.json", "PASS", f"{len(gens)} generations, {base_rate} -> {final_rate}")
    return data


# ---------------------------------------------------------------------------
# Check 3: phase cards (agents/phase_card_agent.md hard rules)
# ---------------------------------------------------------------------------
def validate_phase_card(card: dict, index: int) -> str | None:
    """Return the first violation found for a single card, or None if valid."""
    script = card.get("script")
    if not isinstance(script, list):
        return f"card {index}: missing/invalid 'script' list"

    n = len(script)
    if not (5 <= n <= 9):
        return f"card {index}: script has {n} lines, expected 5-9"

    speakers_seen: set[str] = set()
    banner_count = 0
    prev_consensus = -1.0

    for i, line in enumerate(script):
        if not isinstance(line, dict):
            return f"card {index} line {i}: not an object"

        speaker = line.get("speaker")
        if speaker not in AGENT_WHITELIST:
            return f"card {index} line {i}: speaker '{speaker}' not in agent whitelist"
        speakers_seen.add(speaker)

        text = line.get("text")
        if not isinstance(text, str) or len(text) > 90:
            return f"card {index} line {i}: text missing or exceeds 90 chars ({len(text) if isinstance(text, str) else 'n/a'})"

        action = line.get("action")
        if action not in VALID_ACTIONS:
            return f"card {index} line {i}: action '{action}' not one of {sorted(VALID_ACTIONS)}"

        consensus = line.get("consensus")
        if not isinstance(consensus, (int, float)) or isinstance(consensus, bool):
            return f"card {index} line {i}: consensus missing or not numeric"
        if consensus < prev_consensus:
            return f"card {index} line {i}: consensus decreased ({prev_consensus} -> {consensus})"
        prev_consensus = consensus

        if "banner" in line and line["banner"] is not None:
            banner_count += 1

    last = script[-1]
    if last.get("action") != "resolve" or last.get("consensus") != 1.0:
        return f"card {index}: last line must be action=='resolve' and consensus==1.0"

    if not (1 <= banner_count <= 3):
        return f"card {index}: {banner_count} banners, expected 1-3"

    if not (4 <= len(speakers_seen) <= 6):
        return f"card {index}: {len(speakers_seen)} distinct speakers, expected 4-6"

    return None


def check_phase_cards() -> None:
    pattern = os.path.join(ROOT, "viz", "phase_cards*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        record("phase cards", "SKIP", "no viz/phase_cards*.json found -- phase-card agent not run yet")
        return

    for path in files:
        rel = os.path.relpath(path, ROOT)
        try:
            data = load_json(path)
        except Exception as exc:
            record(f"phase cards ({rel})", "FAIL", f"invalid JSON: {exc}")
            continue

        cards = data.get("cards") if isinstance(data, dict) else None
        if not isinstance(cards, list) or not cards:
            record(f"phase cards ({rel})", "FAIL", "no 'cards' list found")
            continue

        violation = None
        for idx, card in enumerate(cards):
            violation = validate_phase_card(card, idx)
            if violation:
                break

        if violation:
            record(f"phase cards ({rel})", "FAIL", violation)
        else:
            record(f"phase cards ({rel})", "PASS", f"{len(cards)} cards, all hard rules satisfied")


# ---------------------------------------------------------------------------
# Check 4: agent event shape (SSE contract), only if run.json exists
# ---------------------------------------------------------------------------
def check_event_shape(run_data: dict | None) -> None:
    if run_data is None:
        record("agent event shape", "SKIP", "run.json not found -- nothing to check")
        return

    events = run_data.get("events", [])
    required = {"speaker", "text", "action", "consensus"}
    allowed = required | {"banner"}

    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            record("agent event shape", "FAIL", f"event {i}: not an object")
            return
        keys = set(ev.keys())
        if keys - allowed:
            record("agent event shape", "FAIL", f"event {i}: unexpected keys {sorted(keys - allowed)}")
            return
        if not required.issubset(keys):
            record("agent event shape", "FAIL", f"event {i}: missing keys {sorted(required - keys)}")
            return
        if not isinstance(ev["speaker"], str) or not isinstance(ev["text"], str):
            record("agent event shape", "FAIL", f"event {i}: speaker/text must be strings")
            return
        if ev["action"] not in VALID_ACTIONS:
            record("agent event shape", "FAIL", f"event {i}: invalid action '{ev['action']}'")
            return
        if not isinstance(ev["consensus"], (int, float)) or isinstance(ev["consensus"], bool):
            record("agent event shape", "FAIL", f"event {i}: consensus must be numeric")
            return
        if "banner" in ev and ev["banner"] is not None and not isinstance(ev["banner"], str):
            record("agent event shape", "FAIL", f"event {i}: banner must be a string when present")
            return

    record("agent event shape", "PASS", f"{len(events)} events, shape valid")


# ---------------------------------------------------------------------------
# Summary + entrypoint
# ---------------------------------------------------------------------------
def print_summary() -> int:
    name_w = max((len(r.name) for r in RESULTS), default=4)
    name_w = max(name_w, len("CHECK"))
    status_w = len("STATUS")

    print()
    print(f"{'CHECK'.ljust(name_w)}  {'STATUS'.ljust(status_w)}  DETAIL")
    print("-" * (name_w + status_w + 40))
    for r in RESULTS:
        print(f"{r.name.ljust(name_w)}  {r.status.ljust(status_w)}  {r.detail}")
    print("-" * (name_w + status_w + 40))

    n_pass = sum(1 for r in RESULTS if r.status == "PASS")
    n_fail = sum(1 for r in RESULTS if r.status == "FAIL")
    n_skip = sum(1 for r in RESULTS if r.status == "SKIP")
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print()

    return 1 if n_fail else 0


def main() -> int:
    check_graph_payload()
    run_data = check_run_json()
    check_phase_cards()
    check_event_shape(run_data)
    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
