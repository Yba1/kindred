"""Kindred evolution-loop shared contract.

Every module in `loop/` imports its types and constants from HERE so parallel
builders cannot drift. Do NOT change signatures without telling the integrator.

Pipeline:
  personas.py  -> generate_personas(n, seed) -> list[Persona]
  pairs.py     -> generate_pairs(personas, n, seed) -> list[Pair]   (calibrated ~41% same-topic land-rate)
  features.py  -> pair_features(a, b) -> list[float]                (order == FEATURES)
  scorer.py    -> score_pair(a, b, weights) -> float                (logistic fallback; Pioneer overrides)
  evolve.py    -> evolve(personas, pairs, generations, seed) -> EvolveResult
  narrate.py   -> narrate(result) -> list[dict]                     (agent events for the village)
  replay.py    -> write_run(result, events, path) / load_run(path) -> RunFile
  run.py       -> CLI: `python -m loop.run [--replay run.json]`     (integrator writes this)
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ---- domain vocab (deterministic) ----
TOPICS   = ["ai-infra", "climate", "fintech", "biotech", "consumer", "devtools", "robotics", "edtech"]
STAGES   = ["student", "ic", "senior", "founder", "investor", "operator"]
SEEKING  = ["cofounder", "hire", "mentor", "investor", "peer", "customer"]
EMBED_DIM = 16

# ---- feature order (THE weight vector aligns to this, D->B contract) ----
FEATURES = ["topic_sim", "trajectory_sim", "seeking_match", "embed_cos"]


@dataclass
class Persona:
    id: str
    name: str
    topic: str                       # one of TOPICS
    stage: str                       # current stage, one of STAGES
    trajectory: tuple                # ordered path of stages, e.g. ("ic","senior","founder")
    seeking: str                     # one of SEEKING
    embedding: list                  # length EMBED_DIM, floats in [-1,1]


@dataclass
class Pair:
    a: str                           # persona id
    b: str                           # persona id
    same_topic: bool
    same_trajectory: bool
    seeking_match: bool
    landed: int                      # ground-truth label: 1 = connected, 0 = didn't


@dataclass
class Generation:
    gen: int
    weights: list                    # length == len(FEATURES)
    rate: float                      # held-out connection-rate this generation


@dataclass
class EvolveResult:
    generations: list                # list[Generation], ordered gen 0..N
    weight_vectors: list             # ordered list of weight lists (D->B, drives graph re-cluster)
    base_rate: float                 # generation-0 held-out rate (~0.41)
    final_rate: float                # last generation held-out rate (~0.84)
    seed: int


# run.json schema (loop -> viz bridge):
#   {
#     "meta":   {"generations": int, "seed": int, "base_rate": float, "final_rate": float},
#     "weights":[[w0,w1,w2,w3], ...],                       # ordered, one per generation
#     "generations":[{"gen":0,"rate":0.41,"weights":[...]}, ...],
#     "events":[{"speaker","text","action","consensus","banner"}, ...]   # SSE playback
#   }
RunFile = dict
