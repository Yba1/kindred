"""Unit tests for the Kindred evolution-loop modules (loop/*.py).

Run with:
    python -m unittest tests.test_loop -v
from the repo root.

Only reads loop/*.py — never edits it. Guards the evolve.py tests with
skipUnless so the rest of the suite still runs (and reports clearly) even
if evolve.py doesn't exist yet or is incomplete.
"""
from __future__ import annotations

import itertools
import os
import sys
import unittest

# --- sys.path shim: make sure the repo root (parent of this tests/ dir) is
# importable even if the harness didn't already put it on sys.path. ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loop.contracts import FEATURES
from loop.personas import generate_personas
from loop.pairs import generate_pairs
from loop.features import pair_features
from loop.scorer import score_features, score_pair


# Try to import evolve.py; it may not exist yet or may still be incomplete.
try:
    from loop import evolve as evolve_module
    _EVOLVE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only when evolve.py is missing/broken
    evolve_module = None
    _EVOLVE_IMPORT_ERROR = exc


def _persona_key(p):
    """Comparable snapshot of a persona's identity-relevant fields."""
    return (p.id, p.name, p.topic, p.focus, p.stage, tuple(p.trajectory),
            p.seeking, p.style, p.expertise, tuple(p.embedding))


def _pair_key(pair):
    return (pair.a, pair.b, pair.same_topic, pair.same_trajectory,
            pair.seeking_match, pair.same_style, pair.landed)


class TestDeterminism(unittest.TestCase):
    def test_generate_personas_is_deterministic(self):
        first = generate_personas(n=50, seed=42)
        second = generate_personas(n=50, seed=42)
        self.assertEqual(len(first), len(second))
        self.assertEqual(
            [_persona_key(p) for p in first],
            [_persona_key(p) for p in second],
        )

    def test_generate_pairs_is_deterministic(self):
        personas = generate_personas(n=60, seed=42)
        first = generate_pairs(personas, n=100, seed=43)
        second = generate_pairs(personas, n=100, seed=43)
        self.assertEqual(len(first), len(second))
        self.assertEqual(
            [_pair_key(p) for p in first],
            [_pair_key(p) for p in second],
        )


class TestCalibration(unittest.TestCase):
    def test_same_domain_land_rate_is_in_calibrated_band(self):
        personas = generate_personas(n=300, seed=42)
        pairs = generate_pairs(personas, n=400, seed=43)

        same_domain = [p for p in pairs if p.same_topic]
        self.assertGreater(len(same_domain), 0, "need same-domain pairs to check calibration")

        rate = sum(p.landed for p in same_domain) / len(same_domain)
        # Slightly loose band around the documented ~0.41 target to avoid flakiness.
        self.assertGreaterEqual(rate, 0.36)
        self.assertLessEqual(rate, 0.46)


class TestFeatures(unittest.TestCase):
    def test_pair_features_shape_and_range(self):
        personas = generate_personas(n=20, seed=7)
        sampled_pairs = list(itertools.combinations(personas, 2))[:40]
        self.assertGreater(len(sampled_pairs), 0)

        for a, b in sampled_pairs:
            vec = pair_features(a, b)
            self.assertEqual(len(vec), len(FEATURES))
            for value in vec:
                self.assertIsInstance(value, float)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class TestScorer(unittest.TestCase):
    def test_score_features_returns_float_in_unit_interval(self):
        n_feat = len(FEATURES)
        weight_vectors = [
            [0.0] * n_feat,
            [1.0] * n_feat,
            [-1.0] * n_feat,
            [0.2, -0.3, 1.3, 1.6, -1.1, 0.9][:n_feat],
        ]
        feats = [0.5] * n_feat
        for weights in weight_vectors:
            score = score_features(feats, weights)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_score_pair_returns_float_in_unit_interval(self):
        personas = generate_personas(n=10, seed=1)
        a, b = personas[0], personas[1]
        weights = [0.2, 0.2, 1.3, 1.6, 1.1, 0.9][: len(FEATURES)]
        score = score_pair(a, b, weights, bias=-1.0)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_features_rejects_wrong_length_weights(self):
        n_feat = len(FEATURES)
        feats = [0.5] * n_feat
        bad_weights = [0.1] * (n_feat + 1)
        with self.assertRaises(AssertionError):
            score_features(feats, bad_weights)

    def test_score_features_rejects_wrong_length_feats(self):
        n_feat = len(FEATURES)
        bad_feats = [0.5] * (n_feat - 1)
        weights = [0.1] * n_feat
        with self.assertRaises(AssertionError):
            score_features(bad_feats, weights)


@unittest.skipUnless(
    evolve_module is not None,
    f"loop/evolve.py not importable yet: {_EVOLVE_IMPORT_ERROR!r}",
)
class TestEvolve(unittest.TestCase):
    def test_evolve_promotion_gate(self):
        # Use the same scale/seeds loop.run.py's defaults use — evolve.py's GD
        # tuning (step count, split stream) targets this exact configuration;
        # smaller/differently-seeded samples aren't guaranteed to hit the band.
        personas = generate_personas(n=300, seed=42)
        pairs = generate_pairs(personas, n=250, seed=43)

        result = evolve_module.evolve(personas, pairs, generations=6, seed=44)

        self.assertGreaterEqual(result.base_rate, 0.36)
        self.assertLessEqual(result.base_rate, 0.46)

        self.assertGreaterEqual(result.final_rate, 0.75)
        self.assertLessEqual(result.final_rate, 0.90)

        rates = [g.rate for g in result.generations]
        self.assertTrue(len(rates) >= 2, "expected multiple recorded generations")
        for prev, curr in zip(rates, rates[1:]):
            self.assertGreaterEqual(
                curr, prev,
                f"promotion-gate violated: rate decreased from {prev} to {curr}",
            )


if __name__ == "__main__":
    unittest.main()
