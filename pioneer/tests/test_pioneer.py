"""Tests for the Kindred match scorer.

    cd pioneer && python -m unittest discover -s tests -v

stdlib unittest, so it runs anywhere python does — pytest picks these up too.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kindred_pioneer import features, metrics, mockdata, paths, scorer  # noqa: E402
from kindred_pioneer import pioneer_client  # noqa: E402
from kindred_pioneer.baseline import CosineBaseline  # noqa: E402
from kindred_pioneer.embeddings import cosine, embed  # noqa: E402
from kindred_pioneer.model import LogisticScorer, fit_logistic  # noqa: E402
from kindred_pioneer.schema import Person  # noqa: E402
from kindred_pioneer.server import ScoreHandler  # noqa: E402
from kindred_pioneer.train import stratified_folds, stratified_split, train_local_scorer  # noqa: E402


def a_person(**overrides) -> Person:
    base = dict(
        id="p001",
        name="Ada K.",
        domain="devtools",
        prior_domain="finance",
        stage="building",
        seniority=8,
        city="SF",
        interests=["agent infrastructure", "observability"],
        seeking=["cofounder", "seed capital"],
        offering=["technical mentorship"],
    )
    base.update(overrides)
    return Person(**base)


class TestSchema(unittest.TestCase):
    def test_bio_is_rendered_and_contains_the_profile(self):
        p = a_person()
        for token in ("Ada K.", "devtools", "finance", "cofounder", "technical mentorship", "SF"):
            self.assertIn(token, p.bio)

    def test_from_any_accepts_person_and_dict(self):
        p = a_person()
        self.assertIs(Person.from_any(p), p)
        self.assertEqual(Person.from_any(p.to_dict()).id, "p001")

    def test_from_any_rejects_incomplete_input(self):
        with self.assertRaises(ValueError):
            Person.from_any({"id": "x", "name": "y"})
        with self.assertRaises(TypeError):
            Person.from_any(["not", "a", "person"])

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            a_person(stage="hibernating")

    def test_extra_keys_are_ignored(self):
        blob = a_person().to_dict()
        blob["actian_vector_id"] = "vec-42"  # other workstreams add their own fields
        self.assertEqual(Person.from_any(blob).id, "p001")


class TestEmbeddings(unittest.TestCase):
    def test_deterministic_across_calls(self):
        self.assertTrue(np.array_equal(embed("hello agent infra"), embed("hello agent infra")))

    def test_self_similarity_is_one(self):
        self.assertAlmostEqual(cosine("a b c", "a b c"), 1.0, places=9)

    def test_related_text_scores_above_unrelated(self):
        anchor = "payments rails and underwriting in fintech"
        near = "underwriting and payments rails, fintech compliance"
        far = "protein design and lab automation for assays"
        self.assertGreater(cosine(anchor, near), cosine(anchor, far))

    def test_empty_text_does_not_divide_by_zero(self):
        self.assertEqual(cosine("", "anything"), 0.0)


class TestFeatures(unittest.TestCase):
    def test_symmetric_in_its_arguments(self):
        a, b = a_person(), a_person(id="p002", domain="fintech", stage="scaling", seniority=14)
        np.testing.assert_allclose(features.pair_features(a, b), features.pair_features(b, a))

    def test_cos_bio_is_feature_zero(self):
        self.assertEqual(features.FEATURE_NAMES[0], "cos_bio")
        a, b = a_person(), a_person(id="p002", domain="bio")
        self.assertAlmostEqual(features.pair_features(a, b)[0], cosine(a.bio, b.bio), places=9)

    def test_directional_fit_reads_ask_against_offer(self):
        asker = a_person(seeking=["seed capital"], offering=["design help"])
        giver = a_person(id="p002", seeking=["design help"], offering=["seed capital"])
        stranger = a_person(id="p003", seeking=["seed capital"], offering=["ml hiring"])
        idx = features.FEATURE_NAMES.index("directional_fit")
        self.assertGreater(
            features.pair_features(asker, giver)[idx],
            features.pair_features(asker, stranger)[idx],
        )

    def test_two_people_wanting_the_same_scarce_thing_are_not_complementary(self):
        # The cosine baseline's blind spot: identical asks look similar but nobody can supply.
        twin_a = a_person(seeking=["seed capital"], offering=["design help"])
        twin_b = a_person(id="p002", seeking=["seed capital"], offering=["design help"])
        idx = features.FEATURE_NAMES.index("directional_fit")
        self.assertEqual(features.pair_features(twin_a, twin_b)[idx], 0.0)

    def test_feature_vector_matches_declared_length(self):
        vec = features.pair_features(a_person(), a_person(id="p002"))
        self.assertEqual(len(vec), features.N_FEATURES)
        self.assertEqual(len(features.FEATURE_NAMES), features.N_FEATURES)
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_pair_text_carries_both_profiles(self):
        a, b = a_person(), a_person(id="p002", name="Bo L.")
        text = features.pair_to_text(a, b)
        self.assertIn("PERSON A:", text)
        self.assertIn("Bo L.", text)
        self.assertIn("TRAJECTORY:", text)


class TestMetrics(unittest.TestCase):
    def test_f1_matches_hand_computation(self):
        y = np.array([1.0, 1.0, 0.0, 0.0])
        s = np.array([0.9, 0.4, 0.8, 0.1])
        # threshold 0.5 -> tp=1, fp=1, fn=1 -> precision=recall=0.5 -> f1=0.5
        self.assertAlmostEqual(metrics.f1_at(y, s, 0.5), 0.5)

    def test_f1_is_zero_when_nothing_is_predicted_positive(self):
        y = np.array([1.0, 0.0])
        self.assertEqual(metrics.f1_at(y, np.array([0.1, 0.2]), 0.9), 0.0)

    def test_auc_of_perfect_ranking_is_one(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(metrics.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)

    def test_auc_of_constant_scores_is_half(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(metrics.roc_auc(y, np.ones(4)), 0.5)

    def test_auc_of_inverted_ranking_is_zero(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(metrics.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_best_threshold_finds_a_perfect_split(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(metrics.f1_at(y, s, metrics.best_threshold(y, s)), 1.0)

    def test_evaluate_reports_consistent_counts(self):
        y = np.array([1.0, 1.0, 0.0, 0.0])
        result = metrics.evaluate(y, np.array([0.9, 0.4, 0.8, 0.1]), 0.5)
        self.assertEqual(result.n, 4)
        self.assertEqual(result.positives, 2)
        self.assertAlmostEqual(result.accuracy, 0.5)


class TestModel(unittest.TestCase):
    def test_recovers_a_clean_signal(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 3))
        y = (X[:, 0] + 0.4 * rng.normal(size=400) > 0).astype(float)
        model = fit_logistic(X, y, l2=0.1)
        self.assertGreater(model.weights[0], 1.0)
        self.assertGreater(abs(model.weights[0]), abs(model.weights[1]))
        self.assertGreater(metrics.roc_auc(y, model.predict_proba(X)), 0.9)

    def test_probabilities_stay_in_range_under_extreme_input(self):
        model = fit_logistic(np.random.default_rng(1).normal(size=(50, 3)),
                             np.array([0.0, 1.0] * 25), l2=1.0)
        probs = model.predict_proba(np.full((3, 3), 1e6))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        self.assertTrue(np.all(np.isfinite(probs)))

    def test_constant_column_does_not_blow_up(self):
        X = np.hstack([np.random.default_rng(2).normal(size=(40, 2)), np.ones((40, 1))])
        model = fit_logistic(X, np.array([0.0, 1.0] * 20), l2=1.0)
        self.assertTrue(np.all(np.isfinite(model.weights)))

    def test_save_load_roundtrip_preserves_predictions(self):
        rng = np.random.default_rng(3)
        X, y = rng.normal(size=(60, 4)), (rng.random(60) > 0.5).astype(float)
        model = fit_logistic(X, y, l2=1.0)
        model.threshold = 0.42
        path = paths.ARTIFACT_DIR / "_roundtrip_test.json"
        try:
            model.save(path)
            restored = LogisticScorer.load(path)
            np.testing.assert_allclose(model.predict_proba(X), restored.predict_proba(X))
            self.assertAlmostEqual(restored.threshold, 0.42)
        finally:
            path.unlink(missing_ok=True)


class TestSplitting(unittest.TestCase):
    def setUp(self):
        self.pairs = mockdata.make_pairs(mockdata.make_cohort(40, seed=5), 120, seed=5)

    def test_split_is_disjoint_and_complete(self):
        train, test = stratified_split(self.pairs, seed=0)
        self.assertEqual(len(train) + len(test), len(self.pairs))
        self.assertEqual(len({id(p) for p in train} & {id(p) for p in test}), 0)

    def test_split_preserves_landing_rate(self):
        train, test = stratified_split(self.pairs, seed=0)
        overall = features.labels(self.pairs).mean()
        self.assertAlmostEqual(features.labels(train).mean(), overall, delta=0.06)
        self.assertAlmostEqual(features.labels(test).mean(), overall, delta=0.06)

    def test_split_is_deterministic_for_a_seed(self):
        first, _ = stratified_split(self.pairs, seed=1)
        second, _ = stratified_split(self.pairs, seed=1)
        self.assertEqual([p.a.id for p in first], [p.a.id for p in second])

    def test_folds_cover_every_row_exactly_once(self):
        y = features.labels(self.pairs)
        folds = stratified_folds(y, 5, seed=0)
        covered = np.concatenate(folds)
        self.assertEqual(sorted(covered.tolist()), list(range(len(y))))

    def test_every_fold_has_both_classes(self):
        y = features.labels(self.pairs)
        for fold in stratified_folds(y, 5, seed=0):
            self.assertGreater(y[fold].sum(), 0)
            self.assertLess(y[fold].sum(), len(fold))


class TestMockData(unittest.TestCase):
    def test_generation_is_deterministic(self):
        first, _ = mockdata.build_datasets(seed=0)
        second, _ = mockdata.build_datasets(seed=0)
        self.assertEqual([p.label for p in first], [p.label for p in second])
        self.assertEqual([p.a.id for p in first], [p.a.id for p in second])

    def test_pairs_are_distinct_and_never_self_paired(self):
        pairs, _ = mockdata.build_datasets(seed=0)
        seen = {frozenset((p.a.id, p.b.id)) for p in pairs}
        self.assertEqual(len(seen), len(pairs))
        self.assertFalse(any(p.a.id == p.b.id for p in pairs))

    def test_cohorts_do_not_overlap(self):
        main, cold = mockdata.build_datasets(seed=0)
        main_ids = {p.a.id for p in main} | {p.b.id for p in main}
        cold_ids = {p.a.id for p in cold} | {p.b.id for p in cold}
        self.assertEqual(main_ids & cold_ids, set())

    def test_landing_rate_is_calibrated_near_the_repo_figure(self):
        main, _ = mockdata.build_datasets(seed=0)
        self.assertAlmostEqual(features.labels(main).mean(), 0.41, delta=0.08)

    def test_complementary_pair_beats_duplicate_pair(self):
        giver = a_person(id="g", stage="scaling", seniority=12,
                         seeking=["peer group"], offering=["seed capital", "go-to-market advice"])
        asker = a_person(id="s", stage="scaling", seniority=12,
                         seeking=["seed capital"], offering=["design help"])
        twin = a_person(id="t", stage="scaling", seniority=12,
                        seeking=["seed capital"], offering=["design help"])
        self.assertGreater(
            mockdata.landing_probability(asker, giver),
            mockdata.landing_probability(asker, twin),
        )

    def test_jsonl_roundtrip(self):
        pairs, _ = mockdata.build_datasets(seed=0)
        path = paths.DATA_DIR / "_roundtrip_test.jsonl"
        try:
            mockdata.write_jsonl(pairs[:20], path)
            restored = mockdata.read_jsonl(path)
            self.assertEqual(len(restored), 20)
            self.assertEqual(restored[0].a.id, pairs[0].a.id)
            self.assertEqual(restored[0].label, pairs[0].label)
        finally:
            path.unlink(missing_ok=True)


class TestScorerAPI(unittest.TestCase):
    """The surface the evolution loop actually calls."""

    @classmethod
    def setUpClass(cls):
        if not paths.MODEL_PATH.exists():
            raise unittest.SkipTest("no trained model — run python -m kindred_pioneer.train")

    def test_returns_a_probability(self):
        value = scorer.score_pair(a_person(), a_person(id="p002", domain="bio"))
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_accepts_dicts_as_well_as_dataclasses(self):
        a, b = a_person(), a_person(id="p002", domain="climate")
        self.assertAlmostEqual(scorer.score_pair(a, b), scorer.score_pair(a.to_dict(), b.to_dict()))

    def test_is_symmetric(self):
        a, b = a_person(), a_person(id="p002", domain="fintech", stage="scaling")
        self.assertAlmostEqual(scorer.score_pair(a, b), scorer.score_pair(b, a), places=12)

    def test_batch_matches_individual_calls(self):
        people = [a_person(id=f"p{i:03d}", domain=d)
                  for i, d in enumerate(["devtools", "bio", "climate", "security"])]
        pairs = [(people[0], people[1]), (people[2], people[3]), (people[1], people[3])]
        np.testing.assert_allclose(
            scorer.score_pairs(pairs), [scorer.score_pair(a, b) for a, b in pairs]
        )

    def test_empty_batch_is_empty(self):
        self.assertEqual(scorer.score_pairs([]), [])

    def test_decide_agrees_with_the_threshold(self):
        a, b = a_person(), a_person(id="p002", domain="bio")
        self.assertEqual(scorer.decide(a, b), scorer.score_pair(a, b) >= scorer.threshold())

    def test_explain_returns_ranked_drivers(self):
        result = scorer.explain(a_person(), a_person(id="p002", domain="bio"))
        self.assertIn("score", result)
        self.assertIn(result["verdict"], {"connect", "pass"})
        magnitudes = [abs(d["contribution"]) for d in result["drivers"]]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        self.assertTrue(all(d["feature"] in features.FEATURE_NAMES for d in result["drivers"]))

    def test_explain_score_matches_score_pair(self):
        a, b = a_person(), a_person(id="p002", domain="climate")
        self.assertAlmostEqual(scorer.explain(a, b)["score"], scorer.score_pair(a, b), places=12)

    def test_info_reports_readiness(self):
        state = scorer.info()
        self.assertTrue(state["ready"])
        self.assertEqual(state["backend"], "local")
        self.assertEqual(state["features"], list(features.FEATURE_NAMES))

    def test_scores_track_the_trained_direction(self):
        """A complementary, same-trajectory pair should outrank a mismatched one."""
        asker = a_person(id="a", prior_domain="finance", stage="building",
                         seniority=8, city="SF", seeking=["seed capital"], offering=["design help"])
        giver = a_person(id="g", prior_domain="finance", stage="building",
                         seniority=8, city="SF", seeking=["design help"], offering=["seed capital"])
        mismatch = a_person(id="m", prior_domain="medicine", stage="exploring",
                            seniority=1, city="Berlin",
                            seeking=["seed capital"], offering=["regulatory guidance"])
        self.assertGreater(scorer.score_pair(asker, giver), scorer.score_pair(asker, mismatch))


class TestBeatsBaseline(unittest.TestCase):
    """The win condition, re-derived from scratch rather than read from a file."""

    def test_scorer_outranks_cosine_on_held_out_pairs(self):
        pairs, cold = mockdata.build_datasets(seed=0)
        train, test = stratified_split(pairs, seed=0)
        model, _ = train_local_scorer(train, seed=0)
        base = CosineBaseline().fit_threshold(train)

        y = features.labels(test)
        scorer_scores = model.predict_proba(features.feature_matrix(test))
        base_scores = base.scores(test)

        scorer_f1 = metrics.f1_at(y, scorer_scores, model.threshold)
        base_f1 = metrics.f1_at(y, base_scores, base.threshold)
        self.assertGreater(scorer_f1, base_f1)
        self.assertGreater(metrics.roc_auc(y, scorer_scores), metrics.roc_auc(y, base_scores))

    def test_scorer_generalises_to_unseen_people(self):
        pairs, cold = mockdata.build_datasets(seed=0)
        model, _ = train_local_scorer(pairs, seed=0)
        base = CosineBaseline().fit_threshold(pairs)
        y = features.labels(cold)
        self.assertGreater(
            metrics.f1_at(y, model.predict_proba(features.feature_matrix(cold)), model.threshold),
            metrics.f1_at(y, base.scores(cold), base.threshold),
        )


class TestPioneerClient(unittest.TestCase):
    """No network: the parts that must be right before a key is ever plugged in."""

    def test_training_jsonl_uses_pioneer_classification_schema(self):
        payload = pioneer_client.PioneerClient.build_classification_jsonl(
            [("pair one", 1), ("pair two", 0)]
        )
        rows = [json.loads(line) for line in payload.decode().strip().split("\n")]
        self.assertEqual(rows[0], {"text": "pair one", "label": "connect"})
        self.assertEqual(rows[1], {"text": "pair two", "label": "pass"})

    def test_extracts_probability_from_a_label_score_list(self):
        response = {"result": {"classifications": {
            "will_connect": [{"label": "connect", "score": 0.83}, {"label": "pass", "score": 0.17}]
        }}}
        self.assertAlmostEqual(pioneer_client.extract_connect_probability(response), 0.83)

    def test_extracts_probability_from_a_flat_mapping(self):
        self.assertAlmostEqual(
            pioneer_client.extract_connect_probability({"result": {"connect": 0.61, "pass": 0.39}}),
            0.61,
        )

    def test_infers_from_the_negative_class_alone(self):
        response = {"result": [{"label": "pass", "score": 0.25}]}
        self.assertAlmostEqual(pioneer_client.extract_connect_probability(response), 0.75)

    def test_normalises_percentages(self):
        self.assertAlmostEqual(
            pioneer_client.extract_connect_probability({"label": "connect", "score": 92.0}), 0.92
        )

    def test_raises_rather_than_guessing_when_the_class_is_absent(self):
        with self.assertRaises(pioneer_client.PioneerError):
            pioneer_client.extract_connect_probability({"result": {"entities": {"org": ["Apple"]}}})

    def test_client_refuses_to_construct_without_a_key(self):
        with self.assertRaises(pioneer_client.PioneerError):
            pioneer_client.PioneerClient(key=None, base_url="https://example.invalid")

    def test_dig_finds_nested_status(self):
        self.assertEqual(
            pioneer_client._dig({"data": {"job": {"status": "complete"}}}, "status"), "complete"
        )


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not paths.MODEL_PATH.exists():
            raise unittest.SkipTest("no trained model — run python -m kindred_pioneer.train")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), ScoreHandler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def post(self, route: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.base + route,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            with exc:  # close the error body so it doesn't warn on GC
                return exc.code, json.loads(exc.read())

    def test_health_reports_ready(self):
        with urllib.request.urlopen(self.base + "/health", timeout=10) as resp:
            payload = json.loads(resp.read())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["ready"])

    def test_score_matches_the_in_process_call(self):
        a, b = a_person(), a_person(id="p002", domain="bio")
        status, payload = self.post("/score", {"a": a.to_dict(), "b": b.to_dict()})
        self.assertEqual(status, 200)
        self.assertAlmostEqual(payload["score"], scorer.score_pair(a, b), places=9)
        self.assertAlmostEqual(payload["threshold"], scorer.threshold(), places=9)

    def test_batch_route_returns_one_score_per_pair(self):
        a, b = a_person().to_dict(), a_person(id="p002", domain="climate").to_dict()
        status, payload = self.post("/score/batch", {"pairs": [{"a": a, "b": b}] * 3})
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["scores"]), 3)
        # Not exact equality: a batched matmul can differ from a single-row one in
        # the last bit (~1e-16) depending on how BLAS blocks the rows.
        np.testing.assert_allclose(payload["scores"], payload["scores"][0], atol=1e-12)

    def test_explain_route(self):
        a, b = a_person().to_dict(), a_person(id="p002", domain="security").to_dict()
        status, payload = self.post("/explain", {"a": a, "b": b})
        self.assertEqual(status, 200)
        self.assertIn("drivers", payload)

    def test_malformed_pair_is_a_400(self):
        status, payload = self.post("/score", {"a": {"id": "x"}, "b": {"id": "y"}})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_missing_body_keys_is_a_400(self):
        status, _ = self.post("/score", {"a": {"id": "x"}})
        self.assertEqual(status, 400)

    def test_unknown_route_is_a_404(self):
        status, _ = self.post("/nope", {})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
