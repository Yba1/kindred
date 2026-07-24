"""One training job + the honest comparison against the embedding baseline.

    python -m kindred_pioneer.train                     # local head, offline, deterministic
    python -m kindred_pioneer.train --backend pioneer   # also run the real Pioneer fine-tune

Writes artifacts/model.json (what the loop loads), artifacts/metrics.json and
artifacts/report.md, and prints the comparison table to stdout.

Protocol, so the headline number is defensible:
  * 75/25 stratified train/test split of the labeled pairs.
  * The scorer's L2 strength *and* decision threshold come from 5-fold CV inside
    train, using out-of-fold predictions only.
  * The baseline's threshold is fitted on the whole training split — if anything
    generous to it, since cosine has no other parameters to overfit with.
  * Test is scored once, at the end, by both.
  * Reported next to the point estimate: a bootstrap CI on the F1 delta, a
    10-seed split sweep, and a cold-start cohort of people never seen in training.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from . import embeddings, features, metrics, mockdata, paths, pioneer_client
from .baseline import CosineBaseline, TrivialBaseline
from .model import LogisticScorer, fit_logistic
from .schema import LabeledPair

# Wide on purpose: at 150 rows and 10 features CV consistently picks the heavily
# shrunk end, and a grid that topped out at 10 was clipping the choice.
L2_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
CV_FOLDS = 5


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def stratified_split(
    pairs: list[LabeledPair], seed: int = 0, train_fraction: float = 0.75
) -> tuple[list[LabeledPair], list[LabeledPair]]:
    """Train/test split preserving the landing rate in each part.

    There is no separate validation split: with only ~200 pairs a 40-row
    validation set makes hyperparameter selection pure noise (we measured it
    picking the worse model). Selection happens by k-fold CV *inside* train,
    which leaves test untouched and gives the fit more data.
    """
    rng = np.random.default_rng(seed)
    train: list[LabeledPair] = []
    test: list[LabeledPair] = []

    for label in (0, 1):
        group = [p for p in pairs if p.label == label]
        idx = rng.permutation(len(group))
        n_train = int(round(train_fraction * len(group)))
        train += [group[i] for i in idx[:n_train]]
        test += [group[i] for i in idx[n_train:]]

    shuffler = random.Random(seed)
    shuffler.shuffle(train)
    shuffler.shuffle(test)
    return train, test


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Indices for k stratified folds — keeps every fold scoreable."""
    rng = np.random.default_rng(seed + 101)
    folds: list[list[int]] = [[] for _ in range(k)]
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        for slot, i in enumerate(rng.permutation(idx)):
            folds[slot % k].append(int(i))
    return [np.array(sorted(f), dtype=int) for f in folds]


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def out_of_fold_scores(
    X: np.ndarray, y: np.ndarray, l2: float, folds: list[np.ndarray]
) -> np.ndarray:
    """Predictions for every training row from a model that never saw that row."""
    oof = np.zeros(len(y), dtype=np.float64)
    for fold in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[fold] = False
        model = fit_logistic(X[mask], y[mask], l2=l2)
        oof[fold] = model.predict_proba(X[fold])
    return oof


def train_local_scorer(
    train_pairs: list[LabeledPair], seed: int = 0, k: int = CV_FOLDS
) -> tuple[LogisticScorer, float]:
    """Fit the scorer, choosing L2 and the decision threshold by k-fold CV on train.

    Both are picked from out-of-fold predictions, so neither is tuned against
    data the final model was fit to, and test stays untouched until the end.
    """
    X, y = features.feature_matrix(train_pairs), features.labels(train_pairs)
    folds = stratified_folds(y, k, seed)

    best_l2, best_auc, best_oof = L2_GRID[0], -1.0, None
    for l2 in L2_GRID:
        oof = out_of_fold_scores(X, y, l2, folds)
        auc = metrics.roc_auc(y, oof)
        if auc > best_auc:
            best_l2, best_auc, best_oof = l2, auc, oof

    model = fit_logistic(X, y, l2=best_l2)
    assert best_oof is not None
    model.threshold = metrics.best_threshold(y, best_oof)
    return model, best_l2


def run_pioneer_job(
    train_pairs: list[LabeledPair], test_pairs: list[LabeledPair], seed: int
) -> tuple[pioneer_client.TrainingResult, np.ndarray, float]:
    """Fine-tune on Pioneer, then score the held-out split with the result.

    Five-fold CV would mean five fine-tunes, so the Pioneer path instead holds
    out 15% of train to set its threshold and fits on the rest. Only called with
    an API key present; any failure propagates, because a Pioneer run that didn't
    happen must not be reported as one that did.
    """
    fit_pairs, threshold_pairs = stratified_split(train_pairs, seed=seed, train_fraction=0.85)
    client = pioneer_client.PioneerClient()

    rows = [(features.pair_to_text(p.a, p.b), p.label) for p in fit_pairs]
    payload = pioneer_client.PioneerClient.build_classification_jsonl(rows)
    print(f"[pioneer] uploading {len(rows)} rows as '{pioneer_client.DATASET_NAME}' ...")
    client.upload_dataset(pioneer_client.DATASET_NAME, payload)
    client.wait_for_dataset(pioneer_client.DATASET_NAME)

    print(f"[pioneer] LoRA fine-tuning {pioneer_client.DEFAULT_BASE_MODEL} ...")
    job_id = client.start_training(pioneer_client.DATASET_NAME)
    status = client.wait_for_training(job_id)
    print(f"[pioneer] job {job_id} complete")

    try:
        pioneer_metrics = client.evaluate(job_id, pioneer_client.DATASET_NAME)
    except pioneer_client.PioneerError as exc:  # their eval is a nice-to-have, not the gate
        print(f"[pioneer] evaluation skipped: {exc}")
        pioneer_metrics = {}

    def score_all(pairs: list[LabeledPair]) -> np.ndarray:
        return np.array(
            [client.score_text(job_id, features.pair_to_text(p.a, p.b)) for p in pairs],
            dtype=np.float64,
        )

    threshold = metrics.best_threshold(features.labels(threshold_pairs), score_all(threshold_pairs))
    result = pioneer_client.TrainingResult(
        job_id=job_id,
        status=str(pioneer_client._dig(status, "status") or "complete"),
        base_model=pioneer_client.DEFAULT_BASE_MODEL,
        dataset=pioneer_client.DATASET_NAME,
        metrics=pioneer_metrics if isinstance(pioneer_metrics, dict) else {},
        logs_tail=client.training_logs(job_id),
    )
    return result, score_all(test_pairs), threshold


# --------------------------------------------------------------------------
# robustness checks
# --------------------------------------------------------------------------

def seed_sweep(pairs: list[LabeledPair], seeds: range) -> dict[str, object]:
    """Re-split and refit across seeds — guards against a lucky single split."""
    scorer_f1: list[float] = []
    baseline_f1: list[float] = []
    for seed in seeds:
        tr, te = stratified_split(pairs, seed=seed)
        model, _ = train_local_scorer(tr, seed=seed)
        y_te = features.labels(te)

        scorer_f1.append(
            metrics.f1_at(y_te, model.predict_proba(features.feature_matrix(te)), model.threshold)
        )
        base = CosineBaseline().fit_threshold(tr)
        baseline_f1.append(metrics.f1_at(y_te, base.scores(te), base.threshold))

    scorer_arr, base_arr = np.array(scorer_f1), np.array(baseline_f1)
    return {
        "seeds": list(seeds),
        "scorer_f1_mean": float(scorer_arr.mean()),
        "scorer_f1_sd": float(scorer_arr.std(ddof=1)),
        "baseline_f1_mean": float(base_arr.mean()),
        "baseline_f1_sd": float(base_arr.std(ddof=1)),
        "mean_delta": float((scorer_arr - base_arr).mean()),
        "scorer_wins": int(np.sum(scorer_arr > base_arr)),
        "n_seeds": len(scorer_arr),
        "scorer_f1": scorer_f1,
        "baseline_f1": baseline_f1,
    }


def coldstart_check(
    main_pairs: list[LabeledPair], cold_pairs: list[LabeledPair], seed: int
) -> dict[str, object]:
    """Train on the full main cohort, test on people the model has never seen."""
    model, _ = train_local_scorer(main_pairs, seed=seed)
    base = CosineBaseline().fit_threshold(main_pairs)

    y = features.labels(cold_pairs)
    scorer = metrics.evaluate(y, model.predict_proba(features.feature_matrix(cold_pairs)), model.threshold)
    baseline = metrics.evaluate(y, base.scores(cold_pairs), base.threshold)
    return {
        "n": len(cold_pairs),
        "scorer": asdict(scorer),
        "baseline": asdict(baseline),
        "delta_f1": scorer.f1 - baseline.f1,
    }


def bayes_ceiling(pairs: list[LabeledPair]) -> dict[str, float]:
    """What a model that knew the generative probability exactly would score.

    Labels are Bernoulli draws, so F1 well under 1.0 is the ceiling, not a bug.
    Printing it stops the headline number being read against an impossible 1.0.
    """
    y = features.labels(pairs)
    p = np.array([pair.p_true for pair in pairs], dtype=np.float64)
    return {
        "roc_auc": metrics.roc_auc(y, p),
        "f1": metrics.f1_at(y, p, metrics.best_threshold(y, p)),
    }


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Kindred pair scorer.")
    parser.add_argument("--backend", choices=("local", "pioneer"), default="local",
                        help="'pioneer' additionally runs the real fine-tune (needs PIONEER_API_KEY)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--people", type=int, default=64)
    parser.add_argument("--regenerate", action="store_true", help="rebuild data/pairs.jsonl")
    parser.add_argument("--sweep-seeds", type=int, default=10)
    args = parser.parse_args(argv)

    # ---- data -----------------------------------------------------------
    if args.regenerate or not paths.PAIRS_PATH.exists():
        main_pairs, cold_pairs = mockdata.build_datasets(
            seed=args.seed, n_people=args.people, n_pairs=args.pairs
        )
        mockdata.write_jsonl(main_pairs, paths.PAIRS_PATH)
        mockdata.write_jsonl(cold_pairs, paths.COLDSTART_PATH)
        print(f"generated {len(main_pairs)} pairs -> {paths.PAIRS_PATH.name}, "
              f"{len(cold_pairs)} cold-start pairs -> {paths.COLDSTART_PATH.name}")
    else:
        main_pairs = mockdata.read_jsonl(paths.PAIRS_PATH)
        cold_pairs = mockdata.read_jsonl(paths.COLDSTART_PATH)
        print(f"loaded {len(main_pairs)} pairs from {paths.PAIRS_PATH.name}")

    train_pairs, test_pairs = stratified_split(main_pairs, seed=args.seed)
    y_test = features.labels(test_pairs)
    landing_rate = float(features.labels(main_pairs).mean())
    ceiling = bayes_ceiling(main_pairs)

    print(f"\nsplit: train={len(train_pairs)}  test={len(test_pairs)}  "
          f"| landing rate={landing_rate:.1%}  | embeddings={embeddings.backend_name()}")
    print(f"ceiling: a model that knew the true landing probability would score "
          f"F1={ceiling['f1']:.3f}, AUC={ceiling['roc_auc']:.3f} (labels are Bernoulli draws)")

    # Keep a reviewable copy of exactly what gets uploaded to Pioneer.
    pioneer_client.save_dataset_preview(
        [(features.pair_to_text(p.a, p.b), p.label) for p in train_pairs],
        paths.PIONEER_UPLOAD_PREVIEW,
        limit=5,
    )

    # ---- baselines ------------------------------------------------------
    trivial = TrivialBaseline().fit_threshold(train_pairs)
    cosine = CosineBaseline().fit_threshold(train_pairs)
    cosine_test_scores = cosine.scores(test_pairs)

    trivial_test = metrics.evaluate(y_test, trivial.scores(test_pairs), trivial.threshold)
    cosine_test = metrics.evaluate(y_test, cosine_test_scores, cosine.threshold)

    # ---- scorer ---------------------------------------------------------
    model, chosen_l2 = train_local_scorer(train_pairs, seed=args.seed)
    scorer_scores = model.predict_proba(features.feature_matrix(test_pairs))
    scorer_test = metrics.evaluate(y_test, scorer_scores, model.threshold)

    pioneer_result = pioneer_test = None
    if args.backend == "pioneer":
        if not pioneer_client.is_configured():
            print("\n[pioneer] PIONEER_API_KEY is not set — cannot run the fine-tune.", file=sys.stderr)
            return 2
        pioneer_result, p_test, p_threshold = run_pioneer_job(train_pairs, test_pairs, args.seed)
        pioneer_test = metrics.evaluate(y_test, p_test, p_threshold)
        model.backend = "pioneer"

    # ---- report ---------------------------------------------------------
    print(f"\nHELD-OUT TEST SET (n={len(test_pairs)}, {int(y_test.sum())} landed)")
    print("-" * len(metrics.HEADER))
    print(metrics.HEADER)
    print("-" * len(metrics.HEADER))
    print(trivial_test.row("always-connect floor"))
    print(cosine_test.row("cosine baseline"))
    print(scorer_test.row("kindred scorer (local)"))
    if pioneer_test is not None:
        print(pioneer_test.row("kindred scorer (Pioneer)"))
    print("-" * len(metrics.HEADER))

    delta = scorer_test.f1 - cosine_test.f1
    lo, med, hi = metrics.bootstrap_delta_f1(
        y_test, scorer_scores, cosine_test_scores, model.threshold, cosine.threshold, seed=args.seed
    )
    print(f"\nF1 delta vs cosine baseline: {delta:+.3f}  "
          f"(bootstrap 95% CI {lo:+.3f} .. {hi:+.3f}, median {med:+.3f})")
    print(f"verdict: {'BEATS BASELINE' if delta > 0 else 'DOES NOT BEAT BASELINE'} "
          f"({'CI excludes zero' if lo > 0 else f'CI includes zero at n={len(test_pairs)}'})")

    sweep = seed_sweep(main_pairs, range(args.sweep_seeds))
    print(f"\n{args.sweep_seeds}-seed split sweep: "
          f"scorer F1 {sweep['scorer_f1_mean']:.3f} +/- {sweep['scorer_f1_sd']:.3f} | "
          f"baseline F1 {sweep['baseline_f1_mean']:.3f} +/- {sweep['baseline_f1_sd']:.3f} | "
          f"mean delta {sweep['mean_delta']:+.3f} | wins {sweep['scorer_wins']}/{sweep['n_seeds']}")

    cold = coldstart_check(main_pairs, cold_pairs, seed=args.seed)
    print(f"cold-start cohort (n={cold['n']}, unseen people): "
          f"scorer F1 {cold['scorer']['f1']:.3f} | baseline F1 {cold['baseline']['f1']:.3f} | "
          f"delta {cold['delta_f1']:+.3f}")

    print(f"\nlearned weights (standardised, |w| desc, L2={chosen_l2}):")
    for name, weight in model.coefficients():
        print(f"  {name:<24} {weight:+.3f}")

    # ---- persist --------------------------------------------------------
    model.save(paths.MODEL_PATH)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": args.seed,
        "backend": model.backend,
        "l2": chosen_l2,
        "cv_folds": CV_FOLDS,
        "dataset": {
            "n_pairs": len(main_pairs),
            "landing_rate": landing_rate,
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
            "source": "mock (mockdata.build_datasets) — swap for the loop's real landings",
        },
        "ceiling": ceiling,
        "test": {
            "trivial": asdict(trivial_test),
            "cosine_baseline": asdict(cosine_test),
            "scorer": asdict(scorer_test),
            **({"pioneer": asdict(pioneer_test)} if pioneer_test is not None else {}),
        },
        "delta_f1_vs_cosine": delta,
        "delta_f1_bootstrap_ci": [lo, med, hi],
        "seed_sweep": sweep,
        "coldstart": cold,
        "coefficients": dict(model.coefficients()),
        "pioneer_job": pioneer_result.to_dict() if pioneer_result else None,
    }
    paths.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    paths.METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_report(payload, model)
    print(f"\nwrote {paths.MODEL_PATH.name}, {paths.METRICS_PATH.name}, {paths.REPORT_PATH.name} "
          f"-> {paths.ARTIFACT_DIR}")
    return 0 if delta > 0 else 1


def write_report(payload: dict, model: LogisticScorer) -> None:
    test = payload["test"]
    lo, med, hi = payload["delta_f1_bootstrap_ci"]
    sweep, cold, ceiling = payload["seed_sweep"], payload["coldstart"], payload["ceiling"]
    job = payload["pioneer_job"]

    def row(name: str, s: dict) -> str:
        return (f"| {name} | {s['f1']:.3f} | {s['precision']:.3f} | {s['recall']:.3f} | "
                f"{s['accuracy']:.3f} | {s['roc_auc']:.3f} |")

    lines = [
        "# Kindred match scorer — held-out results",
        "",
        f"Generated {payload['generated_at']} · seed {payload['seed']} · backend `{payload['backend']}` "
        f"· L2 {payload['l2']} · {payload['cv_folds']}-fold CV",
        "",
        f"{payload['dataset']['n_pairs']} labeled pairs ({payload['dataset']['landing_rate']:.1%} landed), "
        f"split {payload['dataset']['n_train']} train / {payload['dataset']['n_test']} test. "
        f"{payload['dataset']['source']}.",
        "",
        "The scorer's L2 and threshold come from out-of-fold predictions inside train; the "
        "baseline's threshold is fitted on all of train. Test is scored once.",
        "",
        "| model | F1 | precision | recall | accuracy | ROC-AUC |",
        "|---|---|---|---|---|---|",
        row("always-connect floor", test["trivial"]),
        row("cosine baseline", test["cosine_baseline"]),
        row("**kindred scorer**", test["scorer"]),
    ]
    if "pioneer" in test:
        lines.append(row("**kindred scorer (Pioneer fine-tune)**", test["pioneer"]))

    lines += [
        "",
        f"**F1 delta vs cosine baseline: {payload['delta_f1_vs_cosine']:+.3f}** "
        f"(bootstrap 95% CI {lo:+.3f} .. {hi:+.3f}).",
        "",
        f"For scale: a model that knew each pair's true landing probability would score "
        f"F1 {ceiling['f1']:.3f} / AUC {ceiling['roc_auc']:.3f}. Labels are Bernoulli draws, so "
        f"that — not 1.0 — is the ceiling.",
        "",
        "## Does it hold up",
        "",
        f"- **{sweep['n_seeds']}-seed split sweep** — scorer {sweep['scorer_f1_mean']:.3f} "
        f"± {sweep['scorer_f1_sd']:.3f} vs baseline {sweep['baseline_f1_mean']:.3f} "
        f"± {sweep['baseline_f1_sd']:.3f}; mean delta {sweep['mean_delta']:+.3f}; "
        f"scorer wins {sweep['scorer_wins']}/{sweep['n_seeds']} splits.",
        f"- **Cold-start cohort** (n={cold['n']}, people never seen in training) — "
        f"scorer F1 {cold['scorer']['f1']:.3f} vs baseline F1 {cold['baseline']['f1']:.3f} "
        f"({cold['delta_f1']:+.3f}).",
        "",
        "## What the scorer learned",
        "",
        "| feature | standardised weight |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | {weight:+.3f} |" for name, weight in model.coefficients()]

    lines += ["", "## Pioneer fine-tune", ""]
    if job:
        lines += [
            f"- Job `{job['job_id']}` · base model `{job['base_model']}` · LoRA · "
            f"dataset `{job['dataset']}` · status `{job['status']}`",
            f"- Pioneer-reported metrics: `{json.dumps(job['metrics'])[:300]}`",
        ]
    else:
        lines += [
            "Not run for these numbers — no `PIONEER_API_KEY` in this environment, so the table",
            "above comes from the local logistic head. The Pioneer path is wired and runnable",
            "(`--backend pioneer`): it uploads the same pairs as a classification dataset, LoRA-",
            "fine-tunes `fastino/gliner2-base-v1`, and scores the same held-out split through",
            "`POST /inference`. `data/pioneer_train.preview.jsonl` is the exact upload format.",
        ]

    lines += [
        "",
        "## Caveats",
        "",
        f"- {payload['dataset']['n_test']} test rows is small. Read the CI, not just the point estimate.",
        "- Labels are mock. The generator encodes the product thesis — ask/offer fit and shared",
        "  trajectory dominate, shared topic is a weak positive — calibrated so the base landing",
        "  rate is ~40%, matching the 41% the rest of the repo quotes, with the best decile of",
        "  pairs at ~96% and the worst at ~8%. So this measures whether the scorer *recovers that",
        "  structure* from 200 noisy labels, not whether the thesis is true of real users. Re-run",
        "  against the loop's real landings to answer that.",
        "- The scorer reads structured profile fields; the baseline reads profile text only. That",
        "  gap is the finding, not a handicap: `cos_bio` is feature 0, so the scorer strictly",
        "  contains the baseline's signal and adds direction (who asks vs who offers) and gaps",
        "  (stage, seniority) that a similarity score cannot represent.",
        "",
    ]
    paths.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    paths.REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
