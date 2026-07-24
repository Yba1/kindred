"""Classification metrics — no sklearn dependency, numpy only.

Kept deliberately small and readable so the F1 number on the slide can be
traced to the arithmetic that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: float
    threshold: float
    n: int
    positives: int

    def row(self, label: str) -> str:
        return (
            f"{label:<26} {self.f1:>7.3f} {self.precision:>10.3f} "
            f"{self.recall:>8.3f} {self.accuracy:>9.3f} {self.roc_auc:>8.3f} {self.threshold:>10.3f}"
        )


HEADER = (
    f"{'model':<26} {'F1':>7} {'precision':>10} {'recall':>8} {'accuracy':>9} {'ROC-AUC':>8} {'threshold':>10}"
)


def f1_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    pred = (scores >= threshold).astype(np.float64)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC with ties handled by average ranks (Mann-Whitney U)."""
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((np.sum(ranks[y == 1]) - pos * (pos + 1) / 2.0) / (pos * neg))


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float) -> Scores:
    pred = (scores >= threshold).astype(np.float64)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return Scores(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=(tp + tn) / len(y) if len(y) else 0.0,
        roc_auc=roc_auc(y, scores),
        threshold=threshold,
        n=len(y),
        positives=int(np.sum(y == 1)),
    )


def best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 — fitted on validation data only, never on test.

    Candidates are midpoints between adjacent distinct scores, plus a point
    below the minimum so "predict everything positive" stays reachable.
    """
    uniq = np.unique(scores)
    if len(uniq) == 1:
        return float(uniq[0])
    candidates = np.concatenate(([uniq[0] - 1e-6], (uniq[:-1] + uniq[1:]) / 2.0, [uniq[-1]]))
    best, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        score = f1_at(y, scores, float(t))
        if score > best_f1:
            best_f1, best = score, float(t)
    return best


def bootstrap_delta_f1(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile CI for F1(a) - F1(b), resampling the held-out rows.

    Returns (lower_2.5%, median, upper_97.5%). Small test sets make a point
    estimate alone misleading, so the report carries this interval next to it.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    drawn = 0
    attempts = 0
    while drawn < n_boot and attempts < n_boot * 20:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if np.all(y_b == 0) or np.all(y_b == 1):
            continue  # F1 is undefined without both classes present
        deltas[drawn] = f1_at(y_b, scores_a[idx], threshold_a) - f1_at(y_b, scores_b[idx], threshold_b)
        drawn += 1
    deltas = deltas[:drawn]
    return (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 50.0)),
        float(np.percentile(deltas, 97.5)),
    )
