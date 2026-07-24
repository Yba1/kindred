"""The local scorer head: L2-regularised logistic regression, fit by IRLS.

Why not sklearn: it isn't a dependency of this repo, and at 11 features by
~120 rows Newton-IRLS converges in under ten iterations and is a page of code.
The fitted weights serialise to plain JSON, so a trained scorer is a 2 KB file
the loop can load with no ML runtime at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Split by sign so neither exp() branch overflows.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class LogisticScorer:
    """Standardises features, then applies a logistic head."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    threshold: float = 0.5
    feature_names: tuple[str, ...] = FEATURE_NAMES
    l2: float = 1.0
    backend: str = "local-logistic"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Z = (X - self.mean) / self.scale
        return sigmoid(Z @ self.weights + self.bias)

    def coefficients(self) -> list[tuple[str, float]]:
        """Standardised coefficients, largest magnitude first — the demo's 'why'."""
        return sorted(
            zip(self.feature_names, self.weights.tolist()),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "threshold": float(self.threshold),
            "l2": float(self.l2),
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "LogisticScorer":
        return cls(
            weights=np.asarray(blob["weights"], dtype=np.float64),
            bias=float(blob["bias"]),
            mean=np.asarray(blob["mean"], dtype=np.float64),
            scale=np.asarray(blob["scale"], dtype=np.float64),
            threshold=float(blob.get("threshold", 0.5)),
            feature_names=tuple(blob.get("feature_names", FEATURE_NAMES)),
            l2=float(blob.get("l2", 1.0)),
            backend=blob.get("backend", "local-logistic"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticScorer":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> LogisticScorer:
    """Newton-IRLS with an L2 penalty on the weights (bias unpenalised)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0  # constant columns pass through untouched
    Z = np.hstack([(X - mean) / scale, np.ones((len(X), 1))])

    n_params = Z.shape[1]
    beta = np.zeros(n_params, dtype=np.float64)
    penalty = np.eye(n_params) * l2
    penalty[-1, -1] = 0.0  # never shrink the intercept

    for _ in range(max_iter):
        p = sigmoid(Z @ beta)
        w = np.clip(p * (1 - p), 1e-9, None)
        gradient = Z.T @ (y - p) - penalty @ beta
        hessian = (Z.T * w) @ Z + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta += step
        if float(np.max(np.abs(step))) < tol:
            break

    return LogisticScorer(
        weights=beta[:-1],
        bias=float(beta[-1]),
        mean=mean,
        scale=scale,
        l2=l2,
    )
