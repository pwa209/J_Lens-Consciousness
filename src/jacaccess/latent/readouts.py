"""Training-fold linear output maps and residual standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class LinearReadout:
    name: str
    weight: FloatArray
    intercept: FloatArray
    residual_standard_deviation: FloatArray
    training_score: float

    def predict(self, latent: FloatArray) -> FloatArray:
        return np.asarray(latent) @ self.weight.T + self.intercept


def fit_ridge_readout(
    name: str,
    latent: FloatArray,
    targets: FloatArray,
    ridge: float = 1e-3,
) -> LinearReadout:
    x = np.asarray(latent, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("latent must have shape [sample, state]")
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("targets must share the latent sample axis")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    coefficient = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]),
        centered_x.T @ centered_y,
    ).T
    intercept = y_mean - x_mean @ coefficient.T
    prediction = x @ coefficient.T + intercept
    residual = y - prediction
    residual_sd = residual.std(axis=0, ddof=1)
    if np.any(~np.isfinite(residual_sd)) or np.any(residual_sd <= 0):
        raise ValueError("readout residual variance is zero or invalid")
    denominator = np.sum((y - y_mean) ** 2)
    r_squared = 1.0 - np.sum(residual**2) / denominator if denominator > 0 else 0.0
    return LinearReadout(
        name=name,
        weight=coefficient.astype(np.float32),
        intercept=intercept.astype(np.float32),
        residual_standard_deviation=residual_sd.astype(np.float32),
        training_score=float(r_squared),
    )


def fit_logistic_readout(
    name: str,
    latent: FloatArray,
    labels: NDArray[np.integer] | NDArray[np.str_],
    regularization_c: float = 1.0,
    seed: int = 20260730,
) -> LinearReadout:
    """Fit a training-fold multinomial logit map and return pre-softmax weights."""

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError("logistic readouts require scikit-learn") from exc
    x = np.asarray(latent)
    y = np.asarray(labels)
    model = LogisticRegression(
        C=regularization_c,
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(x, y)
    decision = model.decision_function(x)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
        weight = np.vstack([-model.coef_[0], model.coef_[0]])
        intercept = np.asarray([-model.intercept_[0], model.intercept_[0]])
    else:
        weight = model.coef_
        intercept = model.intercept_
    encoded = model.classes_.searchsorted(y)
    residual = np.eye(len(model.classes_))[encoded] - _softmax(decision)
    residual_sd = residual.std(axis=0, ddof=1)
    residual_sd = np.maximum(residual_sd, 1e-4)
    return LinearReadout(
        name=name,
        weight=np.asarray(weight, dtype=np.float32),
        intercept=np.asarray(intercept, dtype=np.float32),
        residual_standard_deviation=np.asarray(residual_sd, dtype=np.float32),
        training_score=float(model.score(x, y)),
    )


def _softmax(values: FloatArray) -> FloatArray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)

