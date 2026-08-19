"""Participant-level directional and cluster sign-flip inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Cluster:
    start: int
    stop: int
    mass: float
    p_value: float


def directional_sign_flip(values: np.ndarray, permutations: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("at least two finite participant contrasts are required")
    observed = values.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(permutations, len(values)))
    null = (signs * values).mean(axis=1)
    return float((1 + np.sum(null >= observed)) / (permutations + 1))


def _clusters(statistic: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    selected = np.asarray(statistic) >= threshold
    changes = np.diff(np.pad(selected.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        (int(start), int(stop), float(np.sum(statistic[start:stop])))
        for start, stop in zip(starts, stops, strict=True)
    ]


def cluster_sign_flip(
    participant_timecourses: np.ndarray,
    *,
    permutations: int = 5000,
    cluster_forming_p: float = 0.01,
    seed: int = 20260730,
) -> tuple[np.ndarray, list[Cluster]]:
    values = np.asarray(participant_timecourses, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("values must be participant x time with at least two participants")
    statistic = stats.ttest_1samp(values, 0.0, axis=0, nan_policy="omit").statistic
    threshold = float(stats.t.ppf(1 - cluster_forming_p, df=values.shape[0] - 1))
    observed = _clusters(statistic, threshold)
    rng = np.random.default_rng(seed)
    maximum = np.zeros(permutations)
    for index in range(permutations):
        signed = values * rng.choice((-1.0, 1.0), size=(values.shape[0], 1))
        null_stat = stats.ttest_1samp(signed, 0.0, axis=0, nan_policy="omit").statistic
        null_clusters = _clusters(null_stat, threshold)
        maximum[index] = max((cluster[2] for cluster in null_clusters), default=0.0)
    clusters = [
        Cluster(start, stop, mass, float((1 + np.sum(maximum >= mass)) / (permutations + 1)))
        for start, stop, mass in observed
    ]
    return statistic, clusters
