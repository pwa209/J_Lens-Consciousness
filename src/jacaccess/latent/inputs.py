"""Condition-blind physical inputs and fixed time bases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jacaccess.io.events import assert_primary_model_fields_safe

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class PhysicalInputs:
    values: FloatArray
    feature_names: tuple[str, ...]


def gaussian_time_basis(
    times_seconds: FloatArray,
    count: int = 8,
    width_scale: float = 1.5,
) -> FloatArray:
    times = np.asarray(times_seconds, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times_seconds must be a one-dimensional time axis")
    if count < 1:
        raise ValueError("basis count must be positive")
    centers = np.linspace(times.min(), times.max(), count)
    spacing = (times.max() - times.min()) / max(count - 1, 1)
    width = max(spacing * width_scale, np.finfo(float).eps)
    return np.exp(-0.5 * ((times[:, None] - centers[None, :]) / width) ** 2)


def build_physical_inputs(
    *,
    times_seconds: FloatArray,
    trial_count: int,
    impulse_onsets_seconds: Mapping[str, FloatArray | float | None],
    trial_covariates: Mapping[str, FloatArray],
    time_basis_count: int = 8,
) -> PhysicalInputs:
    """Build ``[trial, time, feature]`` inputs without condition labels."""

    times = np.asarray(times_seconds, dtype=np.float64)
    if trial_count < 1:
        raise ValueError("trial_count must be positive")
    feature_names = list(impulse_onsets_seconds) + list(trial_covariates)
    assert_primary_model_fields_safe(feature_names)
    pieces: list[FloatArray] = []
    names: list[str] = []

    for name, raw_onsets in impulse_onsets_seconds.items():
        impulses = np.zeros((trial_count, times.size, 1), dtype=np.float32)
        if raw_onsets is not None:
            onsets = np.asarray(raw_onsets, dtype=np.float64)
            if onsets.ndim == 0:
                onsets = np.full(trial_count, float(onsets))
            if onsets.shape != (trial_count,):
                raise ValueError(f"impulse {name!r} must be scalar or one value per trial")
            for trial, onset in enumerate(onsets):
                if np.isfinite(onset):
                    index = int(np.argmin(np.abs(times - onset)))
                    impulses[trial, index, 0] = 1.0
        pieces.append(impulses)
        names.append(name)

    for name, raw_values in trial_covariates.items():
        values = np.asarray(raw_values, dtype=np.float32)
        if values.shape != (trial_count,):
            raise ValueError(f"covariate {name!r} must have one value per trial")
        pieces.append(np.broadcast_to(values[:, None, None], (trial_count, times.size, 1)).copy())
        names.append(name)

    basis = gaussian_time_basis(times, time_basis_count).astype(np.float32)
    pieces.append(np.broadcast_to(basis[None], (trial_count, *basis.shape)).copy())
    names.extend(f"time_basis_{index + 1:02d}" for index in range(time_basis_count))
    return PhysicalInputs(values=np.concatenate(pieces, axis=-1), feature_names=tuple(names))

