"""Outcome-blind epoch artifact metrics and robust rejection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def robust_z(values: FloatArray, epsilon: float = 1e-12) -> FloatArray:
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    return 0.6744897501960817 * (values - median) / max(mad, epsilon)


def epoch_artifact_mask(
    epochs_volts: FloatArray,
    *,
    peak_to_peak_uv_max: float = 150.0,
    robust_z_max: float = 6.0,
) -> tuple[NDArray[np.bool_], dict[str, FloatArray]]:
    """Return rejection mask using amplitude, variance and kurtosis only."""

    values = np.asarray(epochs_volts, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("epochs must have shape [trial, channel, time]")
    peak_to_peak_uv = np.ptp(values, axis=-1).max(axis=-1) * 1e6
    amplitude = np.max(np.abs(values), axis=(-2, -1))
    variance = np.var(values, axis=(-2, -1))
    centered = values - values.mean(axis=-1, keepdims=True)
    second = np.mean(centered**2, axis=(-2, -1))
    fourth = np.mean(centered**4, axis=(-2, -1))
    kurtosis = fourth / np.maximum(second**2, np.finfo(float).eps)
    scores = {
        "amplitude_robust_z": robust_z(amplitude),
        "variance_robust_z": robust_z(variance),
        "kurtosis_robust_z": robust_z(kurtosis),
        "peak_to_peak_uv": peak_to_peak_uv,
    }
    reject = peak_to_peak_uv > peak_to_peak_uv_max
    for name in ("amplitude_robust_z", "variance_robust_z", "kurtosis_robust_z"):
        reject |= np.abs(scores[name]) > robust_z_max
    return reject, scores

