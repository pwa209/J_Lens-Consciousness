"""Ordered propagation of state perturbations through local Jacobians."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def ordered_propagators(
    jacobians: FloatArray,
    horizons: Iterable[int] = (2, 4, 8, 16),
) -> dict[int, FloatArray]:
    """Return ``J[t+h-1] @ ... @ J[t]`` for every valid start.

    Parameters
    ----------
    jacobians:
        Array shaped ``[trial, time, state, state]``.
    horizons:
        Positive integer horizons in samples.

    Returns
    -------
    dict
        Each value is shaped ``[trial, time-horizon+1, state, state]``.

    Notes
    -----
    The function is a reference implementation. Production code should call it
    on trial chunks instead of materializing a complete-dataset tensor.
    """

    j = np.asarray(jacobians)
    if j.ndim != 4:
        raise ValueError("jacobians must have shape [trial, time, state, state]")
    if j.shape[-1] != j.shape[-2]:
        raise ValueError("the final two Jacobian dimensions must be square")
    if not np.issubdtype(j.dtype, np.floating):
        raise TypeError("jacobians must have a floating dtype")

    horizon_values = tuple(sorted(set(int(h) for h in horizons)))
    if not horizon_values or horizon_values[0] < 1:
        raise ValueError("horizons must contain positive integers")
    if horizon_values[-1] > j.shape[1]:
        raise ValueError("a horizon exceeds the available Jacobian time axis")

    n_trials, n_times, state_dim, _ = j.shape
    output: dict[int, FloatArray] = {}
    for horizon in horizon_values:
        n_starts = n_times - horizon + 1
        product = np.broadcast_to(
            np.eye(state_dim, dtype=j.dtype),
            (n_trials, n_starts, state_dim, state_dim),
        ).copy()
        for offset in range(horizon):
            product = np.matmul(j[:, offset : offset + n_starts], product)
        output[horizon] = product
    return output

