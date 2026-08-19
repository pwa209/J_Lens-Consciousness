"""Rotation-invariant Jacobian accessibility metrics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class GeometryMetrics:
    """Per-trial, per-time geometry metrics and top right-singular subspaces."""

    gain: FloatArray
    broadcast: FloatArray
    persistence: FloatArray
    concentration: FloatArray
    effective_rank: FloatArray
    top_subspace: FloatArray


@dataclass(frozen=True)
class AccessBaseline:
    """Training-baseline location and scale for Access Index components."""

    means: dict[str, float]
    standard_deviations: dict[str, float]
    epsilon: float = 1e-4


def compose_standardized_maps(
    propagators: Mapping[int, FloatArray],
    output_blocks: Mapping[str, FloatArray],
    residual_standard_deviations: Mapping[str, FloatArray],
) -> tuple[FloatArray, dict[str, slice]]:
    """Compose propagated states with standardized output maps.

    All horizons are aligned to the longest horizon's valid start-time axis.
    Within each output block, all horizon-specific maps are concatenated. This
    lets broadcast energy pool over horizons while preserving output blocks.
    """

    if not propagators:
        raise ValueError("at least one propagator is required")
    if not output_blocks:
        raise ValueError("at least one output block is required")
    if set(output_blocks) != set(residual_standard_deviations):
        raise ValueError("every output block needs residual standard deviations")

    ordered_horizons = sorted(propagators)
    first = np.asarray(propagators[ordered_horizons[0]])
    if first.ndim != 4 or first.shape[-1] != first.shape[-2]:
        raise ValueError("propagators must have shape [trial, time, state, state]")
    common_times = min(np.asarray(propagators[h]).shape[1] for h in ordered_horizons)
    n_trials, _, state_dim, _ = first.shape

    pieces: list[FloatArray] = []
    block_slices: dict[str, slice] = {}
    cursor = 0
    for name, raw_map in output_blocks.items():
        output_map = np.asarray(raw_map, dtype=first.dtype)
        residual_sd = np.asarray(residual_standard_deviations[name], dtype=first.dtype)
        if output_map.ndim != 2 or output_map.shape[1] != state_dim:
            raise ValueError(f"output block {name!r} must have shape [output, state]")
        if residual_sd.shape != (output_map.shape[0],):
            raise ValueError(f"residual SD for {name!r} has the wrong shape")
        if np.any(~np.isfinite(residual_sd)) or np.any(residual_sd <= 0):
            raise ValueError(f"residual SD for {name!r} must be positive and finite")

        standardized = output_map / residual_sd[:, None]
        standardized = standardized / np.sqrt(output_map.shape[0])
        block_parts: list[FloatArray] = []
        for horizon in ordered_horizons:
            p = np.asarray(propagators[horizon])
            if p.shape[0] != n_trials or p.shape[-2:] != (state_dim, state_dim):
                raise ValueError("propagator shapes disagree")
            block_parts.append(
                np.einsum("od,ntde->ntoe", standardized, p[:, :common_times], optimize=True)
            )
        combined_block = np.concatenate(block_parts, axis=2)
        pieces.append(combined_block)
        next_cursor = cursor + combined_block.shape[2]
        block_slices[name] = slice(cursor, next_cursor)
        cursor = next_cursor

    return np.concatenate(pieces, axis=2), block_slices


def geometry_from_maps(
    maps: FloatArray,
    block_slices: Mapping[str, slice],
    rank: int = 4,
    persistence_lag: int = 5,
    epsilon: float = 1e-4,
) -> GeometryMetrics:
    """Calculate all primary and secondary geometry metrics.

    Parameters
    ----------
    maps:
        Standardized maps shaped ``[trial, time, output_dimension, state]``.
    block_slices:
        Output-block row slices. Each slice should include that block at every
        registered horizon.
    """

    g = np.asarray(maps)
    if g.ndim != 4:
        raise ValueError("maps must have shape [trial, time, output, state]")
    if not block_slices:
        raise ValueError("block_slices cannot be empty")
    if rank < 1 or rank > min(g.shape[-2:]):
        raise ValueError("rank must not exceed the map rank bound")
    if persistence_lag < 1:
        raise ValueError("persistence_lag must be positive")

    _, singular_values, vh = np.linalg.svd(g, full_matrices=False)
    squared = singular_values**2
    total_energy = squared.sum(axis=-1)
    safe_total = np.maximum(total_energy, epsilon)

    gain = total_energy / g.shape[-2]
    concentration = squared[..., :rank].sum(axis=-1) / safe_total

    spectral_probabilities = squared / safe_total[..., None]
    effective_rank = np.exp(
        -np.sum(
            np.where(
                spectral_probabilities > 0,
                spectral_probabilities * np.log(np.maximum(spectral_probabilities, epsilon)),
                0.0,
            ),
            axis=-1,
        )
    )

    top_subspace = vh[..., :rank, :]
    projections = np.einsum("...md,...kd->...mk", g, top_subspace, optimize=True)
    block_energies = np.stack(
        [
            np.sum(projections[..., block_slice, :] ** 2, axis=(-2, -1))
            for block_slice in block_slices.values()
        ],
        axis=-1,
    )
    block_total = block_energies.sum(axis=-1)
    broadcast = np.full(block_total.shape, np.nan, dtype=g.dtype)
    valid = block_total > epsilon
    if len(block_slices) == 1:
        broadcast[valid] = 1.0
    else:
        probabilities = np.divide(
            block_energies,
            block_total[..., None],
            out=np.zeros_like(block_energies),
            where=valid[..., None],
        )
        entropy = -np.sum(
            np.where(
                probabilities > 0,
                probabilities * np.log(np.maximum(probabilities, epsilon)),
                0.0,
            ),
            axis=-1,
        )
        broadcast[valid] = entropy[valid] / np.log(len(block_slices))

    persistence = np.full(gain.shape, np.nan, dtype=g.dtype)
    if g.shape[1] > persistence_lag:
        earlier = top_subspace[:, :-persistence_lag]
        later = top_subspace[:, persistence_lag:]
        overlaps = np.matmul(earlier, np.swapaxes(later, -1, -2))
        canonical_correlations = np.linalg.svd(overlaps, compute_uv=False)
        persistence[:, :-persistence_lag] = np.mean(canonical_correlations**2, axis=-1)

    return GeometryMetrics(
        gain=gain,
        broadcast=np.clip(broadcast, 0.0, 1.0),
        persistence=np.clip(persistence, 0.0, 1.0),
        concentration=np.clip(concentration, 0.0, 1.0),
        effective_rank=effective_rank,
        top_subspace=top_subspace,
    )


def _access_components(
    metrics: GeometryMetrics,
    epsilon: float,
) -> dict[str, FloatArray]:
    return {
        "log_gain": np.log(np.maximum(metrics.gain, epsilon)),
        "logit_broadcast": np.log(
            np.clip(metrics.broadcast, epsilon, 1.0 - epsilon)
            / (1.0 - np.clip(metrics.broadcast, epsilon, 1.0 - epsilon))
        ),
        "fisher_persistence": np.arctanh(
            np.sqrt(np.clip(metrics.persistence, 0.0, 1.0 - epsilon))
        ),
        "logit_concentration": np.log(
            np.clip(metrics.concentration, epsilon, 1.0 - epsilon)
            / (1.0 - np.clip(metrics.concentration, epsilon, 1.0 - epsilon))
        ),
    }


def fit_access_baseline(
    metrics: GeometryMetrics,
    baseline_mask: NDArray[np.bool_],
    epsilon: float = 1e-4,
) -> AccessBaseline:
    """Fit Access Index normalization on training-fold baseline samples only."""

    mask = np.asarray(baseline_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != metrics.gain.shape[1]:
        raise ValueError("baseline_mask must match the metric time axis")
    if not np.any(mask):
        raise ValueError("baseline_mask selects no samples")

    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for name, values in _access_components(metrics, epsilon).items():
        selected = values[:, mask]
        means[name] = float(np.nanmean(selected))
        standard_deviations[name] = float(np.nanstd(selected, ddof=1))
        if not np.isfinite(standard_deviations[name]) or standard_deviations[name] <= epsilon:
            raise ValueError(f"baseline component {name!r} has zero or invalid variance")
    return AccessBaseline(means, standard_deviations, epsilon)


def apply_access_index(
    metrics: GeometryMetrics,
    baseline: AccessBaseline,
    component_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> FloatArray:
    """Apply a training-fold baseline to held-out geometry metrics."""

    standardized = apply_standardized_components(metrics, baseline)
    names = tuple(standardized)
    weights = np.asarray(component_weights, dtype=float)
    if weights.shape != (len(names),) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(f"component_weights must contain {len(names)} non-negative values")
    weights = weights / weights.sum()
    stacked = np.stack([standardized[name] for name in names], axis=-1)
    return np.sum(stacked * weights, axis=-1)


def apply_standardized_components(
    metrics: GeometryMetrics,
    baseline: AccessBaseline,
) -> dict[str, FloatArray]:
    """Return each training-baseline-standardized primary geometry component."""

    components = _access_components(metrics, baseline.epsilon)
    names = (
        "log_gain",
        "logit_broadcast",
        "fisher_persistence",
        "logit_concentration",
    )
    return {
        name: (components[name] - baseline.means[name]) / baseline.standard_deviations[name]
        for name in names
    }
