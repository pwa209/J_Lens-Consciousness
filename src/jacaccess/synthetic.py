"""Deterministic synthetic smoke test for the production metric formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jacaccess.jacobian.metrics import (
    apply_access_index,
    compose_standardized_maps,
    fit_access_baseline,
    geometry_from_maps,
)
from jacaccess.jacobian.propagate import ordered_propagators


def run_synthetic(seed: int = 20260730) -> dict[str, float | int]:
    """Run a small accessible/inaccessible reference simulation."""

    rng = np.random.default_rng(seed)
    n_per_condition = 40
    n_trials = 2 * n_per_condition
    n_times = 36
    state_dim = 32

    basis, _ = np.linalg.qr(rng.normal(size=(state_dim, state_dim)))
    shared = basis[:, :4]
    shared_projector = shared @ shared.T
    labels = np.repeat([1, 0], n_per_condition)

    jacobians = np.empty((n_trials, n_times, state_dim, state_dim), dtype=np.float32)
    identity = np.eye(state_dim, dtype=np.float32)
    for trial, accessible in enumerate(labels):
        for time in range(n_times):
            noise = 0.002 * rng.normal(size=(state_dim, state_dim))
            shared_gain = 0.035 if accessible else -0.015
            jacobians[trial, time] = (
                0.985 * identity + shared_gain * shared_projector + noise
            )

    propagators = ordered_propagators(jacobians, horizons=(2, 4, 8, 16))
    output_blocks: dict[str, np.ndarray] = {}
    residual_sd: dict[str, np.ndarray] = {}
    for block_index in range(6):
        local = basis[:, 4 + block_index]
        output_map = np.stack(
            [
                0.60 * shared[:, block_index % 4] + 0.20 * local,
                0.45 * shared[:, (block_index + 1) % 4] + 0.25 * local,
            ],
            axis=0,
        ).astype(np.float32)
        name = f"region_{block_index + 1}"
        output_blocks[name] = output_map
        residual_sd[name] = np.ones(2, dtype=np.float32)

    maps, block_slices = compose_standardized_maps(
        propagators,
        output_blocks,
        residual_sd,
    )
    metrics = geometry_from_maps(
        maps,
        block_slices,
        rank=4,
        persistence_lag=5,
    )
    baseline_mask = np.zeros(metrics.gain.shape[1], dtype=bool)
    baseline_mask[:8] = True
    baseline = fit_access_baseline(metrics, baseline_mask)
    access_index = apply_access_index(metrics, baseline)

    analysis_time = 10
    accessible = labels == 1
    inaccessible = labels == 0
    return {
        "seed": seed,
        "trials": n_trials,
        "state_dimensions": state_dim,
        "gain_difference": float(
            np.nanmean(metrics.gain[accessible, analysis_time])
            - np.nanmean(metrics.gain[inaccessible, analysis_time])
        ),
        "broadcast_difference": float(
            np.nanmean(metrics.broadcast[accessible, analysis_time])
            - np.nanmean(metrics.broadcast[inaccessible, analysis_time])
        ),
        "access_index_difference": float(
            np.nanmean(access_index[accessible, analysis_time])
            - np.nanmean(access_index[inaccessible, analysis_time])
        ),
        "rotation_reference_gain": float(np.nanmean(metrics.gain)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    summary = run_synthetic(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

