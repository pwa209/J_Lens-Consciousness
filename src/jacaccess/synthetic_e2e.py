"""Dependency-light end-to-end fixture through sealed condition-blind metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jacaccess.io.manifest import sha256_file
from jacaccess.jacobian.metrics import (
    apply_access_index,
    compose_standardized_maps,
    fit_access_baseline,
    geometry_from_maps,
)
from jacaccess.jacobian.propagate import ordered_propagators
from jacaccess.latent.folds import assign_folds, split_fold
from jacaccess.latent.inputs import build_physical_inputs
from jacaccess.latent.pca import fit_pca_whitening
from jacaccess.latent.readouts import fit_ridge_readout


def _condition_blind_geometry(
    jacobians: np.ndarray,
    output_maps: dict[str, np.ndarray],
    residual_sd: dict[str, np.ndarray],
) -> object:
    propagators = ordered_propagators(jacobians, (2, 4, 8, 16))
    maps, slices = compose_standardized_maps(propagators, output_maps, residual_sd)
    return geometry_from_maps(maps, slices, rank=4, persistence_lag=5)


def run_synthetic_e2e(
    output_directory: Path,
    *,
    seed: int = 20260730,
    heldout_fold: int = 0,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    trials, times, sensors, state_dimensions = 80, 36, 40, 32
    trial_ids = [f"synthetic-{index:04d}" for index in range(trials)]
    condition = np.asarray([index % 2 for index in range(trials)], dtype=np.int8)
    strata = ["accessible" if value else "inaccessible" for value in condition]
    assignments = assign_folds("synthetic", "001", trial_ids, strata, seed=seed)
    split = split_fold(
        "synthetic",
        "001",
        trial_ids,
        assignments,
        heldout_fold,
        validation_fraction=0.10,
        seed=seed,
    )

    source_latent = rng.normal(size=(trials, times, state_dimensions))
    sensor_mixing = rng.normal(size=(state_dimensions, sensors))
    sensor_epochs = source_latent @ sensor_mixing + 0.05 * rng.normal(
        size=(trials, times, sensors)
    )
    pca = fit_pca_whitening(sensor_epochs[split.train], components=state_dimensions)
    latent = pca.transform(sensor_epochs).astype(np.float32)
    time_seconds = np.arange(times, dtype=np.float64) / 100.0 - 0.20
    physical = build_physical_inputs(
        times_seconds=time_seconds,
        trial_count=trials,
        impulse_onsets_seconds={"target_onset": 0.0},
        trial_covariates={"physical_contrast": np.linspace(0.1, 0.9, trials)},
        time_basis_count=8,
    )

    output_maps: dict[str, np.ndarray] = {}
    residual_sd: dict[str, np.ndarray] = {}
    flattened_train = latent[split.train].reshape(-1, state_dimensions)
    for region in range(6):
        start = region * 6
        stop = min(start + 2, sensors)
        targets = sensor_epochs[split.train, :, start:stop].reshape(-1, stop - start)
        readout = fit_ridge_readout(f"region_{region + 1}", flattened_train, targets)
        output_maps[readout.name] = readout.weight
        residual_sd[readout.name] = readout.residual_standard_deviation

    basis, _ = np.linalg.qr(rng.normal(size=(state_dimensions, state_dimensions)))
    projector = basis[:, :4] @ basis[:, :4].T
    jacobians = np.broadcast_to(
        0.985 * np.eye(state_dimensions, dtype=np.float32),
        (trials, times - 1, state_dimensions, state_dimensions),
    ).copy()
    jacobians += 0.001 * rng.normal(size=jacobians.shape).astype(np.float32)
    poststim = np.arange(times - 1) >= 20
    accessible_trials = np.flatnonzero(condition.astype(bool))
    poststim_times = np.flatnonzero(poststim)
    jacobians[np.ix_(accessible_trials, poststim_times)] += (0.030 * projector).astype(
        np.float32
    )

    training_metrics = _condition_blind_geometry(
        jacobians[split.train],
        output_maps,
        residual_sd,
    )
    metric_times = training_metrics.gain.shape[1]
    baseline_mask = np.zeros(metric_times, dtype=bool)
    baseline_mask[:10] = True
    baseline = fit_access_baseline(training_metrics, baseline_mask)
    test_metrics = _condition_blind_geometry(
        jacobians[split.test],
        output_maps,
        residual_sd,
    )
    access = apply_access_index(test_metrics, baseline).astype(np.float32)

    output_directory.mkdir(parents=True, exist_ok=True)
    sealed_path = output_directory / "sealed_metrics.npz"
    np.savez_compressed(
        sealed_path,
        original_trial_id=np.asarray(trial_ids)[split.test],
        time_seconds=time_seconds[:metric_times],
        gain=test_metrics.gain,
        broadcast=test_metrics.broadcast,
        persistence=test_metrics.persistence,
        concentration=test_metrics.concentration,
        effective_rank=test_metrics.effective_rank,
        access_index=access,
        physical_input_shape=np.asarray(physical.values.shape),
    )
    sealed_hash = sha256_file(sealed_path)
    condition_path = output_directory / "condition_table.json"
    condition_path.write_text(
        json.dumps(
            {
                trial_ids[index]: int(condition[index])
                for index in split.test
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    analysis_time = min(16, metric_times - 6)
    heldout_condition = condition[split.test].astype(bool)
    difference = float(
        np.nanmean(access[heldout_condition, analysis_time])
        - np.nanmean(access[~heldout_condition, analysis_time])
    )
    summary: dict[str, object] = {
        "seed": seed,
        "fold": heldout_fold,
        "train_trials": int(split.train.size),
        "validation_trials": int(split.validation.size),
        "test_trials": int(split.test.size),
        "physical_input_features": len(physical.feature_names),
        "sealed_metrics_sha256": sealed_hash,
        "condition_joined_after_seal": True,
        "heldout_access_index_difference": difference,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    summary = run_synthetic_e2e(args.output, seed=args.seed, heldout_fold=args.fold)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
