"""One-participant cross-fitted human EEG analysis runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from jacaccess.config import configuration_hash, load_yaml
from jacaccess.jacobian.chunked import (
    MetricComputationConfig,
    calculate_metric_chunk,
    process_metric_partitions,
)
from jacaccess.jacobian.metrics import GeometryMetrics, fit_access_baseline
from jacaccess.latent.folds import assign_folds, split_fold
from jacaccess.latent.inputs import build_physical_inputs
from jacaccess.latent.pca import fit_pca_whitening
from jacaccess.latent.readouts import fit_ridge_readout


def _aligned_table(path: Path, trial_ids: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    table["original_trial_id"] = table["original_trial_id"].astype(str)
    if table["original_trial_id"].duplicated().any():
        raise ValueError(f"duplicate trial IDs in {path}")
    aligned = table.set_index("original_trial_id").reindex(trial_ids)
    if aligned.isna().all(axis=1).any():
        raise ValueError(f"{path} does not cover every accepted trial")
    return aligned.reset_index()


def _strata(condition: pd.DataFrame) -> list[str]:
    excluded = {"dataset_id", "participant_id", "original_trial_id"}
    fields = [name for name in condition.columns if name not in excluded]
    if not fields:
        raise ValueError("condition table has no fields for balanced fold allocation")
    return condition[fields].astype(str).agg("|".join, axis=1).tolist()


def _numeric_physical_inputs(
    physical: pd.DataFrame,
    times: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    excluded = {
        "dataset_id",
        "participant_id",
        "original_trial_id",
        "event_type",
        "onset_seconds",
    }
    onset_fields: dict[str, np.ndarray | float | None] = {
        "stimulus_onset": np.zeros(len(physical), dtype=np.float64)
    }
    for name in physical.columns:
        if name.endswith("_onset_seconds") and name not in excluded:
            onset_fields[name] = pd.to_numeric(physical[name], errors="coerce").to_numpy()
    covariates: dict[str, np.ndarray] = {}
    for name in physical.columns:
        if name in excluded or name in onset_fields:
            continue
        numeric = pd.to_numeric(physical[name], errors="coerce")
        if numeric.notna().all():
            covariates[name] = numeric.to_numpy(dtype=np.float32)
        else:
            levels = sorted(physical[name].astype(str).unique())
            for level in levels[1:]:
                covariates[f"{name}__{level}"] = (
                    physical[name].astype(str).to_numpy() == level
                ).astype(np.float32)
    result = build_physical_inputs(
        times_seconds=times,
        trial_count=len(physical),
        impulse_onsets_seconds=onset_fields,
        trial_covariates=covariates,
        time_basis_count=8,
    )
    return result.values, result.feature_names


def _channel_groups(config: dict[str, object], channel_count: int) -> dict[str, np.ndarray]:
    raw = config.get("output_channel_groups")
    if not isinstance(raw, dict) or len(raw) < 2:
        raise ValueError(
            "verified dataset config must define at least two output_channel_groups "
            "as zero-based channel-index lists"
        )
    groups = {str(name): np.asarray(indices, dtype=int) for name, indices in raw.items()}
    for name, indices in groups.items():
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= channel_count):
            raise ValueError(f"invalid channel indices for output group {name!r}")
    return groups


def _sanitize_sensor_channels(sensor: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Handle only fully missing sensor channels before training-fold PCA.

    A channel with no finite sample cannot contribute to PCA or readout fitting.
    Replacing such a channel by zero preserves the source channel index contract
    for verified output groups while giving it exactly zero variance.  Partially
    missing channels are refused rather than imputed, since their handling would
    require a source- and dataset-specific scientific decision.
    """
    if sensor.ndim != 3:
        raise ValueError("sensor must have trial, time, and channel axes")
    finite = np.isfinite(sensor)
    any_finite = finite.any(axis=(0, 1))
    all_finite = finite.all(axis=(0, 1))
    fully_missing = np.flatnonzero(~any_finite)
    partially_missing = np.flatnonzero(any_finite & ~all_finite)
    if partially_missing.size:
        raise ValueError(
            "sensor contains partially non-finite channels: "
            + ", ".join(str(int(index)) for index in partially_missing)
        )
    if fully_missing.size:
        sensor = np.array(sensor, copy=True)
        sensor[..., fully_missing] = 0.0
    if not np.isfinite(sensor).all():
        raise ValueError("sensor contains non-finite values after sanitization")
    return sensor, [int(index) for index in fully_missing]


def _training_baseline_chunked(
    *,
    model: object,
    latent: np.ndarray,
    inputs: np.ndarray,
    output_maps: dict[str, np.ndarray],
    residual_sd: dict[str, np.ndarray],
    baseline_mask: np.ndarray,
    metric_config: MetricComputationConfig,
    device: str,
) -> object:
    collected: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(latent), metric_config.trial_chunk):
        stop = min(start + metric_config.trial_chunk, len(latent))
        values = calculate_metric_chunk(
            model=model,
            latent_states=latent[start:stop],
            physical_inputs=inputs[start:stop],
            output_maps=output_maps,
            residual_standard_deviations=residual_sd,
            baseline=None,
            config=metric_config,
            device=device,
        )
        for name, value in values.items():
            collected.setdefault(name, []).append(value)
    merged = {name: np.concatenate(parts) for name, parts in collected.items()}
    dummy = np.empty((*merged["gain"].shape, 0, 0), dtype=np.float32)
    geometry = GeometryMetrics(
        gain=merged["gain"],
        broadcast=merged["broadcast"],
        persistence=merged["persistence"],
        concentration=merged["concentration"],
        effective_rank=merged["effective_rank"],
        top_subspace=dummy,
    )
    return fit_access_baseline(geometry, baseline_mask, metric_config.epsilon)


def run_human_fold(
    *,
    dataset_id: str,
    participant_id: str,
    fold: int,
    preprocessed_directory: Path,
    output_directory: Path,
    dataset_config_path: Path,
    analysis_config_path: Path,
    model_config_path: Path,
    device: str = "cuda",
) -> dict[str, object]:
    import torch

    from jacaccess.latent.train import (
        TrainingConfig,
        evaluate_model_qc,
        fit_residual_dynamics,
    )

    dataset_config = load_yaml(dataset_config_path)
    analysis = load_yaml(analysis_config_path)
    model_config = load_yaml(model_config_path)
    if dataset_config.get("adapter_status") != "verified":
        raise RuntimeError("human production runner requires a verified dataset adapter")

    # MNE storage is trial x channel x time; all latent code uses trial x time x feature.
    epochs = np.load(preprocessed_directory / "epochs.npy", mmap_mode="r")
    sensor = np.transpose(np.asarray(epochs), (0, 2, 1))
    sensor, fully_missing_channels = _sanitize_sensor_channels(sensor)
    times = np.load(preprocessed_directory / "time_seconds.npy")
    trial_ids = json.loads(
        (preprocessed_directory / "trial_ids.json").read_text(encoding="utf-8")
    )
    physical = _aligned_table(preprocessed_directory / "physical_events.tsv", trial_ids)
    condition = _aligned_table(preprocessed_directory / "condition_table.tsv", trial_ids)
    crossfit = analysis["crossfit"]
    assignments = assign_folds(
        dataset_id,
        participant_id,
        trial_ids,
        _strata(condition),
        folds=int(crossfit["folds"]),
        seed=int(crossfit["fold_seed"]),
    )
    split = split_fold(
        dataset_id,
        participant_id,
        trial_ids,
        assignments,
        fold,
        validation_fraction=float(crossfit["validation_fraction_of_training"]),
        seed=int(crossfit["fold_seed"]),
    )
    del condition

    latent_dimensions = min(int(crossfit["latent_dimensions"]), sensor.shape[-1])
    pca = fit_pca_whitening(sensor[split.train], components=latent_dimensions)
    latent = pca.transform(sensor).astype(np.float32)
    inputs, input_names = _numeric_physical_inputs(physical, times)
    groups = _channel_groups(dataset_config, sensor.shape[-1])
    output_maps: dict[str, np.ndarray] = {}
    residual_sd: dict[str, np.ndarray] = {}
    flattened_latent = latent[split.train].reshape(-1, latent_dimensions)
    for name, channels in groups.items():
        targets = sensor[split.train][:, :, channels].reshape(-1, len(channels))
        readout = fit_ridge_readout(name, flattened_latent, targets)
        output_maps[name] = readout.weight
        residual_sd[name] = readout.residual_standard_deviation

    training = TrainingConfig(
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        batch_transitions=int(model_config["batch_transitions"]),
        max_epochs=int(model_config["max_epochs"]),
        patience=int(model_config["patience"]),
        gradient_clip_norm=float(model_config["gradient_clip_norm"]),
        rollout_4_weight=float(model_config["rollout_loss_weights"]["four_step"]),
        rollout_8_weight=float(model_config["rollout_loss_weights"]["eight_step"]),
        stability_weight=float(model_config["stability_penalty_weight"]),
        stability_threshold=float(model_config["stability_spectral_threshold"]),
        power_iterations=int(model_config["stability_power_iterations"]),
        seed=int(crossfit["fold_seed"]) + fold,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    fitted = fit_residual_dynamics(
        training_states=torch.from_numpy(latent[split.train]),
        training_inputs=torch.from_numpy(inputs[split.train]),
        validation_states=torch.from_numpy(latent[split.validation]),
        validation_inputs=torch.from_numpy(inputs[split.validation]),
        hidden_dimensions=int(model_config["hidden_dimensions"]),
        alpha=float(model_config["alpha"]),
        config=training,
        checkpoint_path=output_directory / "training-checkpoint.pt",
        device=device,
    )
    qc = evaluate_model_qc(
        fitted.model,
        torch.from_numpy(latent[split.test]),
        torch.from_numpy(inputs[split.test]),
        device=device,
    )
    jacobian = analysis["jacobian"]
    metric_config = MetricComputationConfig(
        horizons=tuple(int(value) for value in jacobian["horizons_samples"]),
        rank=int(jacobian["rank"]),
        persistence_lag=int(jacobian["persistence_lag_samples"]),
        epsilon=float(jacobian["metric_epsilon"]),
        trial_chunk=int(analysis["compute"]["jacobian_trial_chunk"]),
    )
    metric_times = times[: latent.shape[1] - max(metric_config.horizons)]
    baseline_mask = (metric_times >= -0.2) & (metric_times <= 0.0)
    baseline = _training_baseline_chunked(
        model=fitted.model,
        latent=latent[split.train],
        inputs=inputs[split.train],
        output_maps=output_maps,
        residual_sd=residual_sd,
        baseline_mask=baseline_mask,
        metric_config=metric_config,
        device=device,
    )
    config_hash = configuration_hash(
        {"analysis": analysis, "model": model_config, "dataset": dataset_config}
    )
    sealed = process_metric_partitions(
        model=fitted.model,
        dataset_id=dataset_id,
        participant_id=participant_id,
        fold=fold,
        original_trial_ids=np.asarray(trial_ids)[split.test].tolist(),
        latent_states=latent[split.test],
        physical_inputs=inputs[split.test],
        time_seconds=times,
        output_maps=output_maps,
        residual_standard_deviations=residual_sd,
        baseline=baseline,
        output_directory=output_directory / "metrics",
        configuration_hash=config_hash,
        config=metric_config,
        device=device,
    )
    np.savez_compressed(
        output_directory / "fold-assets.npz",
        train=split.train,
        validation=split.validation,
        test=split.test,
        pca_mean=pca.mean,
        pca_components=pca.components,
        pca_eigenvalues=pca.eigenvalues,
        input_names=np.asarray(input_names),
    )
    # This is the first point where outcome/condition data are reintroduced.
    heldout_condition = _aligned_table(
        preprocessed_directory / "condition_table.tsv",
        np.asarray(trial_ids)[split.test].tolist(),
    )
    heldout_condition.to_csv(output_directory / "heldout-conditions.tsv", sep="\t", index=False)
    summary: dict[str, object] = {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "fold": fold,
        "configuration_sha256": config_hash,
        "sealed_metric_index": str(sealed),
        "condition_joined_after_metric_seal": True,
        "best_epoch": fitted.best_epoch,
        "best_validation_loss": fitted.best_validation_loss,
        "qc": asdict(qc),
        "sensor_sanitization": {
            "fully_missing_channels_replaced_with_zero": fully_missing_channels,
            "partially_missing_channels_allowed": False,
        },
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            run_human_fold(
                dataset_id=args.dataset,
                participant_id=args.participant,
                fold=args.fold,
                preprocessed_directory=args.preprocessed,
                output_directory=args.output,
                dataset_config_path=args.dataset_config,
                analysis_config_path=args.analysis_config,
                model_config_path=args.model_config,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
