"""Chunked GPU Jacobians reduced to partitioned Parquet metric tables."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from jacaccess.io.manifest import sha256_file
from jacaccess.jacobian.metrics import (
    AccessBaseline,
    apply_access_index,
    apply_standardized_components,
    compose_standardized_maps,
    fit_access_baseline,
    geometry_from_maps,
)
from jacaccess.jacobian.propagate import ordered_propagators

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class MetricComputationConfig:
    horizons: tuple[int, ...] = (2, 4, 8, 16)
    rank: int = 4
    persistence_lag: int = 5
    epsilon: float = 1e-4
    trial_chunk: int = 64


def calculate_metric_chunk(
    *,
    model: object,
    latent_states: FloatArray,
    physical_inputs: FloatArray,
    output_maps: Mapping[str, FloatArray],
    residual_standard_deviations: Mapping[str, FloatArray],
    baseline: AccessBaseline | None,
    config: MetricComputationConfig = MetricComputationConfig(),
    device: str = "cuda",
) -> dict[str, FloatArray]:
    """Calculate geometry for a trial chunk and immediately return small arrays."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("chunked Jacobian calculation requires PyTorch") from exc
    states = torch.as_tensor(latent_states, dtype=torch.float32, device=device)
    inputs = torch.as_tensor(physical_inputs, dtype=torch.float32, device=device)
    if states.shape[:2] != inputs.shape[:2]:
        raise ValueError("states and physical inputs must share trial/time axes")
    model.eval()
    with torch.no_grad():
        jacobians = model.analytic_jacobian(states[:, :-1], inputs[:, :-1])
    jacobian_numpy = jacobians.detach().cpu().numpy()
    del jacobians, states, inputs
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    propagators = ordered_propagators(jacobian_numpy, config.horizons)
    maps, block_slices = compose_standardized_maps(
        propagators,
        output_maps,
        residual_standard_deviations,
    )
    metrics = geometry_from_maps(
        maps,
        block_slices,
        rank=config.rank,
        persistence_lag=config.persistence_lag,
        epsilon=config.epsilon,
    )
    access_index = (
        np.full(metrics.gain.shape, np.nan, dtype=np.float32)
        if baseline is None
        else apply_access_index(metrics, baseline).astype(np.float32)
    )
    result = {
        "gain": metrics.gain.astype(np.float32),
        "broadcast": metrics.broadcast.astype(np.float32),
        "persistence": metrics.persistence.astype(np.float32),
        "concentration": metrics.concentration.astype(np.float32),
        "effective_rank": metrics.effective_rank.astype(np.float32),
        "access_index": access_index,
    }
    if baseline is not None:
        standardized = apply_standardized_components(metrics, baseline)
        result.update({f"z_{name}": values.astype(np.float32) for name, values in standardized.items()})
    return result


def fit_training_baseline(
    *,
    model: object,
    latent_states: FloatArray,
    physical_inputs: FloatArray,
    output_maps: Mapping[str, FloatArray],
    residual_standard_deviations: Mapping[str, FloatArray],
    baseline_time_mask: NDArray[np.bool_],
    config: MetricComputationConfig = MetricComputationConfig(),
    device: str = "cuda",
) -> AccessBaseline:
    values = calculate_metric_chunk(
        model=model,
        latent_states=latent_states,
        physical_inputs=physical_inputs,
        output_maps=output_maps,
        residual_standard_deviations=residual_standard_deviations,
        baseline=None,
        config=config,
        device=device,
    )
    # The top subspace is not required for baseline fitting, so construct the
    # metric container with an empty diagnostic axis.
    from jacaccess.jacobian.metrics import GeometryMetrics

    dummy = np.empty((*values["gain"].shape, 0, 0), dtype=np.float32)
    metrics = GeometryMetrics(
        gain=values["gain"],
        broadcast=values["broadcast"],
        persistence=values["persistence"],
        concentration=values["concentration"],
        effective_rank=values["effective_rank"],
        top_subspace=dummy,
    )
    return fit_access_baseline(metrics, baseline_time_mask, config.epsilon)


def _write_parquet_atomic(
    columns: Mapping[str, object],
    output: Path,
    metadata: Mapping[str, str],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Parquet output requires pyarrow") from exc
    table = pa.table(columns)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata.update(
        {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}
    )
    table = table.replace_schema_metadata(schema_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)


def process_metric_partitions(
    *,
    model: object,
    dataset_id: str,
    participant_id: str,
    fold: int,
    original_trial_ids: Sequence[str],
    latent_states: FloatArray,
    physical_inputs: FloatArray,
    time_seconds: FloatArray,
    output_maps: Mapping[str, FloatArray],
    residual_standard_deviations: Mapping[str, FloatArray],
    baseline: AccessBaseline,
    output_directory: Path,
    configuration_hash: str,
    config: MetricComputationConfig = MetricComputationConfig(),
    device: str = "cuda",
) -> Path:
    """Write restartable Parquet parts plus a sealed partition index."""

    if len(original_trial_ids) != latent_states.shape[0]:
        raise ValueError("trial IDs do not match the latent trial axis")
    expected_times = latent_states.shape[1] - max(config.horizons)
    if time_seconds.shape != (latent_states.shape[1],):
        raise ValueError("time_seconds does not match latent time axis")
    metric_times = np.asarray(time_seconds[:expected_times], dtype=np.float64)
    output_directory.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, object]] = []

    for part_index, start in enumerate(range(0, latent_states.shape[0], config.trial_chunk)):
        stop = min(start + config.trial_chunk, latent_states.shape[0])
        output = output_directory / f"part-{part_index:05d}.parquet"
        if output.exists():
            parts.append(
                {
                    "path": output.name,
                    "size_bytes": output.stat().st_size,
                    "sha256": sha256_file(output),
                }
            )
            continue
        metrics = calculate_metric_chunk(
            model=model,
            latent_states=latent_states[start:stop],
            physical_inputs=physical_inputs[start:stop],
            output_maps=output_maps,
            residual_standard_deviations=residual_standard_deviations,
            baseline=baseline,
            config=config,
            device=device,
        )
        rows = (stop - start) * expected_times
        columns: dict[str, object] = {
            "dataset_id": np.repeat(dataset_id, rows),
            "participant_id": np.repeat(participant_id, rows),
            "fold": np.repeat(np.int16(fold), rows),
            "original_trial_id": np.repeat(
                np.asarray(original_trial_ids[start:stop]),
                expected_times,
            ),
            "time_seconds": np.tile(metric_times, stop - start),
        }
        columns.update({name: values.reshape(-1) for name, values in metrics.items()})
        _write_parquet_atomic(
            columns,
            output,
            {
                "configuration_sha256": configuration_hash,
                "metric_config": json.dumps(asdict(config), sort_keys=True),
            },
        )
        parts.append(
            {
                "path": output.name,
                "size_bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        )

    index = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "fold": fold,
        "configuration_sha256": configuration_hash,
        "trial_count": len(original_trial_ids),
        "time_count": expected_times,
        "parts": parts,
    }
    index_path = output_directory / "partition-index.json"
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    temporary.replace(index_path)
    return index_path
