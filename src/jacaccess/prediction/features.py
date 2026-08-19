"""Trial-level conventional EEG and Jacobian feature construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def window_mask(times: np.ndarray, window_ms: tuple[float, float]) -> np.ndarray:
    low, high = (value / 1000.0 for value in window_ms)
    mask = (times >= low) & (times <= high)
    if not mask.any():
        raise ValueError(f"window {window_ms} selects no samples")
    return mask


def conventional_eeg_features(
    epochs: np.ndarray,
    times: np.ndarray,
    windows_ms: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Return outcome-blind ERP/GFP/power summaries per trial."""

    if epochs.ndim != 3:
        raise ValueError("epochs must be trial x channel x time")
    rows: dict[str, np.ndarray] = {}
    for label, window in windows_ms.items():
        selected = epochs[:, :, window_mask(times, window)]
        mean_channel = selected.mean(axis=-1)
        rows[f"erp_mean__{label}"] = mean_channel.mean(axis=-1)
        rows[f"gfp__{label}"] = mean_channel.std(axis=-1)
        rows[f"power__{label}"] = np.mean(selected**2, axis=(1, 2))
    return pd.DataFrame(rows)


def jacobian_features(
    metric_table: pd.DataFrame,
    windows_ms: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    required = {"original_trial_id", "time_seconds", "access_index"}
    if required - set(metric_table):
        raise ValueError(f"metric table lacks {sorted(required - set(metric_table))}")
    metrics = [
        name
        for name in (
            "gain",
            "broadcast",
            "persistence",
            "concentration",
            "effective_rank",
            "access_index",
        )
        if name in metric_table
    ]
    records: list[pd.DataFrame] = []
    for label, window in windows_ms.items():
        low, high = (value / 1000 for value in window)
        selected = metric_table[
            metric_table["time_seconds"].between(low, high, inclusive="both")
        ]
        aggregate = selected.groupby("original_trial_id", sort=False)[metrics].mean()
        aggregate.columns = [f"jacobian_{name}__{label}" for name in aggregate.columns]
        records.append(aggregate)
    if not records:
        raise ValueError("no feature windows were provided")
    return pd.concat(records, axis=1).reset_index()


def assemble_feature_table(
    *,
    epochs_path: Path,
    times_path: Path,
    trial_ids_path: Path,
    metric_paths: list[Path],
    condition_path: Path,
    output: Path,
    windows_ms: dict[str, tuple[float, float]],
    outcome_source: str | None = None,
) -> None:
    import json

    epochs = np.load(epochs_path, mmap_mode="r")
    times = np.load(times_path)
    trial_ids = json.loads(trial_ids_path.read_text(encoding="utf-8"))
    conventional = conventional_eeg_features(np.asarray(epochs), times, windows_ms)
    conventional.insert(0, "original_trial_id", trial_ids)
    metrics = pd.concat([pd.read_parquet(path) for path in metric_paths], ignore_index=True)
    # Cross-fitted partitions cover each trial once; fail rather than silently average duplicates.
    duplicated = metrics.duplicated(["original_trial_id", "time_seconds"])
    if duplicated.any():
        raise ValueError("cross-fitted metric parts overlap on trial/time keys")
    jacobian = jacobian_features(metrics, windows_ms)
    conditions = pd.read_csv(condition_path, sep="\t")
    features = conventional.merge(jacobian, on="original_trial_id", validate="one_to_one")
    features = features.merge(conditions, on="original_trial_id", validate="one_to_one")
    if outcome_source is not None:
        if outcome_source not in features:
            raise ValueError(f"prediction outcome {outcome_source!r} is absent")
        features["outcome"] = features[outcome_source]
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--times", type=Path, required=True)
    parser.add_argument("--trial-ids", type=Path, required=True)
    metric_source = parser.add_mutually_exclusive_group(required=True)
    metric_source.add_argument("--metrics", type=Path, nargs="+")
    metric_source.add_argument("--metric-root", type=Path)
    parser.add_argument("--outcome-source")
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--window",
        action="append",
        required=True,
        help="NAME:START_MS:STOP_MS",
    )
    args = parser.parse_args()
    windows = {
        value.split(":")[0]: (float(value.split(":")[1]), float(value.split(":")[2]))
        for value in args.window
    }
    metric_paths = (
        args.metrics
        if args.metrics is not None
        else sorted(args.metric_root.rglob("part-*.parquet"))
    )
    if not metric_paths:
        raise FileNotFoundError("no metric Parquet parts were found")
    assemble_feature_table(
        epochs_path=args.epochs,
        times_path=args.times,
        trial_ids_path=args.trial_ids,
        metric_paths=metric_paths,
        condition_path=args.conditions,
        output=args.output,
        windows_ms=windows,
        outcome_source=args.outcome_source,
    )


if __name__ == "__main__":
    main()
