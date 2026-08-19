"""Confirmatory window contrasts from the condition-rejoined human table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jacaccess.stats.permutation import cluster_sign_flip, directional_sign_flip


def _condition_key(value: object) -> str:
    """Return a stable key for equivalent CLI and table condition levels."""

    if pd.isna(value):
        return "<missing>"
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def analyze_contrast(
    table: pd.DataFrame,
    *,
    dataset: str,
    condition_field: str,
    positive: str,
    negative: str,
    window_ms: tuple[float, float],
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = table[table["dataset_id"] == dataset].copy()
    selected[condition_field] = selected[condition_field].map(_condition_key)
    positive = _condition_key(positive)
    negative = _condition_key(negative)
    low, high = (value / 1000 for value in window_ms)
    windowed = selected[selected["time_seconds"].between(low, high, inclusive="both")]
    trial = (
        windowed.groupby(["participant_id", "original_trial_id", condition_field])[
            "access_index"
        ]
        .mean()
        .reset_index()
    )
    participant = (
        trial.groupby(["participant_id", condition_field])["access_index"].mean().unstack()
    )
    if positive not in participant or negative not in participant:
        raise ValueError("requested condition levels are absent")
    contrasts = (participant[positive] - participant[negative]).dropna()
    contrast_table = contrasts.rename("contrast").reset_index()
    result: dict[str, object] = {
        "dataset": dataset,
        "condition_field": condition_field,
        "positive": positive,
        "negative": negative,
        "window_ms": list(window_ms),
        "participants": len(contrasts),
        "mean_contrast": float(contrasts.mean()),
        "directional_sign_flip_p": directional_sign_flip(
            contrasts.to_numpy(), permutations, seed
        ),
    }

    timecourse = (
        selected.groupby(["participant_id", "time_seconds", condition_field])["access_index"]
        .mean()
        .unstack()
    )
    difference = (timecourse[positive] - timecourse[negative]).unstack("time_seconds")
    times = difference.columns.to_numpy(dtype=float)
    statistic, clusters = cluster_sign_flip(
        difference.to_numpy(),
        permutations=permutations,
        seed=seed,
    )
    result["time_seconds"] = times.tolist()
    result["t_statistic"] = np.asarray(statistic).tolist()
    result["clusters"] = [
        {
            "start_seconds": float(times[value.start]),
            "stop_seconds": float(times[value.stop - 1]),
            "mass": value.mass,
            "familywise_p": value.p_value,
        }
        for value in clusters
    ]
    return contrast_table, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-table", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition-field", required=True)
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--window-ms", type=float, nargs=2, required=True)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_parquet(args.human_table)
    contrasts, result = analyze_contrast(
        table,
        dataset=args.dataset,
        condition_field=args.condition_field,
        positive=args.positive,
        negative=args.negative,
        window_ms=tuple(args.window_ms),
        permutations=args.permutations,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(args.output / "participant-contrasts.csv", index=False)
    (args.output / "frequentist.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
