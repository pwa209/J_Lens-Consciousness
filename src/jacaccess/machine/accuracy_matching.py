"""Prespecified machine validation-accuracy matching and weighted fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jacaccess.config import load_yaml

ARCHITECTURES = (
    "feedforward", "recurrent", "shared_workspace", "private_modules",
    "unlimited_shared_state",
)
PERFORMANCE_COLUMNS = {
    "architecture",
    "seed",
    "split",
    "difficulty_bin",
    "correct_count",
    "sample_count",
    "presence_accuracy",
}


def _validate_performance(table: pd.DataFrame, split: str) -> None:
    missing = PERFORMANCE_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{split} performance table lacks {sorted(missing)}")
    if set(table["split"]) != {split}:
        raise ValueError(f"expected only split={split!r}")
    keys = ["architecture", "seed", "difficulty_bin"]
    if table.duplicated(keys).any():
        raise ValueError(f"duplicate {split} architecture/seed/bin rows")
    if (table["sample_count"] <= 0).any():
        raise ValueError(f"{split} performance contains an empty bin")
    reconstructed = table["correct_count"] / table["sample_count"]
    if not np.allclose(reconstructed, table["presence_accuracy"], atol=1e-12):
        raise ValueError(f"{split} accuracy differs from correct_count/sample_count")


def summarize_accuracy_match(
    validation: pd.DataFrame,
    *,
    tolerance: float,
    target_accuracy: float,
) -> tuple[pd.DataFrame, int | None]:
    """Return architecture means by bin and the deterministic common bin."""

    _validate_performance(validation, "validation")
    grouped = (
        validation.groupby(["architecture", "difficulty_bin"], as_index=False)
        .agg(correct_count=("correct_count", "sum"), sample_count=("sample_count", "sum"))
    )
    grouped["architecture_mean_accuracy"] = grouped["correct_count"] / grouped["sample_count"]
    rows: list[dict[str, object]] = []
    for difficulty_bin, values in grouped.groupby("difficulty_bin", sort=True):
        by_architecture = {
            str(row.architecture): float(row.architecture_mean_accuracy)
            for row in values.itertuples()
        }
        missing = set(ARCHITECTURES) - by_architecture.keys()
        if missing:
            raise ValueError(
                f"difficulty bin {difficulty_bin} lacks architectures {sorted(missing)}"
            )
        accuracies = np.asarray([by_architecture[name] for name in ARCHITECTURES])
        spread = float(accuracies.max() - accuracies.min())
        rows.append(
            {
                "difficulty_bin": int(difficulty_bin),
                **{
                    f"{architecture}_mean_accuracy": by_architecture[architecture]
                    for architecture in ARCHITECTURES
                },
                "grand_mean_accuracy": float(accuracies.mean()),
                "architecture_accuracy_spread": spread,
                "within_tolerance": bool(spread <= tolerance + 1e-12),
                "distance_from_target": float(abs(accuracies.mean() - target_accuracy)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("difficulty_bin").reset_index(drop=True)
    eligible = summary[summary["within_tolerance"]].sort_values(
        ["distance_from_target", "difficulty_bin"], kind="stable"
    )
    selected = None if eligible.empty else int(eligible.iloc[0]["difficulty_bin"])
    summary["selected_common_bin"] = summary["difficulty_bin"].eq(selected)
    return summary, selected


def stabilized_ipw_table(validation: pd.DataFrame) -> pd.DataFrame:
    """Weights correct/incorrect trials to a pooled bin-specific target rate."""

    _validate_performance(validation, "validation")
    table = validation.copy()
    architecture_rates = (
        table.groupby(["architecture", "difficulty_bin"], as_index=False)
        .agg(correct_count=("correct_count", "sum"), sample_count=("sample_count", "sum"))
    )
    architecture_rates["architecture_accuracy"] = (
        architecture_rates["correct_count"] / architecture_rates["sample_count"]
    )
    target = (
        architecture_rates.groupby("difficulty_bin", as_index=False)["architecture_accuracy"]
        .mean()
        .rename(columns={"architecture_accuracy": "pooled_target_accuracy"})
    )
    result = table.merge(target, on="difficulty_bin", validate="many_to_one")
    probability = result["presence_accuracy"].clip(1e-4, 1 - 1e-4)
    target_probability = result["pooled_target_accuracy"].clip(1e-4, 1 - 1e-4)
    result["weight_if_correct"] = target_probability / probability
    result["weight_if_incorrect"] = (1 - target_probability) / (1 - probability)
    return result[
        [
            "architecture",
            "seed",
            "difficulty_bin",
            "presence_accuracy",
            "pooled_target_accuracy",
            "weight_if_correct",
            "weight_if_incorrect",
        ]
    ].sort_values(["architecture", "seed", "difficulty_bin"])


def _load_run_tables(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"*/seed-*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"no {filename} files under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def audit_accuracy_matching(
    root: Path,
    output_directory: Path,
    *,
    expected_seeds: int,
    expected_bins: int,
    tolerance: float,
    target_accuracy: float,
) -> dict[str, Any]:
    validation = _load_run_tables(root, "validation-presence-by-bin.parquet")
    test = _load_run_tables(root, "test-presence-by-bin.parquet")
    _validate_performance(validation, "validation")
    _validate_performance(test, "test")
    expected = {
        (architecture, seed, difficulty_bin)
        for architecture in ARCHITECTURES
        for seed in range(expected_seeds)
        for difficulty_bin in range(expected_bins)
    }
    validation_rows = set(
        zip(
            validation["architecture"],
            validation["seed"],
            validation["difficulty_bin"],
            strict=False,
        )
    )
    test_rows = set(
        zip(test["architecture"], test["seed"], test["difficulty_bin"], strict=False)
    )
    failures = []
    if validation_rows != expected:
        failures.append("validation architecture/seed/bin set is incomplete or unexpected")
    if test_rows != expected:
        failures.append("test architecture/seed/bin set is incomplete or unexpected")

    bin_summary, selected = summarize_accuracy_match(
        validation,
        tolerance=tolerance,
        target_accuracy=target_accuracy,
    )
    weights = stabilized_ipw_table(validation)
    test_covariates = (
        test.groupby(["architecture", "seed"], as_index=False)
        .agg(correct_count=("correct_count", "sum"), sample_count=("sample_count", "sum"))
        .sort_values(["architecture", "seed"])
    )
    test_covariates["test_presence_accuracy_covariate"] = (
        test_covariates["correct_count"] / test_covariates["sample_count"]
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    bin_summary.to_csv(output_directory / "accuracy-by-bin.csv", index=False)
    weights.to_parquet(output_directory / "fallback-ipw.parquet", index=False)
    test_covariates.to_csv(output_directory / "test-accuracy-covariates.csv", index=False)
    result: dict[str, Any] = {
        "ready": not failures,
        "mode": "common_threshold_bin" if selected is not None else "weighted_all_bins",
        "selected_threshold_bin": selected,
        "target_validation_presence_accuracy": target_accuracy,
        "maximum_architecture_mean_difference": tolerance,
        "eligible_bins": bin_summary.loc[
            bin_summary["within_tolerance"], "difficulty_bin"
        ].astype(int).tolist(),
        "fallback": (
            None
            if selected is not None
            else {
                "retain_all_threshold_trials": True,
                "weights": "fallback-ipw.parquet",
                "covariates": "test-accuracy-covariates.csv",
            }
        ),
        "failures": failures,
    }
    (output_directory / "accuracy-matching.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/machine"))
    parser.add_argument("--config", type=Path, default=Path("configs/models/machine.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    result = audit_accuracy_matching(
        args.root,
        args.output,
        expected_seeds=int(config["seeds"]),
        expected_bins=int(config["contrast_noise_levels"]),
        tolerance=float(config["accuracy_match_tolerance"]),
        target_accuracy=float(config["accuracy_match_target"]),
    )
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
