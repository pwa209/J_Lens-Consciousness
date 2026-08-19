"""Seed-level paired inference and aggregate tables for human adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sign_flip_test(values: np.ndarray, *, permutations: int = 10000, seed: int = 20260817) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    observed = float(finite.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = permutations
    while remaining:
        count = min(2000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(finite)))
        exceed += int(np.sum((signs * finite).mean(axis=1) >= observed))
        remaining -= count
    return (exceed + 1) / (permutations + 1)


def sign_flip_two_sided(
    values: np.ndarray, *, permutations: int = 10000, seed: int = 20260817
) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    observed = abs(float(finite.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = permutations
    while remaining:
        count = min(2000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(finite)))
        exceed += int(np.sum(np.abs((signs * finite).mean(axis=1)) >= observed))
        remaining -= count
    return (exceed + 1) / (permutations + 1)


def bootstrap_interval(
    values: np.ndarray, *, resamples: int = 10000, seed: int = 20260817
) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(finite), size=(resamples, len(finite)))
    return np.quantile(finite[indices].mean(axis=1), [0.025, 0.975]).astype(float).tolist()


def _holm(records: list[dict[str, Any]], field: str = "p_value") -> None:
    valid = [(index, float(row[field])) for index, row in enumerate(records) if np.isfinite(row[field])]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * value)
        running = max(running, adjusted)
        records[index]["holm_adjusted_p"] = running


def _paired_effect(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return float(np.mean(values) / sd) if np.isfinite(sd) and sd > 0 else float("nan")


def aggregate_costs(experiment_root: Path, output: Path) -> pd.DataFrame:
    rows = []
    for path in experiment_root.glob("checkpoints/*/seed-*/fold-*/*/summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "architecture": value["architecture"],
                "seed": value["seed"],
                "outer_fold": value["outer_fold"],
                "condition": value["condition"],
                "trainable_parameters": value["trainable_parameters"],
                "total_parameters": value["total_parameters"],
                "trainable_fraction": value["trainable_fraction"],
                "relative_l2_parameter_displacement": value[
                    "relative_l2_parameter_displacement"
                ],
                "adaptation_steps": value["adaptation_steps"],
                "human_batches_seen": value["adaptation_steps"],
                "wall_time_seconds": value["wall_time_seconds"],
                "neural_alignment_loss_before": value["neural_alignment_loss_before"],
                "neural_alignment_loss_after": value["neural_alignment_loss_after"],
                "task_accuracy_before": value["task_accuracy_before"],
                "task_accuracy_after": value["task_accuracy_after"],
                "accuracy_change": value["accuracy_change"],
                "performance_gate_passed": value["performance_gate_passed"],
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def aggregate_interventions(experiment_root: Path, original_root: Path, output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in original_root.glob("*/seed-*/intervention.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "architecture": value["architecture"],
                "seed": value["seed"],
                "outer_fold": -1,
                "stage": "task_trained",
                "targeted_drop": value["top_subspace_accuracy_drop"],
                "random_drop": value["random_drop_mean"],
                "causal_specificity": value["top_subspace_accuracy_drop"]
                - value["random_drop_mean"],
            }
        )
    for path in experiment_root.glob("stage-analysis/*/*/seed-*/fold-*/intervention.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        stage = path.parts[-5]
        fold = int(path.parts[-2].split("-")[-1])
        rows.append(
            {
                "architecture": value["architecture"],
                "seed": value["seed"],
                "outer_fold": fold,
                "stage": stage,
                "targeted_drop": value["top_subspace_accuracy_drop"],
                "random_drop": value["random_drop_mean"],
                "causal_specificity": value["top_subspace_accuracy_drop"]
                - value["random_drop_mean"],
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def run_statistics(experiment_root: Path, output_root: Path) -> dict[str, Any]:
    aggregate = experiment_root / "aggregate"
    stages = pd.read_csv(aggregate / "fold-stage-summary.csv")
    pivot = stages.pivot_table(
        index=["architecture", "seed", "outer_fold"],
        columns="stage",
        values="rms_distance",
    ).reset_index()
    pivot["alignment_gain"] = pivot["task_trained"] - pivot["human_adapted"]
    pivot["sham_gain"] = pivot["task_trained"] - pivot["sham_adapted"]
    pivot["human_specific_gain"] = pivot["sham_adapted"] - pivot["human_adapted"]
    pivot.to_csv(aggregate / "sham-comparison.csv", index=False)
    seed = (
        pivot.groupby(["architecture", "seed"], as_index=False)[
            ["alignment_gain", "sham_gain", "human_specific_gain"]
        ].mean()
    )
    primary: list[dict[str, Any]] = []
    sham: list[dict[str, Any]] = []
    for architecture, group in seed.groupby("architecture", sort=True):
        for column, target in (("alignment_gain", primary), ("human_specific_gain", sham)):
            values = group[column].to_numpy(float)
            target.append(
                {
                    "architecture": architecture,
                    "endpoint": column,
                    "seeds": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "iqr": np.quantile(values, [0.25, 0.75]).astype(float).tolist(),
                    "paired_standardized_effect": _paired_effect(values),
                    "confidence_interval_95": bootstrap_interval(values),
                    "p_value": sign_flip_test(values),
                }
            )
    _holm(primary)
    _holm(sham)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "primary.json").write_text(
        json.dumps({"endpoint": "held-out RMS alignment gain", "tests": primary}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_root / "sham-tests.json").write_text(
        json.dumps({"endpoint": "human-specific gain", "tests": sham}, indent=2) + "\n",
        encoding="utf-8",
    )
    transfer = pd.read_csv(aggregate / "external-transfer.csv")
    transfer_tests: list[dict[str, Any]] = []
    if len(transfer):
        transfer_wide = transfer.pivot_table(
            index=["architecture", "seed", "outer_fold", "evaluation_contrast"],
            columns="stage",
            values="rms_distance",
        ).reset_index()
        transfer_wide["alignment_gain"] = (
            transfer_wide["task_trained"] - transfer_wide["human_adapted"]
        )
        for (architecture, contrast), group in transfer_wide.groupby(
            ["architecture", "evaluation_contrast"], sort=True
        ):
            values = group.groupby("seed")["alignment_gain"].mean().to_numpy(float)
            transfer_tests.append(
                {
                    "architecture": architecture,
                    "evaluation_contrast": contrast,
                    "mean_alignment_gain": float(np.mean(values)),
                    "confidence_interval_95": bootstrap_interval(values),
                    "p_value": sign_flip_test(values),
                }
            )
    (output_root / "transfer-tests.json").write_text(
        json.dumps({"tests": transfer_tests}, indent=2) + "\n", encoding="utf-8"
    )
    costs = aggregate_costs(experiment_root, aggregate / "adaptation-cost.csv")
    costs[
        [
            "architecture",
            "seed",
            "outer_fold",
            "condition",
            "task_accuracy_before",
            "task_accuracy_after",
            "accuracy_change",
            "performance_gate_passed",
        ]
    ].to_csv(aggregate / "task-retention.csv", index=False)
    interventions = aggregate_interventions(
        experiment_root, Path("results/machine"), aggregate / "post-adaptation-interventions.csv"
    )
    intervention_tests: list[dict[str, Any]] = []
    if len(interventions):
        trained = interventions[interventions["stage"].eq("task_trained")][
            ["architecture", "seed", "causal_specificity"]
        ].rename(columns={"causal_specificity": "task_trained"})
        adapted = (
            interventions[interventions["stage"].eq("human_adapted")]
            .groupby(["architecture", "seed"], as_index=False)["causal_specificity"]
            .mean()
            .rename(columns={"causal_specificity": "human_adapted"})
        )
        paired = trained.merge(adapted, on=["architecture", "seed"])
        for architecture, group in paired.groupby("architecture"):
            values = (group["human_adapted"] - group["task_trained"]).to_numpy(float)
            intervention_tests.append(
                {
                    "architecture": architecture,
                    "mean_causal_specificity_change": float(np.mean(values)),
                    "confidence_interval_95": bootstrap_interval(values),
                    "p_value": sign_flip_test(values),
                }
            )
    (output_root / "intervention-tests.json").write_text(
        json.dumps({"tests": intervention_tests}, indent=2) + "\n", encoding="utf-8"
    )
    adapted_stage = stages[stages["stage"].eq("human_adapted")]
    adapted_seed = adapted_stage.groupby(["architecture", "seed"], as_index=False).agg(
        heldout_rms_distance=("rms_distance", "mean"),
        heldout_cosine_similarity=("cosine_similarity", "mean"),
    )
    cost_seed = (
        costs[costs["condition"].eq("human_adapted")]
        .groupby(["architecture", "seed"], as_index=False)
        .agg(
            relative_l2_parameter_displacement=(
                "relative_l2_parameter_displacement",
                "mean",
            ),
            task_accuracy_change=("accuracy_change", "mean"),
        )
    )
    architecture_seed = adapted_seed.merge(cost_seed, on=["architecture", "seed"]).merge(
        seed[["architecture", "seed", "human_specific_gain"]],
        on=["architecture", "seed"],
    )
    architecture_summary = (
        architecture_seed.groupby("architecture", as_index=False)
        .agg(
            heldout_rms_distance=("heldout_rms_distance", "mean"),
            heldout_cosine_similarity=("heldout_cosine_similarity", "mean"),
            human_specific_gain=("human_specific_gain", "mean"),
            relative_l2_parameter_displacement=(
                "relative_l2_parameter_displacement",
                "mean",
            ),
            task_accuracy_change=("task_accuracy_change", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values("heldout_rms_distance")
    )
    pairwise = []
    architecture_names = sorted(architecture_seed["architecture"].unique())
    for left_index, left in enumerate(architecture_names):
        for right in architecture_names[left_index + 1 :]:
            joined = architecture_seed[architecture_seed["architecture"].eq(left)].merge(
                architecture_seed[architecture_seed["architecture"].eq(right)],
                on="seed",
                suffixes=("_left", "_right"),
            )
            difference = (
                joined["heldout_rms_distance_left"]
                - joined["heldout_rms_distance_right"]
            ).to_numpy(float)
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "endpoint": "human_adapted_heldout_rms_distance_left_minus_right",
                    "mean_difference": float(np.mean(difference)),
                    "confidence_interval_95": bootstrap_interval(difference),
                    "two_sided_p_value": sign_flip_two_sided(difference),
                    "seeds": len(difference),
                }
            )
    (output_root / "architecture-comparisons.json").write_text(
        json.dumps(
            {
                "all_prespecified_endpoints": architecture_summary.to_dict(orient="records"),
                "pairwise_primary_distance_tests": pairwise,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "primary_tests": len(primary),
        "sham_tests": len(sham),
        "adaptation_runs": len(costs),
        "performance_gate_failures": int((~costs["performance_gate_passed"]).sum()),
        "intervention_rows": len(interventions),
        "transfer_tests": len(transfer_tests),
        "architecture_pairwise_tests": len(pairwise),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_statistics(args.experiment_root, args.output), indent=2))


if __name__ == "__main__":
    main()
