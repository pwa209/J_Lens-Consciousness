"""Retention-gate sensitivity analyses for the human-adaptation extension.

The completed all-run analysis is preserved. This module adds two transparent
gate-aware sensitivity estimands without selecting a result-dependent rule:

1. gate-compliant folds, retaining seeds with at least three valid folds;
2. complete-case seeds, retaining only seeds with all five valid folds.

For alignment gain, the human-adapted run must pass. For the human-specific
gain over sham, both the human-adapted and sham-adapted runs must pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jacaccess.stats.adaptation_comparison import (
    _holm,
    _paired_effect,
    bootstrap_interval,
    sign_flip_test,
)


ARCHITECTURES = (
    "feedforward",
    "recurrent",
    "private_modules",
    "shared_workspace",
    "unlimited_shared_state",
)
KEYS = ["architecture", "seed", "outer_fold"]


def _seed_tests(
    table: pd.DataFrame,
    *,
    endpoint: str,
    gate: str | None,
    minimum_folds: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    selected = table.copy()
    if gate is not None:
        selected = selected[selected[gate].astype(bool)].copy()
    seed = (
        selected.groupby(["architecture", "seed"], as_index=False)
        .agg(value=(endpoint, "mean"), retained_folds=(endpoint, "count"))
    )
    seed = seed[seed["retained_folds"].ge(minimum_folds)].copy()
    records: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        values = seed.loc[seed["architecture"].eq(architecture), "value"].to_numpy(float)
        records.append(
            {
                "architecture": architecture,
                "endpoint": endpoint,
                "seeds": int(len(values)),
                "retained_folds": int(
                    seed.loc[seed["architecture"].eq(architecture), "retained_folds"].sum()
                ),
                "mean": float(np.mean(values)) if len(values) else float("nan"),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "paired_standardized_effect": _paired_effect(values),
                "confidence_interval_95": bootstrap_interval(values),
                "p_value": sign_flip_test(values),
            }
        )
    _holm(records)
    seed["endpoint"] = endpoint
    return records, seed


def run(experiment_root: Path) -> dict[str, Any]:
    aggregate = experiment_root / "aggregate"
    statistics = experiment_root / "statistics"
    stages = pd.read_csv(aggregate / "fold-stage-summary.csv")
    costs = pd.read_csv(aggregate / "adaptation-cost.csv")

    wide = stages.pivot_table(
        index=KEYS,
        columns="stage",
        values="rms_distance",
    ).reset_index()
    wide["alignment_gain"] = wide["task_trained"] - wide["human_adapted"]
    wide["human_specific_gain"] = wide["sham_adapted"] - wide["human_adapted"]

    gates = costs.pivot_table(
        index=KEYS,
        columns="condition",
        values="performance_gate_passed",
        aggfunc="first",
    ).reset_index()
    gates = gates.rename(
        columns={
            "human_adapted": "human_gate_passed",
            "sham_adapted": "sham_gate_passed",
        }
    )
    table = wide.merge(gates, on=KEYS, validate="one_to_one")
    table["both_gate_passed"] = (
        table["human_gate_passed"].astype(bool)
        & table["sham_gate_passed"].astype(bool)
    )

    scenarios = (
        ("all_runs", None, 5),
        ("gate_compliant_folds_minimum_3", "endpoint_gate", 3),
        ("complete_case_seeds", "endpoint_gate", 5),
    )
    output: dict[str, Any] = {
        "gate_definition": "absolute task-accuracy change <= 0.02",
        "inferential_unit": "seed after averaging retained participant folds",
        "scenarios": {},
    }
    seed_tables: list[pd.DataFrame] = []
    for scenario, gate_mode, minimum_folds in scenarios:
        endpoint_output: dict[str, Any] = {}
        for endpoint, endpoint_gate in (
            ("alignment_gain", "human_gate_passed"),
            ("human_specific_gain", "both_gate_passed"),
        ):
            gate = endpoint_gate if gate_mode is not None else None
            records, seed = _seed_tests(
                table,
                endpoint=endpoint,
                gate=gate,
                minimum_folds=minimum_folds,
            )
            endpoint_output[endpoint] = records
            seed.insert(0, "scenario", scenario)
            seed_tables.append(seed)
        output["scenarios"][scenario] = {
            "minimum_retained_folds_per_seed": minimum_folds,
            "tests": endpoint_output,
        }

    exclusions = costs[~costs["performance_gate_passed"].astype(bool)].copy()
    exclusions = exclusions.sort_values(["architecture", "seed", "outer_fold", "condition"])
    exclusions.to_csv(aggregate / "retention-gate-exclusions.csv", index=False)
    pd.concat(seed_tables, ignore_index=True).to_csv(
        aggregate / "retention-sensitivity-seed-estimates.csv", index=False
    )

    failure_counts = (
        exclusions.groupby(["architecture", "condition"], as_index=False)
        .size()
        .rename(columns={"size": "failed_runs"})
    )
    output["failure_counts"] = failure_counts.to_dict(orient="records")
    output["failed_runs"] = int(len(exclusions))
    output["total_runs"] = int(len(costs))
    statistics.mkdir(parents=True, exist_ok=True)
    (statistics / "retention-sensitivity.json").write_text(
        json.dumps(output, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.experiment_root)
    print(
        json.dumps(
            {
                "failed_runs": result["failed_runs"],
                "total_runs": result["total_runs"],
                "scenarios": list(result["scenarios"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
