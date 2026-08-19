"""Validate and write the Experiment 2 engineering/scientific digest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    stages = pd.read_csv(root / "aggregate/seed-stage-summary.csv")
    costs = pd.read_csv(root / "aggregate/adaptation-cost.csv")
    sham = pd.read_csv(root / "aggregate/sham-comparison.csv")
    primary = json.loads((root / "statistics/primary.json").read_text(encoding="utf-8"))
    required = {
        "random_init",
        "task_trained",
        "human_adapted",
        "sham_adapted",
    }
    missing = required - set(stages["stage"])
    expected_runs = 5 * 20 * 5 * 2
    failures = []
    if missing:
        failures.append(f"missing stages: {sorted(missing)}")
    if len(costs) != expected_runs:
        failures.append(f"adaptation runs {len(costs)} != {expected_runs}")
    if costs.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["relative_l2_parameter_displacement", "neural_alignment_loss_after"]
    ).shape[0] != len(costs):
        failures.append("non-finite primary adaptation costs")
    gate_failures = int((~costs["performance_gate_passed"].astype(bool)).sum())
    ranking = (
        stages[stages["stage"].isin(["task_trained", "human_adapted"])]
        .groupby(["stage", "architecture"], as_index=False)["rms_distance"]
        .mean()
        .sort_values(["stage", "rms_distance"])
    )
    article = root / "article"
    article.mkdir(parents=True, exist_ok=True)
    methods = """# Human-adaptation methods digest

Experiment 2 is separate from the frozen Experiment 1. Five architecture classes and the
same 20 seed lineages were evaluated in five outer Gabor participant folds. Within each
fold, held-out participants were reserved for final four-metric geometry evaluation and an
inner participant subset was reserved for checkpoint selection.

The adaptation target was the mean and variance of the six-by-six temporal spatial-cosine
representational-similarity matrix computed directly from preprocessed EEG activity from
150–300 ms. This lower-level, dimension-independent target contains no Jacobian geometry,
gain, broadcast, persistence, concentration, Access Index, RMS distance, cosine endpoint,
or magnitude endpoint. The matched sham jointly permuted the six time labels. The common
encoder, cue projection, and task heads were frozen; only architecture-specific state
formation and transition parameters were trainable. Checkpoints were selected solely by
inner-participant neural-alignment loss. Absolute task-accuracy change at difficulty bin 2
was required to remain at or below 0.02.
"""
    (article / "human-adaptation-methods-digest.md").write_text(methods, encoding="utf-8")
    lines = [
        "# Human-adaptation results digest",
        "",
        "This extension does not alter the completed Experiment 1 or its original ranking.",
        "",
        f"Completed adaptation runs: **{len(costs)} / {expected_runs}**.",
        f"Task-retention gate failures: **{gate_failures}**.",
        "",
        "## Architecture ranking by stage",
        "",
        "| Stage | Rank | Architecture | Mean held-out RMS distance |",
        "|---|---:|---|---:|",
    ]
    for stage, group in ranking.groupby("stage", sort=False):
        for index, row in enumerate(group.itertuples(), start=1):
            lines.append(f"| {stage} | {index} | {row.architecture} | {row.rms_distance:.4f} |")
    lines.extend(["", "## Primary paired tests", ""])
    for test in primary["tests"]:
        lines.append(
            f"- {test['architecture']}: mean alignment gain {test['mean']:.4f}, "
            f"95% CI {test['confidence_interval_95']}, Holm p={test.get('holm_adjusted_p', float('nan')):.4g}."
        )
    human_specific = sham.groupby("architecture")["human_specific_gain"].mean().sort_values(ascending=False)
    lines.extend(["", "## Human-specific gain over sham", ""])
    for architecture, value in human_specific.items():
        lines.append(f"- {architecture}: {value:.4f}")
    lines.extend(
        [
            "",
            "Interpretation is limited to held-out neural-dynamical similarity; no result is evidence that a machine is conscious.",
            "",
        ]
    )
    (article / "human-adaptation-results-digest.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    gate = {
        "passed": not failures,
        "failures": failures,
        "adaptation_runs": len(costs),
        "expected_adaptation_runs": expected_runs,
        "task_retention_gate_failures": gate_failures,
        "stages": sorted(set(stages["stage"])),
    }
    (root / "gates/human-adaptation-production.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
