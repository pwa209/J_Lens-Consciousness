"""Audit machine benchmark or production completion without hiding failed seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jacaccess.config import load_yaml

ARCHITECTURES = (
    "feedforward", "recurrent", "shared_workspace", "private_modules",
    "unlimited_shared_state",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/machine"))
    parser.add_argument("--mode", choices=("benchmark", "production"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-scientific-failure", action="store_true")
    args = parser.parse_args()
    config = load_yaml(Path("configs/models/machine.yaml"))
    seeds = (0,) if args.mode == "benchmark" else tuple(range(int(config["seeds"])))
    failures: list[str] = []
    summaries: list[dict[str, object]] = []
    interventions: list[dict[str, object]] = []
    scientific_failures: list[str] = []
    for architecture in ARCHITECTURES:
        for seed in seeds:
            directory = args.root / architecture / f"seed-{seed}"
            summary_path = directory / "summary.json"
            if not summary_path.exists():
                failures.append(f"missing {summary_path}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries.append(summary)
            relative_error = abs(
                int(summary["parameters"]) - int(config["parameter_target"])
            ) / int(config["parameter_target"])
            if relative_error > float(config["parameter_tolerance_fraction"]):
                failures.append(f"{architecture}/seed-{seed}: parameter tolerance failed")
            if args.mode == "production":
                path = directory / "intervention.json"
                if not path.exists():
                    failures.append(f"missing {path}")
                else:
                    interventions.append(json.loads(path.read_text(encoding="utf-8")))
    success_counts = {
        architecture: sum(
            bool(item.get("passes_intervention_criterion"))
            for item in interventions
            if item.get("architecture") == architecture
        )
        for architecture in ARCHITECTURES
    }
    if args.mode == "production":
        for architecture, count in success_counts.items():
            if count < int(config["successful_seeds_required"]):
                scientific_failures.append(
                    f"{architecture}: {count} successful intervention seeds; "
                    f"{config['successful_seeds_required']} required"
                )
        accuracy_path = Path(
            "results/aggregate/machine/accuracy-matching/accuracy-matching.json"
        )
        if not accuracy_path.exists():
            failures.append(f"missing {accuracy_path}")
            accuracy_matching = None
        else:
            accuracy_matching = json.loads(accuracy_path.read_text(encoding="utf-8"))
            if not accuracy_matching.get("ready"):
                failures.append("machine accuracy-matching audit is incomplete")
    else:
        accuracy_matching = None
    report = {
        "mode": args.mode,
        "ready": not failures and (args.allow_scientific_failure or not scientific_failures),
        "execution_complete": not failures,
        "scientific_criteria_passed": not scientific_failures,
        "expected_runs": len(ARCHITECTURES) * len(seeds),
        "completed_summaries": len(summaries),
        "completed_interventions": len(interventions),
        "successful_intervention_seeds": success_counts,
        "accuracy_matching": accuracy_matching,
        "failures": failures,
        "scientific_criterion_failures": scientific_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures or (scientific_failures and not args.allow_scientific_failure):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
