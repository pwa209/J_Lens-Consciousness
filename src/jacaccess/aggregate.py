"""Strict aggregation of completed human folds and machine seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def aggregate_human(
    root: Path,
    output: Path,
    participants_path: Path | None = None,
) -> None:
    included: set[tuple[str, str]] | None = None
    if participants_path is not None:
        with participants_path.open(encoding="utf-8", newline="") as handle:
            included = {
                (row["dataset_id"], row["participant_id"])
                for row in csv.DictReader(handle, delimiter="\t")
                if str(row.get("include", "")).strip().lower() in {"1", "true", "yes"}
            }
    frames: list[pd.DataFrame] = []
    seen: set[tuple[str, str, int]] = set()
    for index_path in sorted(root.rglob("partition-index.json")):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        key = (index["dataset_id"], index["participant_id"], int(index["fold"]))
        if included is not None and key[:2] not in included:
            continue
        if key in seen:
            raise ValueError(f"duplicate human fold {key}")
        seen.add(key)
        metrics = pd.concat(
            [pd.read_parquet(index_path.parent / part["path"]) for part in index["parts"]],
            ignore_index=True,
        )
        condition_path = index_path.parent.parent / "heldout-conditions.tsv"
        conditions = pd.read_csv(condition_path, sep="\t")
        frames.append(
            metrics.merge(
                conditions,
                on=["dataset_id", "participant_id", "original_trial_id"],
                validate="many_to_one",
            )
        )
    if not frames:
        raise FileNotFoundError(f"no completed human metric partitions under {root}")
    combined = pd.concat(frames, ignore_index=True)
    keys = ["dataset_id", "participant_id", "original_trial_id", "time_seconds"]
    if combined.duplicated(keys).any():
        raise ValueError("cross-fitted human output contains duplicate trial/time rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)


def aggregate_machine(root: Path, output: Path) -> None:
    signatures = sorted(root.rglob("jacobian-signatures.parquet"))
    interventions = sorted(root.rglob("intervention.json"))
    if not signatures or not interventions:
        raise FileNotFoundError("machine analysis outputs are incomplete")
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_parquet(path) for path in signatures], ignore_index=True).to_parquet(
        output / "jacobian-signatures.parquet", index=False
    )
    records = [json.loads(path.read_text(encoding="utf-8")) for path in interventions]
    table = pd.DataFrame(records)
    table.to_csv(output / "interventions.csv", index=False)
    summary = (
        table.groupby("architecture")
        .agg(
            successful_seeds=("passes_intervention_criterion", "sum"),
            completed_seeds=("seed", "count"),
            mean_top_drop=("top_subspace_accuracy_drop", "mean"),
            mean_random_drop=("random_drop_mean", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "architecture-summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    human = sub.add_parser("human")
    human.add_argument("--root", type=Path, required=True)
    human.add_argument("--output", type=Path, required=True)
    human.add_argument("--participants", type=Path)
    machine = sub.add_parser("machine")
    machine.add_argument("--root", type=Path, required=True)
    machine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "human":
        aggregate_human(args.root, args.output, args.participants)
    else:
        aggregate_machine(args.root, args.output)


if __name__ == "__main__":
    main()
