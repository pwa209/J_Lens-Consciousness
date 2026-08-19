"""Audit human production completeness and prespecified model-QC thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from jacaccess.config import load_yaml


def dataset_failure_audit(
    failed_folds: Counter[str],
    expected_folds: Counter[str],
    maximum_fraction: float,
) -> tuple[dict[str, float], list[str]]:
    """Return per-dataset rates and failures above the configured ceiling."""
    fractions = {
        dataset: failed_folds[dataset] / max(expected, 1)
        for dataset, expected in sorted(expected_folds.items())
    }
    failures = [
        (
            f"{dataset}: fold failure fraction {fraction:.3f} exceeds "
            f"{maximum_fraction}"
        )
        for dataset, fraction in fractions.items()
        if fraction > maximum_fraction
    ]
    return fractions, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-quality-failure", action="store_true")
    args = parser.parse_args()
    with Path("configs/execution/participants.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        participants = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get("include", "")).strip().lower() in {"1", "true", "yes"}
        ]
    thresholds = load_yaml(Path("configs/models/human.yaml"))["quality_control"]
    failures: list[str] = []
    warnings: list[str] = []
    fold_failures = 0
    failed_folds_by_dataset: Counter[str] = Counter()
    expected_folds_by_dataset: Counter[str] = Counter(
        row["dataset_id"] for row in participants for _ in range(5)
    )
    summaries = 0
    for row in participants:
        dataset = row["dataset_id"]
        participant = row["participant_id"]
        qc_path = Path(f"data/derivatives/preprocessed/{dataset}/{participant}/qc.json")
        if not qc_path.exists():
            failures.append(f"missing {qc_path}")
        else:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            if not qc.get("included"):
                failures.append(f"{dataset}/{participant}: preprocessing QC excluded")
        for fold in range(5):
            path = Path(f"results/human/{dataset}/{participant}/fold-{fold}/summary.json")
            if not path.exists():
                failures.append(f"missing {path}")
                fold_failures += 1
                failed_folds_by_dataset[dataset] += 1
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            summaries += 1
            qc = summary["qc"]
            fold_failed = False
            if float(qc["heldout_r2"]) < float(thresholds["heldout_r2_min"]):
                warnings.append(f"{dataset}/{participant}/fold-{fold}: heldout R2 failed")
                fold_failed = True
            if float(qc["improvement_over_persistence"]) < float(
                thresholds["improvement_over_persistence_min"]
            ):
                warnings.append(
                    f"{dataset}/{participant}/fold-{fold}: persistence improvement failed"
                )
                fold_failed = True
            if fold_failed:
                fold_failures += 1
                failed_folds_by_dataset[dataset] += 1
    expected = len(participants) * 5
    failure_fraction = fold_failures / max(expected, 1)
    per_dataset_failure_fraction, threshold_failures = dataset_failure_audit(
        failed_folds_by_dataset,
        expected_folds_by_dataset,
        float(thresholds["dataset_failure_fraction_max"]),
    )
    failures.extend(threshold_failures)
    for path in (
        Path("results/aggregate/human.parquet"),
        Path("results/prediction/nested-cv/summary.json"),
    ):
        if not path.exists():
            failures.append(f"missing {path}")
    report = {
        "ready": not failures,
        "participants": len(participants),
        "expected_folds": expected,
        "completed_summaries": summaries,
        "model_qc_failure_fraction": failure_fraction,
        "model_qc_failure_fraction_by_dataset": per_dataset_failure_fraction,
        "failures": failures,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures and not args.allow_quality_failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
