"""Audit final study outputs and write a compact completion inventory."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from jacaccess.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = (
    "feedforward", "recurrent", "shared_workspace", "private_modules",
    "unlimited_shared_state",
)


def included_participants() -> list[dict[str, str]]:
    with (ROOT / "configs/execution/participants.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get("include", "")).strip().lower() in {"1", "true", "yes"}
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/provenance/final-results-inventory.json"),
    )
    args = parser.parse_args()
    participants = included_participants()
    expected_human = len(participants) * 5
    expected_machine = len(ARCHITECTURES) * 20
    human_summaries = [
        ROOT / "results" / "human" / row["dataset_id"] / row["participant_id"]
        / f"fold-{fold}" / "summary.json"
        for row in participants
        for fold in range(5)
        if (
            ROOT / "results" / "human" / row["dataset_id"] / row["participant_id"]
            / f"fold-{fold}" / "summary.json"
        ).exists()
    ]
    machine_summaries = list((ROOT / "results/machine").glob("*/seed-*/summary.json"))
    machine_signatures = list(
        (ROOT / "results/machine").glob("*/seed-*/jacobian-signatures.parquet")
    )
    machine_interventions = list(
        (ROOT / "results/machine").glob("*/seed-*/intervention.json")
    )
    required = [
        ROOT / "results/study_complete.flag",
        ROOT / "results/gates/human-production.json",
        ROOT / "results/gates/machine-production.json",
        ROOT / "results/aggregate/human.parquet",
        ROOT / "results/aggregate/machine/architecture-summary.csv",
        ROOT / "results/aggregate/machine/accuracy-matching/accuracy-matching.json",
        ROOT / "results/prediction/nested-cv/summary.json",
        ROOT / "results/figures/main-human-timecourse.png",
        ROOT / "results/figures/main-machine-comparison.png",
        ROOT / "results/figures/supplement-metric-distributions.png",
        ROOT / "results/figures/science-advances/figure-manifest.json",
    ]
    contrast_outputs: list[Path] = []
    for dataset in ("gabor", "somato", "kronemer"):
        config = load_yaml(ROOT / f"configs/datasets/{dataset}.yaml")
        for index, _contrast in enumerate(config.get("primary_contrasts", [])):
            contrast_outputs.append(
                ROOT / f"results/statistics/{dataset}-{index}/bayes-factor.json"
            )
    failures = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    failures.extend(
        str(path.relative_to(ROOT)) for path in contrast_outputs if not path.exists()
    )
    figure_manifest = ROOT / "results/figures/science-advances/figure-manifest.json"
    if figure_manifest.exists():
        figure_payload = json.loads(figure_manifest.read_text(encoding="utf-8"))
        if not figure_payload.get("ready"):
            failures.append("Science Advances figure manifest is not ready")
        if figure_payload.get("main_figures") != 6:
            failures.append("Science Advances figure manifest does not contain six main figures")
        if figure_payload.get("supplementary_figures") != 3:
            failures.append("Science Advances figure manifest does not contain three supplements")
    counts = {
        "included_participants": len(participants),
        "human_fold_summaries": len(human_summaries),
        "expected_human_fold_summaries": expected_human,
        "machine_training_summaries": len(machine_summaries),
        "machine_signature_tables": len(machine_signatures),
        "machine_intervention_reports": len(machine_interventions),
        "expected_machine_runs": expected_machine,
    }
    if len(human_summaries) != expected_human:
        failures.append("human fold summary count does not match included participants")
    for name, values in (
        ("machine training", machine_summaries),
        ("machine signature", machine_signatures),
        ("machine intervention", machine_interventions),
    ):
        if len(values) != expected_machine:
            failures.append(f"{name} count is {len(values)}, expected {expected_machine}")
    report = {
        "ready": not failures,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "counts": counts,
        "failures": failures,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
