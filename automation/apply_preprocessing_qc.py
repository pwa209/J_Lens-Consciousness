"""Freeze the ordinary-article analysis roster after outcome-blind preprocessing QC."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def is_included(value: object) -> bool:
    """Return whether a roster value marks a participant as eligible for QC."""

    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=Path, default=Path("configs/execution/participants.tsv"))
    parser.add_argument("--qc-root", type=Path, default=Path("data/derivatives/preprocessed"))
    parser.add_argument(
        "--inventory", type=Path, default=Path("configs/execution/participants-acquired.tsv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/gates/preprocessing-roster.json")
    )
    args = parser.parse_args()
    if not args.inventory.exists():
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.participants, args.inventory)
    with args.inventory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    decisions = []
    for row in rows:
        # Source-level exclusions (for example, an incomplete recording) cannot
        # have preprocessing QC. Preserve those decisions without demanding an
        # output that the production workflow intentionally never creates.
        if not is_included(row.get("include")):
            decisions.append(
                {
                    "dataset_id": row["dataset_id"],
                    "participant_id": row["participant_id"],
                    "included": False,
                    "exclusion_stage": "source_inventory",
                    "reason": row.get("reason", ""),
                    "qc": None,
                }
            )
            continue
        qc_path = args.qc_root / row["dataset_id"] / row["participant_id"] / "qc.json"
        if not qc_path.exists():
            raise SystemExit(f"missing preprocessing QC: {qc_path}")
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        included = bool(qc.get("included"))
        row["include"] = "1" if included else "0"
        if not included:
            row["reason"] = "outcome-blind preprocessing QC exclusion: " + "; ".join(
                str(value) for value in qc.get("deviations", [])
            )
        decisions.append(
            {
                "dataset_id": row["dataset_id"],
                "participant_id": row["participant_id"],
                "included": included,
                "exclusion_stage": None if included else "preprocessing_qc",
                "qc": qc,
            }
        )
    for dataset in ("gabor", "kronemer", "somato"):
        if not any(item["dataset_id"] == dataset and item["included"] for item in decisions):
            raise SystemExit(f"preprocessing QC left no eligible {dataset} participants")
    temporary = args.participants.with_suffix(args.participants.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.participants)
    payload = {
        "ready": True,
        "acquired": len(decisions),
        "included": sum(item["included"] for item in decisions),
        "excluded": sum(not item["included"] for item in decisions),
        "decisions": decisions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("ready", "acquired", "included", "excluded")}, indent=2))


if __name__ == "__main__":
    main()
