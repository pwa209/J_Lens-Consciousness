"""Generate Phase 1 source evidence without changing scientific mappings."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from jacaccess.io.source_inspection import inspect_source_tree

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "source-inspection"


def _inspect(dataset: str, participant: str, source: Path) -> dict[str, object]:
    report = inspect_source_tree(source)
    report.update({"dataset_id": dataset, "participant_id": participant})
    destination = OUTPUT / dataset / f"{participant}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = {
        "gabor": _inspect("gabor", "sub-10", ROOT / "data" / "raw" / "gabor" / "sub-10"),
        "somato": _inspect("somato", "pilot-tree", ROOT / "data" / "raw" / "somato"),
        "kronemer_report": _inspect(
            "kronemer",
            "223_RP_EEG",
            ROOT / "data" / "raw" / "kronemer" / "223_RP_EEG",
        ),
        "kronemer_no_report": _inspect(
            "kronemer",
            "238_NRP",
            ROOT / "data" / "raw" / "kronemer" / "238_NRP",
        ),
    }
    failures: list[str] = []
    for name, report in reports.items():
        if not report["file_count"]:
            failures.append(f"{name}: no extracted files")
        if not report["signal_candidates"]:
            failures.append(f"{name}: no recognized signal candidates")
    evidence = {
        "status": "INSPECTION_COMPLETE" if not failures else "INSPECTION_INCOMPLETE",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reports": {
            name: {
                "file_count": report["file_count"],
                "total_size_bytes": report["total_size_bytes"],
                "signal_candidate_count": len(report["signal_candidates"]),
                "tabular_file_count": len(report["tabular_files"]),
                "matlab_file_count": len(report["matlab_files"]),
            }
            for name, report in reports.items()
        },
        "failures": failures,
        "next_gate": (
            "Verify mappings/configs and participants; the queue never infers scientific labels."
        ),
    }
    (OUTPUT / "phase1-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
