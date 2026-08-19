"""Refuse production expansion until all source adapters are verified."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jacaccess.config import load_yaml  # noqa: E402

datasets = ("kronemer", "gabor", "somato")
failures: list[str] = []
for dataset in datasets:
    config = load_yaml(ROOT / "configs" / "datasets" / f"{dataset}.yaml")
    if config.get("adapter_status") != "verified":
        failures.append(f"{dataset} adapter status is {config.get('adapter_status')!r}")
    if not config.get("event_columns") and dataset != "somato":
        failures.append(f"{dataset} event_columns are not verified")
    if len(config.get("output_channel_groups", {})) < 2:
        failures.append(f"{dataset} needs at least two verified output_channel_groups")
    if not config.get("primary_contrasts"):
        failures.append(f"{dataset} primary_contrasts are not verified")
    if not config.get("prediction_outcome"):
        failures.append(f"{dataset} prediction_outcome is not verified")

participants_path = ROOT / "configs" / "execution" / "participants.tsv"
with participants_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
included_rows = [
    row for row in rows if str(row.get("include", "")).strip().lower() in {"1", "true", "yes"}
]
if not included_rows:
    failures.append("participants.tsv contains no verified participants")

report = {
    "ready": not failures,
    "failures": failures,
    "participant_rows": len(rows),
    "included_participant_rows": len(included_rows),
}
output = ROOT / "results" / "adapter-gate.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if failures:
    raise SystemExit(2)
