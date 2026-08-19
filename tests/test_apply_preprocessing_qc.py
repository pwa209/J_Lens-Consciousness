from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ApplyPreprocessingQCTests(unittest.TestCase):
    def test_preserves_source_exclusion_without_requiring_qc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            participants = base / "participants.tsv"
            inventory = base / "participants-acquired.tsv"
            qc_root = base / "preprocessed"
            report = base / "preprocessing-roster.json"
            rows = [
                {"dataset_id": "gabor", "participant_id": "good", "include": "1", "reason": ""},
                {"dataset_id": "gabor", "participant_id": "missing", "include": "0", "reason": "incomplete source"},
                {"dataset_id": "somato", "participant_id": "good", "include": "1", "reason": ""},
                {"dataset_id": "kronemer", "participant_id": "good", "include": "1", "reason": ""},
            ]
            for path in (participants, inventory):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
            for dataset in ("gabor", "somato", "kronemer"):
                output = qc_root / dataset / "good"
                output.mkdir(parents=True)
                (output / "qc.json").write_text(
                    json.dumps({"included": True, "deviations": []}), encoding="utf-8"
                )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "automation/apply_preprocessing_qc.py"),
                    "--participants", str(participants),
                    "--inventory", str(inventory),
                    "--qc-root", str(qc_root),
                    "--report", str(report),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(report.read_text(encoding="utf-8"))
            excluded = next(
                item for item in payload["decisions"] if item["participant_id"] == "missing"
            )
            self.assertEqual(payload["acquired"], 4)
            self.assertEqual(payload["included"], 3)
            self.assertEqual(payload["excluded"], 1)
            self.assertEqual(excluded["exclusion_stage"], "source_inventory")
            self.assertIsNone(excluded["qc"])


if __name__ == "__main__":
    unittest.main()
