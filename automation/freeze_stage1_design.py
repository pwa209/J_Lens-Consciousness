"""Hash the prospective design before any fifth-architecture result completes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "docs/registered-report/STAGE1_PROTOCOL.md",
    "docs/registered-report/EXISTING_DATA_BIAS_DECLARATION.md",
    "configs/analysis/registered_stage1.yaml",
    "configs/execution/pilot_firewall.tsv",
    "configs/models/machine.yaml",
    "src/jacaccess/machine/architectures.py",
    "src/jacaccess/machine/analyze.py",
    "automation/run_unlimited_shared_state.sh",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = ROOT / "results/registration/stage1-design-lock.json"
    fifth = list((ROOT / "results/machine/unlimited_shared_state").glob("seed-*/summary.json"))
    payload = {
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "completed_fifth_architecture_summaries_at_lock": len(fifth),
        "hash_algorithm": "sha256",
        "files": {name: digest(ROOT / name) for name in FILES},
    }
    if fifth:
        raise SystemExit("refusing prospective lock: a fifth-architecture summary already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

