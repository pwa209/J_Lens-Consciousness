"""Record pre-analysis implementation amendments and the production manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results-extension/human-adaptation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = [
        ROOT / "configs/models/machine.yaml",
        ROOT / "src/jacaccess/machine/analyze.py",
        ROOT / "src/jacaccess/machine/stage_comparison.py",
        ROOT / "src/jacaccess/stats/adaptation_comparison.py",
    ]
    amendment = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "timing": "after adaptation launch and before any Experiment 2 held-out geometry",
        "heldout_geometry_inspected": False,
        "changes": [
            "populated the required stage task_accuracy output field",
            "batched the same 100 random-subspace intervention evaluations in chunks of 10",
            "added the planned architecture-comparison and task-retention output artifacts",
        ],
        "unchanged": [
            "human target",
            "participant splits",
            "adaptation objective and hyperparameters",
            "checkpoint selection",
            "geometry definitions",
            "hypotheses and statistical endpoints",
        ],
        "file_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths
        },
    }
    provenance = EXPERIMENT / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / "source-freeze-amendment.json").write_text(
        json.dumps(amendment, indent=2) + "\n", encoding="utf-8"
    )
    config = yaml.safe_load(
        (ROOT / "configs/analysis/human_adaptation.yaml").read_text(encoding="utf-8")
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "human_adaptation",
        "architectures": config["architectures"],
        "seeds": list(range(int(config["seeds"]))),
        "outer_subject_folds": list(range(int(config["outer_subject_folds"]))),
        "adaptation_conditions": ["human_adapted", "sham_adapted"],
        "planned_adaptation_runs": len(config["architectures"])
        * int(config["seeds"])
        * int(config["outer_subject_folds"])
        * 2,
        "random_initialization": "exact_training_seed_reconstruction",
        "primary_endpoint": "held-out RMS alignment gain versus task-trained with sham comparison",
        "machine_config_sha256": sha256(ROOT / "configs/models/machine.yaml"),
        "extension_config_sha256": sha256(
            EXPERIMENT / "config/human-adaptation-frozen.yaml"
        ),
    }
    (provenance / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
