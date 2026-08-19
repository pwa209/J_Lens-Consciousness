"""Validate repository YAML contracts after the full environment is installed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jacaccess.config import validate_repository_configs  # noqa: E402

report = validate_repository_configs(ROOT)
output = ROOT / "results" / "configuration-validation.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(2)

