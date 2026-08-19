"""Run dependency-light checks on the development computer."""

from __future__ import annotations

import compileall
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from jacaccess.synthetic import run_synthetic  # noqa: E402
from jacaccess.synthetic_e2e import run_synthetic_e2e  # noqa: E402


def main() -> None:
    if not compileall.compile_dir(SRC, quiet=1):
        raise SystemExit("source compilation failed")

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)

    summary = run_synthetic()
    for field in ("gain_difference", "broadcast_difference", "access_index_difference"):
        if not isinstance(summary[field], float):
            raise SystemExit(f"synthetic field {field} is invalid")
    output = ROOT / "results" / "synthetic" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    e2e_output = ROOT / "results" / "synthetic-e2e" / "fold-00"
    e2e_summary = run_synthetic_e2e(e2e_output)
    if not e2e_summary["condition_joined_after_seal"]:
        raise SystemExit("synthetic condition table was joined before metric sealing")
    print(f"local checks passed; synthetic summaries written under {ROOT / 'results'}")


if __name__ == "__main__":
    main()
