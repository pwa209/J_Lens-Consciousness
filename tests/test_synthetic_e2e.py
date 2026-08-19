from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jacaccess.synthetic_e2e import run_synthetic_e2e


class SyntheticEndToEndTests(unittest.TestCase):
    def test_sealed_metrics_precede_condition_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = run_synthetic_e2e(root)
            self.assertTrue(summary["condition_joined_after_seal"])
            self.assertEqual(len(summary["sealed_metrics_sha256"]), 64)
            self.assertTrue((root / "sealed_metrics.npz").is_file())
            self.assertTrue((root / "condition_table.json").is_file())
            self.assertTrue((root / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()

