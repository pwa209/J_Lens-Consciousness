from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jacaccess.environment_check import (
    _cgroup_cpu_cores,
    _cgroup_memory_limit_bytes,
    validate_production_environment,
)


class CgroupEnvironmentTests(unittest.TestCase):
    def test_v2_memory_and_cpu_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.max").write_text(str(90 * 2**30), encoding="utf-8")
            (root / "cpu.max").write_text("2500000 100000\n", encoding="utf-8")
            self.assertEqual(_cgroup_memory_limit_bytes(root), 90 * 2**30)
            self.assertEqual(_cgroup_cpu_cores(root), 25.0)

    def test_unlimited_v2_values_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.max").write_text("max\n", encoding="utf-8")
            (root / "cpu.max").write_text("max 100000\n", encoding="utf-8")
            self.assertIsNone(_cgroup_memory_limit_bytes(root))
            self.assertIsNone(_cgroup_cpu_cores(root))

    def test_effective_resource_thresholds_are_enforced(self) -> None:
        report = {
            "torch_installed": True,
            "cuda_available": True,
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_vram_gib": 31.8,
            "compute_capability": [12, 0],
            "system_ram_gib": 90.0,
            "effective_cpu_cores": 25.0,
            "scratch_free_gib": 2900.0,
        }
        self.assertEqual(
            validate_production_environment(
                report,
                minimum_ram_gib=80,
                minimum_scratch_gib=2500,
                minimum_cpu_cores=25,
            ),
            [],
        )
        failures = validate_production_environment(report)
        self.assertTrue(any("effective system RAM" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
