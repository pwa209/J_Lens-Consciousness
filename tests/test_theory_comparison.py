from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from jacaccess.stats.theory_comparison import (
    HUMAN_COLUMNS,
    METRICS,
    compare_theories,
    human_contrasts,
    machine_contrasts,
)


class TheoryComparisonTests(unittest.TestCase):
    def test_human_contrasts_match_numeric_levels_from_yaml(self) -> None:
        rows = []
        for level, value in ((0.0, 1.0), (1.0, 2.0)):
            rows.append(
                {
                    "dataset_id": "gabor",
                    "participant_id": "p1",
                    "original_trial_id": f"t{int(level)}",
                    "seen": level,
                    "time_seconds": 0.2,
                    **{column: value for column in HUMAN_COLUMNS.values()},
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = (
                "primary_contrasts:\n"
                "  - condition_field: seen\n    positive: 1\n    negative: 0\n"
                "    window_ms: [150, 300]\n"
            )
            for dataset in ("gabor", "kronemer", "somato"):
                (root / f"{dataset}.yaml").write_text(definition, encoding="utf-8")
            result = human_contrasts(pd.DataFrame(rows), root)
        self.assertEqual(len(result), len(METRICS))
        self.assertTrue((result["contrast"] == 1.0).all())

    def test_skips_seed_without_both_accuracy_classes(self) -> None:
        rows = []
        for seed, correctness in ((0, (False, True)), (1, (True,))):
            for correct in correctness:
                for repetition in range(2):
                    rows.append(
                        {
                            "architecture": "feedforward",
                            "seed": seed,
                            "presence_correct": correct,
                            "difficulty_bin": 2,
                            **{
                                metric: 0.2 + seed + repetition + int(correct)
                                for metric in METRICS
                            },
                        }
                    )
        result = machine_contrasts(
            pd.DataFrame(rows), {"selected_threshold_bin": 2}, Path("unused.parquet")
        )
        self.assertEqual(set(result["seed"]), {0})
        self.assertEqual(set(result["metric"]), set(METRICS))

    def test_ranks_architectures_and_reports_capacity_equivalence(self) -> None:
        human_rows = []
        for participant, offset in enumerate((-0.15, -0.05, 0.05, 0.15)):
            for metric_index, metric in enumerate(METRICS, start=1):
                human_rows.append(
                    {
                        "dataset_id": "gabor",
                        "contrast_id": "gabor-0",
                        "participant_id": f"p{participant}",
                        "metric": metric,
                        "contrast": metric_index + offset,
                    }
                )
        machine_rows = []
        values = {
            "feedforward": 0.0,
            "recurrent": 0.5,
            "private_modules": -0.5,
            "shared_workspace": 1.0,
            "unlimited_shared_state": 1.0,
        }
        for architecture, multiplier in values.items():
            for seed in range(4):
                for metric_index, metric in enumerate(METRICS, start=1):
                    machine_rows.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "metric": metric,
                            "contrast": metric_index * multiplier,
                        }
                    )
        distances, report = compare_theories(
            pd.DataFrame(human_rows), pd.DataFrame(machine_rows)
        )
        self.assertEqual(len(distances), 20)
        self.assertEqual(report["winning_architecture"], "shared_workspace")
        self.assertTrue(report["all_primary_metrics_equivalent"])
        self.assertTrue(report["capacity_limit_interpretation_falsified"])


if __name__ == "__main__":
    unittest.main()
