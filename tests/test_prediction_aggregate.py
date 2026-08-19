from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from jacaccess.prediction.aggregate import harmonize_outcomes


class PredictionAggregationTests(unittest.TestCase):
    def test_harmonizes_declared_binary_contrasts_and_drops_third_level(self) -> None:
        table = pd.DataFrame(
            [
                {"dataset_id": "gabor", "outcome": 0.0},
                {"dataset_id": "gabor", "outcome": 1.0},
                {"dataset_id": "somato", "outcome": "no_report"},
                {"dataset_id": "somato", "outcome": "tactile_irrelevant"},
                {"dataset_id": "somato", "outcome": "tactile_relevant"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gabor.yaml").write_text(
                "prediction_outcome: seen\nprimary_contrasts:\n"
                "  - condition_field: seen\n    positive: 1\n    negative: 0\n",
                encoding="utf-8",
            )
            (root / "somato.yaml").write_text(
                "prediction_outcome: task_relevance\nprimary_contrasts:\n"
                "  - condition_field: task_relevance\n"
                "    positive: tactile_relevant\n    negative: tactile_irrelevant\n",
                encoding="utf-8",
            )
            result = harmonize_outcomes(table, root)
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["outcome"]), {0, 1})
        self.assertNotIn("no_report", set(result["outcome_source_level"]))


if __name__ == "__main__":
    unittest.main()
