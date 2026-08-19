from __future__ import annotations

import unittest

import pandas as pd

from jacaccess.stats.report import analyze_contrast


class StatisticsReportTests(unittest.TestCase):
    def test_cli_string_levels_match_numeric_table_levels(self) -> None:
        rows = []
        for participant in ("p1", "p2"):
            for trial, level, value in (("a", 0.0, 1.0), ("b", 1.0, 2.0)):
                rows.append(
                    {
                        "dataset_id": "gabor",
                        "participant_id": participant,
                        "original_trial_id": f"{participant}-{trial}",
                        "seen": level,
                        "time_seconds": 0.2,
                        "access_index": value,
                    }
                )
        contrasts, result = analyze_contrast(
            pd.DataFrame(rows),
            dataset="gabor",
            condition_field="seen",
            positive="1",
            negative="0",
            window_ms=(150, 300),
            permutations=20,
            seed=7,
        )
        self.assertEqual(len(contrasts), 2)
        self.assertEqual(result["mean_contrast"], 1.0)


if __name__ == "__main__":
    unittest.main()
