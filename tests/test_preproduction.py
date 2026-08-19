from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jacaccess.prediction.features import conventional_eeg_features, jacobian_features
from jacaccess.preprocess.artifacts import epoch_artifact_mask


class ArtifactTests(unittest.TestCase):
    def test_large_peak_to_peak_epoch_is_rejected(self) -> None:
        values = np.zeros((4, 2, 20), dtype=float)
        values[2, 0, 4] = 200e-6
        reject, scores = epoch_artifact_mask(values, peak_to_peak_uv_max=150)
        self.assertEqual(reject.tolist(), [False, False, True, False])
        self.assertGreater(scores["peak_to_peak_uv"][2], 150)


class PredictionFeatureTests(unittest.TestCase):
    def test_conventional_features_are_trial_level(self) -> None:
        epochs = np.arange(3 * 2 * 10, dtype=float).reshape(3, 2, 10)
        times = np.linspace(-0.1, 0.35, 10)
        result = conventional_eeg_features(epochs, times, {"early": (0, 200)})
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(
            set(result),
            {"erp_mean__early", "gfp__early", "power__early"},
        )

    def test_jacobian_features_keep_trial_ids(self) -> None:
        table = pd.DataFrame(
            {
                "original_trial_id": np.repeat(["a", "b"], 4),
                "time_seconds": np.tile([0.0, 0.1, 0.2, 0.3], 2),
                "gain": np.arange(8, dtype=float),
                "access_index": np.arange(8, dtype=float) / 2,
            }
        )
        result = jacobian_features(table, {"early": (0, 200)})
        self.assertEqual(result["original_trial_id"].tolist(), ["a", "b"])
        self.assertIn("jacobian_gain__early", result)
        self.assertIn("jacobian_access_index__early", result)


if __name__ == "__main__":
    unittest.main()
