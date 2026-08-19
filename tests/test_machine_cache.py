from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from jacaccess.machine.cache import StimulusCache, prepare_stimulus_cache
from jacaccess.machine.stimuli import generate_stimulus_batch


class MachineCacheTests(unittest.TestCase):
    def test_cache_preserves_exact_indexed_stimuli(self) -> None:
        config = {
            "image_size": [16, 16],
            "train_images": 7,
            "validation_images": 3,
            "test_images": 2,
            "target_present_probability": 2 / 3,
            "contrast_noise_levels": 4,
            "batch_size": 5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            cache_path = root / "cache"
            prepare_stimulus_cache(config_path, 37, cache_path, workers=2)
            cache = StimulusCache(cache_path)
            indices = np.asarray([11, 0, 5, 3], dtype=np.int64)
            actual = cache.batch(indices)
            expected = generate_stimulus_batch(
                indices,
                seed=37,
                image_size=16,
                target_probability=2 / 3,
                levels=4,
            )
            np.testing.assert_array_equal(actual.images, expected.images)
            np.testing.assert_array_equal(actual.task_cues, expected.task_cues)
            np.testing.assert_array_equal(actual.difficulty_bin, expected.difficulty_bin)
            for name in expected.labels:
                np.testing.assert_array_equal(actual.labels[name], expected.labels[name])
                np.testing.assert_array_equal(
                    actual.valid_masks[name], expected.valid_masks[name]
                )


if __name__ == "__main__":
    unittest.main()
