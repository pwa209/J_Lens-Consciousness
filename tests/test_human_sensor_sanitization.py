from __future__ import annotations

import unittest

import numpy as np

from jacaccess.human_pipeline import _sanitize_sensor_channels


class SensorSanitizationTests(unittest.TestCase):
    def test_fully_missing_channel_becomes_zero_and_is_recorded(self) -> None:
        sensor = np.ones((3, 5, 4), dtype=np.float32)
        sensor[..., 3] = np.nan

        sanitized, channels = _sanitize_sensor_channels(sensor)

        self.assertEqual(channels, [3])
        self.assertTrue(np.isfinite(sanitized).all())
        np.testing.assert_array_equal(sanitized[..., 3], 0.0)
        np.testing.assert_array_equal(sanitized[..., :3], 1.0)

    def test_partially_missing_channel_is_rejected(self) -> None:
        sensor = np.ones((3, 5, 4), dtype=np.float32)
        sensor[0, 0, 2] = np.nan

        with self.assertRaisesRegex(ValueError, "partially non-finite channels: 2"):
            _sanitize_sensor_channels(sensor)


if __name__ == "__main__":
    unittest.main()
