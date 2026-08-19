from __future__ import annotations

import unittest

from jacaccess.preprocess.eeg import (
    _notch_before_continuous_resampling,
    _resample_before_continuous_filters,
)


class FakeRaw:
    def __init__(self, sampling_rate_hz: float) -> None:
        self.info = {"sfreq": sampling_rate_hz}
        self.calls: list[tuple[float, int, str]] = []

    def resample(self, sampling_rate_hz: float, *, n_jobs: int, verbose: str) -> None:
        self.calls.append((sampling_rate_hz, n_jobs, verbose))
        self.info["sfreq"] = sampling_rate_hz


class FakeNotchRaw:
    def __init__(self, sampling_rate_hz: float) -> None:
        self.info = {"sfreq": sampling_rate_hz}
        self.calls: list[dict[str, object]] = []

    def apply_function(self, function: object, **kwargs: object) -> None:
        self.calls.append({"function": function, **kwargs})


class ContinuousMemoryTests(unittest.TestCase):
    def test_downsamples_before_expensive_continuous_filters(self) -> None:
        raw = FakeRaw(1000.0)

        changed = _resample_before_continuous_filters(raw, 100.0, n_jobs=1)

        self.assertTrue(changed)
        self.assertEqual(raw.calls, [(100.0, 1, "ERROR")])
        self.assertEqual(raw.info["sfreq"], 100.0)

    def test_does_not_upsample_or_repeat_resampling(self) -> None:
        raw = FakeRaw(100.0)

        changed = _resample_before_continuous_filters(raw, 100.0, n_jobs=4)

        self.assertFalse(changed)
        self.assertEqual(raw.calls, [])

    def test_rejects_nonpositive_target_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _resample_before_continuous_filters(FakeRaw(1000.0), 0.0, n_jobs=1)

    def test_line_notch_is_channel_wise_before_decimation(self) -> None:
        raw = FakeNotchRaw(1000.0)

        applied = _notch_before_continuous_resampling(raw, 60.0, quality_factor=30.0)

        self.assertTrue(applied)
        self.assertEqual(len(raw.calls), 1)
        self.assertEqual(raw.calls[0]["picks"], "eeg")
        self.assertTrue(raw.calls[0]["channel_wise"])
        self.assertEqual(raw.calls[0]["n_jobs"], 1)

    def test_line_notch_is_skipped_above_target_nyquist(self) -> None:
        raw = FakeNotchRaw(100.0)

        applied = _notch_before_continuous_resampling(raw, 60.0, quality_factor=30.0)

        self.assertFalse(applied)
        self.assertEqual(raw.calls, [])


if __name__ == "__main__":
    unittest.main()
