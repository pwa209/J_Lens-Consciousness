from __future__ import annotations

import multiprocessing as mp
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from automation.build_full_participant_roster import _gabor, _kronemer
from jacaccess.config import load_yaml
from jacaccess.human_pipeline import _channel_groups
from jacaccess.io.adapters import VerifiedRepositoryAdapter, get_adapter
from jacaccess.io.standardize import (
    _align_behavior_clock,
    _gabor_interrupted_window,
    _kronemer_behavior_files,
    _somato_location_file,
    _trigger_code,
)
from jacaccess.machine.stimuli import generate_stimulus_batch
from jacaccess.machine.train import (
    _initialize_stimulus_worker,
    _prefetched_batches,
)


class AdapterMappingTests(unittest.TestCase):
    def _touch(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_gabor_output_groups_use_post_eog_channel_order(self) -> None:
        config = load_yaml(Path("configs/datasets/gabor.yaml"))
        groups = _channel_groups(config, channel_count=61)
        self.assertEqual(len(groups), 6)
        maximum = max(int(index) for values in groups.values() for index in values)
        self.assertLess(maximum, 61)

    def test_gabor_trigger_parser_uses_brainvision_code(self) -> None:
        self.assertEqual(_trigger_code("Stimulus/S 10"), 10)
        self.assertEqual(_trigger_code("Stimulus/S  74"), 74)
        self.assertIsNone(_trigger_code("New Segment/"))

    def test_gabor_discontinuity_window_is_excluded_not_relabelled(self) -> None:
        self.assertTrue(
            _gabor_interrupted_window(
                ["Stimulus/S 10", "Stimulus/S  4", "New Segment/"]
            )
        )
        self.assertFalse(_gabor_interrupted_window(["Stimulus/S 10", "Stimulus/S 56"]))

    def test_behavior_clock_alignment_ignores_extra_sync_triggers(self) -> None:
        behavior = np.asarray([90.8655, 102.7939, 115.0844])
        annotations = np.asarray([18.245, 108.0940, 120.0224, 132.3129, 900.0])
        aligned = _align_behavior_clock(behavior, annotations)
        np.testing.assert_allclose(aligned, annotations[1:4], atol=0.001)

    def test_behavior_clock_alignment_handles_small_drift_and_interspersed_triggers(self) -> None:
        behavior = np.asarray([100.0, 200.0, 300.0, 400.0])
        expected = 1.00005 * behavior + 17.25
        annotations = np.sort(np.concatenate((expected, [20.0, 250.0, 900.0])))
        aligned = _align_behavior_clock(behavior, annotations)
        np.testing.assert_allclose(aligned, expected, atol=0.001)

    def test_kronemer_session_separator_is_normalized(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self._touch(root / "Task" / "EEG_Session" / "EEG_Data" / "1_Session_2.raw")
            expected = self._touch(root / "Task" / "EEG_Session" / "Behavioral_Data" / "1_Session_2_test.csv")
            self._touch(root / "Task" / "EEG_Session" / "Behavioral_Data" / "1_Session_1_test.csv")
            files, method = _kronemer_behavior_files(raw, root)
            self.assertEqual(files, [expected])
            self.assertEqual(method, "session_label")

    def test_kronemer_mislabeled_sessions_use_acquisition_order(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eeg = root / "Task" / "EEG_Session" / "EEG_Data"
            behavior = root / "Task" / "EEG_Session" / "Behavioral_Data"
            self._touch(eeg / "1_Session_1.raw")
            raw2 = self._touch(eeg / "1_Session_2.raw")
            self._touch(behavior / "1_Session 1_1000.csv")
            expected = self._touch(behavior / "1_Session 1_1100.csv")
            files, method = _kronemer_behavior_files(raw2, root)
            self.assertEqual(files, [expected])
            self.assertEqual(method, "within_condition_order")

    def test_kronemer_split_run_directory_maps_both_files(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "Task" / "EEG_Session"
            raw = self._touch(session / "EEG_Data" / "Run 1 and 2" / "1_VisualPerTask2.raw")
            first = self._touch(session / "Behavioral_Data" / "Run 1" / "1_Run_1.csv")
            second = self._touch(session / "Behavioral_Data" / "Run 2" / "1_Run_2.csv")
            files, method = _kronemer_behavior_files(raw, root)
            self.assertEqual(files, [first, second])
            self.assertEqual(method, "run_directory")

    def test_kronemer_flat_report_task_one_is_calibration(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "Task" / "EEG_Session"
            raw = self._touch(session / "EEG_Data" / "1_VisualPerTask1.raw")
            self._touch(session / "Behavioral_Data" / "Calibration" / "calibration.csv")
            self._touch(session / "Behavioral_Data" / "Run 1 and 2" / "run12.csv")
            files, method = _kronemer_behavior_files(raw, root)
            self.assertEqual(files, [])
            self.assertEqual(method, "calibration_recording")

    def test_somato_selected_location_filename_is_supported(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self._touch(root / "R" / "EEG_selezionato_locations.mat")
            self.assertEqual(_somato_location_file(root), expected)

    def test_roster_excludes_incomplete_brainvision_set(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eeg = root / "sub-01" / "eeg"
            self._touch(eeg / "sub-01_events.tsv")
            header = self._touch(eeg / "sub-01_eeg.vhdr")
            header.write_text(
                "DataFile=sub-01_eeg.eeg\nMarkerFile=sub-01_eeg.vmrk\n",
                encoding="latin-1",
            )
            self.assertEqual(
                _gabor(root),
                [
                    (
                        "sub-01",
                        "0",
                        "source exclusion: incomplete BrainVision file set",
                    )
                ],
            )

    def test_roster_excludes_kronemer_without_task_raw(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            participant = root / "579_NRP"
            self._touch(participant / ".full_extraction_complete")
            self.assertEqual(
                _kronemer(root),
                [
                    (
                        "579_NRP",
                        "0",
                        "source exclusion: no task EEG recording",
                    )
                ],
            )

    def test_repository_adapters_are_no_longer_placeholders(self) -> None:
        for dataset in ("gabor", "kronemer", "somato"):
            self.assertIsInstance(get_adapter(dataset), VerifiedRepositoryAdapter)

    def test_spawned_stimulus_prefetch_preserves_deterministic_batches(self) -> None:
        config = {
            "image_size": [64, 64],
            "target_present_probability": 2 / 3,
            "contrast_noise_levels": 12,
        }
        indices = [np.asarray([9, 2]), np.asarray([7, 4])]
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_stimulus_worker,
            initargs=(config, 123),
        ) as executor:
            prefetched = list(_prefetched_batches(executor, indices, prefetch=2))
        for batch_indices, generated in zip(indices, prefetched, strict=True):
            expected = generate_stimulus_batch(batch_indices, seed=123)
            np.testing.assert_array_equal(generated[0], expected.images)
            np.testing.assert_array_equal(generated[1], expected.task_cues)


if __name__ == "__main__":
    unittest.main()
