from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jacaccess.io.events import (
    assert_primary_model_fields_safe,
    validate_event_rows,
)
from jacaccess.io.manifest import build_manifest, verify_manifest
from jacaccess.preprocess.qc import participant_qc


class EventContractTests(unittest.TestCase):
    def test_duplicate_trial_key_fails(self) -> None:
        row = {
            "dataset_id": "x",
            "participant_id": "1",
            "original_trial_id": "9",
            "onset_seconds": 1.0,
            "event_type": "target",
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_event_rows([row, row])

    def test_forbidden_primary_field_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "awareness"):
            assert_primary_model_fields_safe(["physical_contrast", "awareness"])
        assert_primary_model_fields_safe(["physical_contrast", "location_x"])


class ManifestTests(unittest.TestCase):
    def test_changed_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "example.bin"
            source.write_bytes(b"before")
            entries = build_manifest([source], root)
            self.assertEqual(verify_manifest(entries, root), [])
            source.write_bytes(b"after")
            self.assertTrue(verify_manifest(entries, root))


class PreprocessingQCTests(unittest.TestCase):
    def test_bad_channel_boundary_is_excluded(self) -> None:
        result = participant_qc(0.15, 0.10)
        self.assertFalse(result.included)

    def test_ica_boundary_is_allowed(self) -> None:
        result = participant_qc(0.10, 0.20)
        self.assertTrue(result.included)


if __name__ == "__main__":
    unittest.main()

