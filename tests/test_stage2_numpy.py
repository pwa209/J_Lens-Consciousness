from __future__ import annotations

import unittest

import numpy as np

from jacaccess.config import (
    configuration_hash,
    validate_analysis_config,
    validate_machine_config,
)
from jacaccess.latent.inputs import build_physical_inputs
from jacaccess.latent.readouts import fit_ridge_readout
from jacaccess.machine.stimuli import generate_stimulus_batch


class ConfigurationTests(unittest.TestCase):
    def test_hash_is_key_order_invariant(self) -> None:
        self.assertEqual(configuration_hash({"a": 1, "b": 2}), configuration_hash({"b": 2, "a": 1}))

    def test_ordinary_analysis_contract(self) -> None:
        valid = {
            "study": {"registered_report_claim": False},
            "crossfit": {"folds": 5, "latent_dimensions": 32},
            "jacobian": {
                "horizons_samples": [2, 4, 8, 16],
                "rank": 4,
                "component_weights": [1 / 3, 1 / 3, 1 / 3],
                "persist_full_tensors": False,
            },
            "compute": {"local_scratch_gb_minimum": 3000},
        }
        self.assertEqual(validate_analysis_config(valid), [])
        invalid = {**valid, "study": {"registered_report_claim": True}}
        self.assertTrue(validate_analysis_config(invalid))

    def test_machine_contract(self) -> None:
        config = {
            "output_heads": [
                "presence",
                "orientation",
                "location",
                "contrast_bin",
                "delayed_action",
            ],
            "integration_state_dimensions": 32,
            "internal_steps": 6,
            "seeds": 20,
            "parameter_target": 2_000_000,
            "parameter_tolerance_fraction": 0.10,
        }
        self.assertEqual(validate_machine_config(config), [])


class PhysicalInputTests(unittest.TestCase):
    def test_shapes_and_impulse_locations(self) -> None:
        times = np.linspace(-0.2, 0.7, 10)
        inputs = build_physical_inputs(
            times_seconds=times,
            trial_count=3,
            impulse_onsets_seconds={"target_onset": 0.0, "mask_onset": np.array([0.3, 0.3, np.nan])},
            trial_covariates={"physical_contrast": np.array([0.1, 0.2, 0.3])},
            time_basis_count=4,
        )
        self.assertEqual(inputs.values.shape, (3, 10, 7))
        self.assertEqual(np.sum(inputs.values[:, :, 0]), 3)
        self.assertEqual(np.sum(inputs.values[:, :, 1]), 2)

    def test_condition_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "awareness"):
            build_physical_inputs(
                times_seconds=np.linspace(-0.2, 0.7, 10),
                trial_count=2,
                impulse_onsets_seconds={},
                trial_covariates={"awareness": np.array([0, 1])},
            )


class ReadoutTests(unittest.TestCase):
    def test_ridge_recovers_linear_map(self) -> None:
        rng = np.random.default_rng(8)
        latent = rng.normal(size=(1000, 5))
        true_weight = rng.normal(size=(3, 5))
        targets = latent @ true_weight.T + 0.01 * rng.normal(size=(1000, 3))
        fitted = fit_ridge_readout("regional", latent, targets)
        np.testing.assert_allclose(fitted.weight, true_weight, atol=2e-3)
        self.assertGreater(fitted.training_score, 0.99)


class StimulusTests(unittest.TestCase):
    def test_generation_is_index_deterministic(self) -> None:
        first = generate_stimulus_batch(np.array([9, 2, 7]))
        second = generate_stimulus_batch(np.array([7, 9, 2]))
        lookup = {index: position for position, index in enumerate([7, 9, 2])}
        for position, index in enumerate([9, 2, 7]):
            np.testing.assert_array_equal(first.images[position], second.images[lookup[index]])

    def test_task_shapes_and_masks(self) -> None:
        batch = generate_stimulus_batch(np.arange(32))
        self.assertEqual(batch.images.shape, (32, 1, 64, 64))
        self.assertEqual(batch.task_cues.shape, (32, 5))
        np.testing.assert_allclose(batch.task_cues.sum(axis=1), 1.0)
        self.assertEqual(set(batch.labels), set(batch.valid_masks))
        np.testing.assert_array_equal(
            batch.valid_masks["orientation"],
            batch.labels["presence"].astype(bool),
        )


if __name__ == "__main__":
    unittest.main()

