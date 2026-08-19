from __future__ import annotations

import unittest

import numpy as np

from jacaccess.jacobian.metrics import (
    apply_access_index,
    compose_standardized_maps,
    fit_access_baseline,
    geometry_from_maps,
)
from jacaccess.jacobian.propagate import ordered_propagators


class PropagationTests(unittest.TestCase):
    def test_horizon_order_uses_latest_jacobian_on_the_left(self) -> None:
        a = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        b = np.array([[1.0, 0.0], [3.0, 1.0]], dtype=np.float64)
        jacobians = np.stack([a, b])[None, ...]
        result = ordered_propagators(jacobians, horizons=(2,))
        np.testing.assert_allclose(result[2][0, 0], b @ a)
        self.assertFalse(np.allclose(a @ b, b @ a))

    def test_every_valid_start_is_returned(self) -> None:
        jacobians = np.broadcast_to(np.eye(3), (2, 6, 3, 3)).copy()
        result = ordered_propagators(jacobians, horizons=(1, 4))
        self.assertEqual(result[1].shape, (2, 6, 3, 3))
        self.assertEqual(result[4].shape, (2, 3, 3, 3))


class MetricTests(unittest.TestCase):
    def test_gain_uses_output_dimension(self) -> None:
        maps = np.array([[[[3.0, 0.0], [0.0, 4.0]]]])
        metrics = geometry_from_maps(
            maps,
            {"a": slice(0, 1), "b": slice(1, 2)},
            rank=2,
            persistence_lag=1,
        )
        self.assertAlmostEqual(float(metrics.gain[0, 0]), 12.5)

    def test_equal_block_energy_has_unit_broadcast(self) -> None:
        maps = np.broadcast_to(np.eye(2), (1, 3, 2, 2)).copy()
        metrics = geometry_from_maps(
            maps,
            {"a": slice(0, 1), "b": slice(1, 2)},
            rank=2,
            persistence_lag=1,
        )
        np.testing.assert_allclose(metrics.broadcast, 1.0)
        np.testing.assert_allclose(metrics.persistence[:, :-1], 1.0)

    def test_rotation_invariance(self) -> None:
        rng = np.random.default_rng(9)
        maps = rng.normal(size=(3, 8, 10, 6))
        rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        blocks = {"a": slice(0, 4), "b": slice(4, 7), "c": slice(7, 10)}
        original = geometry_from_maps(maps, blocks, rank=4, persistence_lag=2)
        rotated = geometry_from_maps(maps @ rotation, blocks, rank=4, persistence_lag=2)
        for name in (
            "gain",
            "broadcast",
            "persistence",
            "concentration",
            "effective_rank",
        ):
            np.testing.assert_allclose(
                getattr(original, name),
                getattr(rotated, name),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )

    def test_composition_groups_horizons_by_output_block(self) -> None:
        p1 = np.broadcast_to(np.eye(2), (1, 4, 2, 2)).copy()
        p2 = np.broadcast_to(2 * np.eye(2), (1, 3, 2, 2)).copy()
        maps, blocks = compose_standardized_maps(
            {1: p1, 2: p2},
            {"neural": np.eye(2), "physical": np.ones((1, 2))},
            {"neural": np.ones(2), "physical": np.ones(1)},
        )
        self.assertEqual(maps.shape, (1, 3, 6, 2))
        self.assertEqual(blocks["neural"], slice(0, 4))
        self.assertEqual(blocks["physical"], slice(4, 6))

    def test_access_baseline_is_training_normalized(self) -> None:
        rng = np.random.default_rng(14)
        maps = rng.normal(size=(12, 12, 8, 6))
        metrics = geometry_from_maps(
            maps,
            {"a": slice(0, 4), "b": slice(4, 8)},
            rank=4,
            persistence_lag=2,
        )
        mask = np.zeros(12, dtype=bool)
        mask[:5] = True
        baseline = fit_access_baseline(metrics, mask)
        access = apply_access_index(metrics, baseline)
        self.assertAlmostEqual(float(np.nanmean(access[:, mask])), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()

