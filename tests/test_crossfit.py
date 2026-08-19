from __future__ import annotations

import unittest

import numpy as np

from jacaccess.latent.folds import assign_folds, split_fold
from jacaccess.latent.pca import fit_pca_whitening


class FoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trial_ids = [f"{index:03d}" for index in range(100)]
        self.strata = ["seen" if index % 2 else "unseen" for index in range(100)]

    def test_assignment_is_deterministic_and_balanced(self) -> None:
        first = assign_folds("gabor", "01", self.trial_ids, self.strata)
        second = assign_folds("gabor", "01", self.trial_ids, self.strata)
        np.testing.assert_array_equal(first, second)
        for stratum in set(self.strata):
            counts = np.bincount(
                first[np.asarray(self.strata) == stratum],
                minlength=5,
            )
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_split_has_no_overlap(self) -> None:
        assignments = assign_folds("gabor", "01", self.trial_ids, self.strata)
        split = split_fold("gabor", "01", self.trial_ids, assignments, heldout_fold=2)
        train = set(split.train.tolist())
        validation = set(split.validation.tolist())
        test = set(split.test.tolist())
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, set(range(100)))


class PCATests(unittest.TestCase):
    def test_training_transform_is_white(self) -> None:
        rng = np.random.default_rng(22)
        mixing = rng.normal(size=(8, 8))
        values = rng.normal(size=(40, 20, 8)) @ mixing
        fitted = fit_pca_whitening(values, components=6)
        latent = fitted.transform(values).reshape(-1, 6)
        np.testing.assert_allclose(latent.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            np.cov(latent, rowvar=False),
            np.eye(6),
            atol=1e-6,
        )

    def test_heldout_values_do_not_change_fit(self) -> None:
        rng = np.random.default_rng(23)
        training = rng.normal(size=(100, 5))
        heldout = rng.normal(loc=100, size=(10, 5))
        before = fit_pca_whitening(training, components=3)
        _ = before.transform(heldout)
        after = fit_pca_whitening(training, components=3)
        np.testing.assert_array_equal(before.mean, after.mean)
        np.testing.assert_array_equal(before.components, after.components)

    def test_whitening_is_invariant_to_voltage_units(self) -> None:
        rng = np.random.default_rng(24)
        values = rng.normal(size=(50, 12, 8))
        volts = fit_pca_whitening(values * 1e-6, components=6)
        microvolts = fit_pca_whitening(values, components=6)
        np.testing.assert_allclose(
            volts.transform(values * 1e-6),
            microvolts.transform(values),
            atol=1e-9,
        )
        self.assertAlmostEqual(volts.epsilon * 1e12, microvolts.epsilon)

    def test_nonfinite_training_values_are_rejected_explicitly(self) -> None:
        values = np.ones((8, 4))
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fit_pca_whitening(values, components=2)


if __name__ == "__main__":
    unittest.main()
