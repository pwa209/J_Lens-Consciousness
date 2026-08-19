from __future__ import annotations

import unittest

import numpy as np

from jacaccess.stats.bayes import directional_jzs_bayes_factor


class DirectionalBayesFactorTests(unittest.TestCase):
    def test_favors_a_clear_positive_effect(self) -> None:
        values = np.array([0.25, 0.31, 0.18, 0.29, 0.35, 0.22, 0.28, 0.33])
        self.assertGreater(directional_jzs_bayes_factor(values), 1.0)

    def test_positive_direction_rejects_same_negative_effect(self) -> None:
        values = np.array([0.25, 0.31, 0.18, 0.29, 0.35, 0.22, 0.28, 0.33])
        positive = directional_jzs_bayes_factor(values)
        negative = directional_jzs_bayes_factor(-values)
        self.assertGreater(positive, negative)
        self.assertLess(negative, 1.0)


if __name__ == "__main__":
    unittest.main()
