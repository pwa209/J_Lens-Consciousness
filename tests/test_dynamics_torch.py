from __future__ import annotations

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class TorchDynamicsTests(unittest.TestCase):
    def test_analytic_jacobian_matches_autograd(self) -> None:
        import torch

        from jacaccess.latent.dynamics import ResidualDynamics

        torch.manual_seed(4)
        model = ResidualDynamics(
            state_dimensions=5,
            hidden_dimensions=7,
            input_dimensions=3,
        ).double()
        states = torch.randn(4, 5, dtype=torch.float64)
        inputs = torch.randn(4, 3, dtype=torch.float64)
        analytic = model.analytic_jacobian(states, inputs)
        automatic = torch.stack(
            [
                torch.autograd.functional.jacobian(
                    lambda value: model.step(value[None], inputs[index : index + 1])[0],
                    states[index],
                )
                for index in range(states.shape[0])
            ]
        )
        relative_error = torch.linalg.norm(analytic - automatic) / torch.linalg.norm(automatic)
        self.assertLess(float(relative_error), 1e-10)
        self.assertTrue(np.isfinite(float(relative_error)))


if __name__ == "__main__":
    unittest.main()

