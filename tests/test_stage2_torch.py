from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class Stage2TorchTests(unittest.TestCase):
    def test_human_loss_and_checkpoint_recovery(self) -> None:
        import torch

        from jacaccess.latent.dynamics import ResidualDynamics
        from jacaccess.latent.losses import multi_horizon_loss
        from jacaccess.latent.train import TrainingConfig, fit_residual_dynamics

        torch.manual_seed(1)
        states = torch.randn(8, 12, 5)
        inputs = torch.randn(8, 12, 2)
        model = ResidualDynamics(5, 7, 2)
        loss = multi_horizon_loss(model, states[:, :9], inputs[:, :8])
        self.assertTrue(torch.isfinite(loss.total))

        config = TrainingConfig(
            batch_transitions=32,
            max_epochs=2,
            patience=2,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "fold.pt"
            first = fit_residual_dynamics(
                training_states=states[:6],
                training_inputs=inputs[:6],
                validation_states=states[6:],
                validation_inputs=inputs[6:],
                hidden_dimensions=7,
                config=config,
                checkpoint_path=checkpoint,
                device="cpu",
            )
            self.assertTrue(checkpoint.is_file())
            resumed = fit_residual_dynamics(
                training_states=states[:6],
                training_inputs=inputs[:6],
                validation_states=states[6:],
                validation_inputs=inputs[6:],
                hidden_dimensions=7,
                config=config,
                checkpoint_path=checkpoint,
                device="cpu",
            )
            self.assertEqual(first.best_epoch, resumed.best_epoch)

    def test_parameter_matched_machine_models(self) -> None:
        import torch

        from jacaccess.machine.architectures import (
            build_architecture,
            count_parameters,
        )
        from jacaccess.machine.jacobian import exact_future_logit_jacobians

        images = torch.randn(2, 1, 64, 64)
        cues = torch.eye(5)[:2]
        for name in (
            "feedforward",
            "recurrent",
            "shared_workspace",
            "private_modules",
            "unlimited_shared_state",
        ):
            model = build_architecture(name)
            parameters = count_parameters(model)
            self.assertLessEqual(abs(parameters - 2_000_000) / 2_000_000, 0.10)
            states, logits = model(images, cues)
            expected_state = 128 if name == "unlimited_shared_state" else 32
            self.assertEqual(states.shape, (2, 6, expected_state))
            self.assertEqual(logits["contrast_bin"].shape, (2, 6, 12))

        small = build_architecture("recurrent")
        jacobians = exact_future_logit_jacobians(small, images[:1], cues[:1], steps=(4,))
        self.assertEqual(jacobians[4].shape[-1], 32)


if __name__ == "__main__":
    unittest.main()
