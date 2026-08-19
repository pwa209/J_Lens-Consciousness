import unittest
from pathlib import Path


class HumanAdaptationDeterminismTests(unittest.TestCase):
    def test_exact_seed_reconstruction(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional locally")
        from jacaccess.config import load_yaml
        from jacaccess.machine.human_adaptation import reconstruct_random_model

        config = load_yaml(Path("configs/models/machine.yaml"))
        first = reconstruct_random_model("feedforward", 3, config, torch.device("cpu"))
        second = reconstruct_random_model("feedforward", 3, config, torch.device("cpu"))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
