import inspect
import unittest
from pathlib import Path

from jacaccess.config import load_yaml
from jacaccess.machine import human_adaptation


class HumanAdaptationLeakageTests(unittest.TestCase):
    def test_config_explicitly_forbids_final_geometry(self) -> None:
        config = load_yaml(Path("configs/analysis/human_adaptation.yaml"))
        human_adaptation.validate_adaptation_config(config)
        self.assertFalse(config["adaptation"]["uses_final_geometry_in_loss"])
        self.assertFalse(config["adaptation"]["uses_final_geometry_for_checkpoint_selection"])

    def test_optimizer_module_does_not_import_geometry_scorers(self) -> None:
        source = inspect.getsource(human_adaptation)
        self.assertNotIn("jacaccess.jacobian", source)
        self.assertNotIn("theory_comparison", source)
        for term in human_adaptation.FORBIDDEN_TARGET_TERMS:
            self.assertNotIn(f'targets["{term}"]', source.lower())


if __name__ == "__main__":
    unittest.main()
