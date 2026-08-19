import unittest

from jacaccess.machine.human_adaptation import performance_gate


class HumanAdaptationOutputTests(unittest.TestCase):
    def test_performance_gate_boundary(self) -> None:
        self.assertTrue(performance_gate(0.02))
        self.assertTrue(performance_gate(-0.02))
        self.assertFalse(performance_gate(0.020001))
        self.assertFalse(performance_gate(-0.020001))


if __name__ == "__main__":
    unittest.main()
