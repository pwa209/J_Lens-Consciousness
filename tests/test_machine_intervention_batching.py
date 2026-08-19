import unittest


class MachineInterventionBatchingTests(unittest.TestCase):
    def test_batched_scores_equal_serial_scores(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional locally")
        from jacaccess.machine.analyze import _score, _scores_by_repeat

        repeats, batch = 4, 7
        generator = torch.Generator().manual_seed(7)
        labels = {
            "a": torch.randint(0, 3, (batch,), generator=generator),
            "b": torch.randint(0, 2, (batch,), generator=generator),
        }
        masks = {
            "a": torch.ones(batch, dtype=torch.bool),
            "b": torch.tensor([True, False, True, True, False, True, True]),
        }
        logits = {
            "a": torch.randn(repeats * batch, 3, generator=generator),
            "b": torch.randn(repeats * batch, 2, generator=generator),
        }
        batched = _scores_by_repeat(logits, labels, masks, repeats)
        serial = []
        for index in range(repeats):
            subset = {
                name: values[index * batch : (index + 1) * batch]
                for name, values in logits.items()
            }
            serial.append(_score(subset, labels, masks))
        self.assertTrue(torch.allclose(batched, torch.tensor(serial)))


if __name__ == "__main__":
    unittest.main()
