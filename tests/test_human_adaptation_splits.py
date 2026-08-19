import unittest

from jacaccess.machine.human_adaptation import subject_splits


class HumanAdaptationSplitTests(unittest.TestCase):
    def test_subject_partitions_are_disjoint_complete_and_deterministic(self) -> None:
        participants = [f"participant-{index:02d}" for index in range(30)]
        first = subject_splits(participants, folds=5, validation_fraction=0.2, seed=17)
        second = subject_splits(participants, folds=5, validation_fraction=0.2, seed=17)
        self.assertEqual(first, second)
        heldout_once = []
        for split in first:
            train, validation, heldout = map(set, split.values())
            self.assertFalse(train & validation)
            self.assertFalse(train & heldout)
            self.assertFalse(validation & heldout)
            self.assertEqual(train | validation | heldout, set(participants))
            heldout_once.extend(heldout)
        self.assertEqual(sorted(heldout_once), sorted(participants))


if __name__ == "__main__":
    unittest.main()
