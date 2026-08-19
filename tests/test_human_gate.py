from __future__ import annotations

from collections import Counter

from automation.check_human_gate import dataset_failure_audit


def test_dataset_failure_audit_allows_rates_below_ceiling() -> None:
    fractions, failures = dataset_failure_audit(
        Counter({"kronemer": 8}),
        Counter({"gabor": 140, "kronemer": 580, "somato": 145}),
        0.20,
    )

    assert fractions == {
        "gabor": 0.0,
        "kronemer": 8 / 580,
        "somato": 0.0,
    }
    assert failures == []


def test_dataset_failure_audit_blocks_rates_above_ceiling() -> None:
    fractions, failures = dataset_failure_audit(
        Counter({"somato": 30}), Counter({"somato": 100}), 0.20
    )

    assert fractions == {"somato": 0.30}
    assert failures == ["somato: fold failure fraction 0.300 exceeds 0.2"]
