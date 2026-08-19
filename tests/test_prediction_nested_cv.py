from __future__ import annotations

import numpy as np
import pandas as pd

from jacaccess.prediction.nested_cv import nested_incremental_auc


def _synthetic_table() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    groups = np.repeat(np.arange(12), 8)
    outcome = np.tile([0, 1], len(groups) // 2)
    signal = outcome + rng.normal(0, 0.5, len(groups))
    return pd.DataFrame(
        {
            "analysis_group": groups,
            "outcome": outcome,
            "erp_mean": signal,
            "power_alpha": rng.normal(size=len(groups)),
            "jacobian_gain": signal + rng.normal(0, 0.2, len(groups)),
            "jacobian_persistence": rng.normal(size=len(groups)),
        }
    )


def test_parallel_grid_matches_sequential_grid() -> None:
    table = _synthetic_table()
    sequential_folds, sequential_summary = nested_incremental_auc(
        table,
        outcome="outcome",
        group="analysis_group",
        seed=123,
        outer_folds=2,
        inner_folds=2,
        jobs=1,
    )
    parallel_folds, parallel_summary = nested_incremental_auc(
        table,
        outcome="outcome",
        group="analysis_group",
        seed=123,
        outer_folds=2,
        inner_folds=2,
        jobs=2,
    )

    pd.testing.assert_frame_equal(sequential_folds, parallel_folds)
    assert sequential_summary == parallel_summary


def test_parallel_grid_rejects_zero_jobs() -> None:
    table = _synthetic_table()
    try:
        nested_incremental_auc(
            table,
            outcome="outcome",
            group="analysis_group",
            outer_folds=2,
            inner_folds=2,
            jobs=0,
        )
    except ValueError as error:
        assert str(error) == "jobs must be at least 1"
    else:
        raise AssertionError("jobs=0 should fail")
