"""Leakage-safe nested elastic-net test of incremental Jacobian prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _candidate_parameters() -> list[tuple[float, float]]:
    return [(c, ratio) for c in (0.01, 0.1, 1.0, 10.0) for ratio in (0.0, 0.5, 1.0)]


def _pipeline(c: float, ratio: float, seed: int) -> object:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=c,
                    l1_ratio=ratio,
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _score_candidate(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    c: float,
    ratio: float,
    seed: int,
) -> tuple[float, float, float]:
    """Score one hyperparameter pair across fixed inner folds.

    Keeping this worker at module scope lets joblib share/memmap the numeric
    design matrix instead of repeatedly serialising the full pandas table.
    """
    from sklearn.metrics import roc_auc_score

    scores: list[float] = []
    for inner_train, inner_test in splits:
        model = _pipeline(c, ratio, seed)
        model.fit(x[inner_train], y[inner_train])
        probability = model.predict_proba(x[inner_test])[:, 1]
        scores.append(float(roc_auc_score(y[inner_test], probability)))
    return c, ratio, float(np.mean(scores))


def nested_incremental_auc(
    table: pd.DataFrame,
    *,
    outcome: str,
    group: str,
    seed: int = 20260730,
    outer_folds: int = 5,
    inner_folds: int = 4,
    jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    if jobs < 1:
        raise ValueError("jobs must be at least 1")

    conventional = [
        name for name in table if name.startswith(("erp_", "gfp_", "power_"))
    ]
    jacobian = [name for name in table if name.startswith("jacobian_")]
    if not conventional or not jacobian:
        raise ValueError("both conventional and Jacobian feature families are required")
    raw_outcome = table[outcome]
    levels = sorted(raw_outcome.dropna().astype(str).unique())
    if len(levels) != 2:
        raise ValueError(f"outcome must have exactly two levels, found {levels}")
    y = raw_outcome.astype(str).map({levels[0]: 0, levels[1]: 1}).to_numpy()
    groups = table[group].astype(str).to_numpy()
    outer = StratifiedGroupKFold(outer_folds, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(outer.split(table, y, groups)):
        inner = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=seed + fold)
        for family, columns in (
            ("conventional", conventional),
            ("conventional_plus_jacobian", conventional + jacobian),
        ):
            x_train = table.iloc[train][columns].to_numpy(dtype=float)
            y_train = y[train]
            inner_splits = list(inner.split(x_train, y_train, groups[train]))
            candidates = _candidate_parameters()
            scored = Parallel(n_jobs=min(jobs, len(candidates)), prefer="processes")(
                delayed(_score_candidate)(
                    x_train,
                    y_train,
                    inner_splits,
                    c=c,
                    ratio=ratio,
                    seed=seed + fold,
                )
                for c, ratio in candidates
            )
            # ``max`` preserves the original candidate order for exact ties,
            # matching the prior sequential implementation.
            best_c, best_ratio, _ = max(scored, key=lambda item: item[2])
            best = (best_c, best_ratio)
            model = _pipeline(*best, seed + fold)
            model.fit(table.iloc[train][columns], y[train])
            probability = model.predict_proba(table.iloc[test][columns])[:, 1]
            rows.append(
                {
                    "fold": fold,
                    "family": family,
                    "auc": roc_auc_score(y[test], probability),
                    "c": best[0],
                    "l1_ratio": best[1],
                    "test_n": len(test),
                }
            )
    fold_results = pd.DataFrame(rows)
    paired = fold_results.pivot(index="fold", columns="family", values="auc")
    increments = paired["conventional_plus_jacobian"] - paired["conventional"]
    rng = np.random.default_rng(seed)
    observed = float(increments.mean())
    signs = rng.choice((-1.0, 1.0), size=(10000, len(increments)))
    p_value = float((1 + np.sum((signs * increments.to_numpy()).mean(1) >= observed)) / 10001)
    summary = {
        "outcome": outcome,
        "outer_folds": outer_folds,
        "mean_conventional_auc": float(paired["conventional"].mean()),
        "mean_augmented_auc": float(paired["conventional_plus_jacobian"].mean()),
        "mean_incremental_auc": observed,
        "directional_sign_flip_p": p_value,
    }
    return fold_results, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--group", default="analysis_group")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    table = pd.read_parquet(args.features)
    folds, summary = nested_incremental_auc(
        table, outcome=args.outcome, group=args.group, jobs=args.jobs
    )
    args.output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output / "fold-results.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
