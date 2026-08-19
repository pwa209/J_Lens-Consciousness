"""Combine participant prediction features with strict key validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jacaccess.config import load_yaml


def _level_key(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def harmonize_outcomes(table: pd.DataFrame, config_root: Path) -> pd.DataFrame:
    """Map each dataset's declared prediction contrast onto a common 0/1 target."""

    required = {"dataset_id", "outcome"}
    if required - set(table):
        raise ValueError(f"prediction features lack {sorted(required - set(table))}")
    records: list[pd.DataFrame] = []
    for dataset, group in table.groupby("dataset_id", sort=True):
        config = load_yaml(config_root / f"{dataset}.yaml")
        outcome_field = config["prediction_outcome"]
        contrasts = [
            contrast
            for contrast in config["primary_contrasts"]
            if contrast["condition_field"] == outcome_field
        ]
        level_pairs = {
            (_level_key(contrast["negative"]), _level_key(contrast["positive"]))
            for contrast in contrasts
        }
        if len(level_pairs) != 1:
            raise ValueError(
                f"dataset {dataset!r} does not declare one unambiguous prediction contrast"
            )
        negative, positive = next(iter(level_pairs))
        normalized = group.copy()
        normalized["outcome_source_level"] = normalized["outcome"].map(_level_key)
        normalized = normalized[
            normalized["outcome_source_level"].isin((negative, positive))
        ].copy()
        observed = set(normalized["outcome_source_level"])
        if observed != {negative, positive}:
            raise ValueError(
                f"dataset {dataset!r} prediction levels are incomplete: {sorted(observed)}"
            )
        normalized["outcome"] = normalized["outcome_source_level"].map(
            {negative: 0, positive: 1}
        ).astype("int8")
        records.append(normalized)
    if not records:
        raise ValueError("no prediction feature rows remain after outcome harmonization")
    return pd.concat(records, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-config-root", type=Path, default=Path("configs/datasets")
    )
    args = parser.parse_args()
    table = pd.concat([pd.read_parquet(path) for path in args.inputs], ignore_index=True)
    keys = ["dataset_id", "participant_id", "original_trial_id"]
    if table.duplicated(keys).any():
        raise ValueError("prediction feature inputs contain duplicate trial keys")
    table = harmonize_outcomes(table, args.dataset_config_root)
    table["analysis_group"] = (
        table["dataset_id"].astype(str) + "::" + table["participant_id"].astype(str)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
