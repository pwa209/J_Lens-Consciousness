"""Five-theory human-machine comparison for the ordinary research article."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from jacaccess.config import load_yaml

HUMAN_COLUMNS = {
    "gain": "z_log_gain",
    "broadcast": "z_logit_broadcast",
    "persistence": "z_fisher_persistence",
    "concentration": "z_logit_concentration",
}
METRICS = tuple(HUMAN_COLUMNS)


def _condition_key(value: object) -> str:
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


def _transform_machine(table: pd.DataFrame, epsilon: float = 1e-4) -> pd.DataFrame:
    result = table.copy()
    result["gain"] = np.log(result["gain"].clip(lower=epsilon))
    clipped = result["broadcast"].clip(epsilon, 1 - epsilon)
    result["broadcast"] = np.log(clipped / (1 - clipped))
    result["persistence"] = np.arctanh(
        np.sqrt(result["persistence"].clip(0, 1 - epsilon))
    )
    clipped = result["concentration"].clip(epsilon, 1 - epsilon)
    result["concentration"] = np.log(clipped / (1 - clipped))
    for metric in METRICS:
        grouped = result.groupby(["architecture", "seed"])[metric]
        mean = grouped.transform("mean")
        scale = grouped.transform("std").clip(lower=epsilon)
        result[metric] = (result[metric] - mean) / scale
    return result


def human_contrasts(human: pd.DataFrame, config_root: Path) -> pd.DataFrame:
    missing = set(HUMAN_COLUMNS.values()) - set(human)
    if missing:
        raise ValueError(
            "human table lacks training-baseline-standardized components: "
            f"{sorted(missing)}"
        )
    records: list[pd.DataFrame] = []
    for dataset in ("gabor", "kronemer", "somato"):
        config = load_yaml(config_root / f"{dataset}.yaml")
        for index, contrast in enumerate(config["primary_contrasts"]):
            low, high = (float(value) / 1000 for value in contrast["window_ms"])
            selected = human[
                (human["dataset_id"] == dataset)
                & human["time_seconds"].between(low, high, inclusive="both")
            ]
            field = contrast["condition_field"]
            trial = (
                selected.groupby(["participant_id", "original_trial_id", field], as_index=False)[
                    list(HUMAN_COLUMNS.values())
                ]
                .mean()
            )
            levels = trial[field].map(_condition_key)
            positive = trial[levels == _condition_key(contrast["positive"])]
            negative = trial[levels == _condition_key(contrast["negative"])]
            positive = positive.groupby("participant_id")[list(HUMAN_COLUMNS.values())].mean()
            negative = negative.groupby("participant_id")[list(HUMAN_COLUMNS.values())].mean()
            difference = positive.subtract(negative).dropna()
            difference = difference.rename(columns={value: key for key, value in HUMAN_COLUMNS.items()})
            long = difference.rename_axis("participant_id").reset_index().melt(
                id_vars="participant_id", var_name="metric", value_name="contrast"
            )
            long.insert(0, "contrast_id", f"{dataset}-{index}")
            long.insert(0, "dataset_id", dataset)
            records.append(long)
    if not records:
        raise ValueError("no human contrasts could be constructed")
    return pd.concat(records, ignore_index=True)


def machine_contrasts(
    signatures: pd.DataFrame,
    matching: dict[str, object],
    weights_path: Path,
) -> pd.DataFrame:
    required = {"architecture", "seed", "presence_correct", "difficulty_bin", *METRICS}
    if required - set(signatures):
        raise ValueError(f"machine signatures lack {sorted(required - set(signatures))}")
    table = _transform_machine(signatures)
    selected_bin = matching.get("selected_threshold_bin")
    if selected_bin is not None:
        table = table[table["difficulty_bin"] == int(selected_bin)].copy()
        table["analysis_weight"] = 1.0
    else:
        weights = pd.read_parquet(weights_path)
        table = table.merge(
            weights[
                [
                    "architecture", "seed", "difficulty_bin",
                    "weight_if_correct", "weight_if_incorrect",
                ]
            ],
            on=["architecture", "seed", "difficulty_bin"],
            validate="many_to_one",
        )
        table["analysis_weight"] = np.where(
            table["presence_correct"], table["weight_if_correct"], table["weight_if_incorrect"]
        )
    rows: list[dict[str, object]] = []
    for (architecture, seed), group in table.groupby(["architecture", "seed"], sort=True):
        for metric in METRICS:
            valid = group[[metric, "presence_correct", "analysis_weight"]].dropna()
            values = {}
            estimable = True
            for correct in (False, True):
                subset = valid[valid["presence_correct"] == correct]
                weight_sum = float(subset["analysis_weight"].sum())
                if subset.empty or not np.isfinite(weight_sum) or weight_sum <= 0:
                    estimable = False
                    break
                values[correct] = np.average(
                    subset[metric], weights=subset["analysis_weight"]
                )
            if not estimable:
                continue
            rows.append(
                {
                    "architecture": architecture,
                    "seed": int(seed),
                    "metric": metric,
                    "contrast": float(values[True] - values[False]),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        complete = (
            result.groupby(["architecture", "seed"])["metric"]
            .nunique()
            .eq(len(METRICS))
            .rename("complete")
            .reset_index()
        )
        result = result.merge(
            complete[complete["complete"]][["architecture", "seed"]],
            on=["architecture", "seed"],
            how="inner",
            validate="many_to_one",
        )
    if result.empty:
        raise ValueError(
            "no machine seed has a complete four-metric correct-minus-incorrect contrast"
        )
    return result


def _interval(values: np.ndarray, confidence: float = 0.90) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    sem = stats.sem(values)
    if not np.isfinite(sem) or sem == 0:
        return mean, mean
    low, high = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
    return float(low), float(high)


def leave_one_dataset_out(human: pd.DataFrame) -> list[dict[str, object]]:
    """Evaluate each dataset against a signature estimated from the other two.

    Contrasts are weighted equally when estimating the training signature so a
    dataset with two contrasts cannot dominate a dataset with one contrast.
    """

    contrast_means = (
        human.groupby(["dataset_id", "contrast_id", "metric"], as_index=False)["contrast"]
        .mean()
    )
    records: list[dict[str, object]] = []
    for held_out in sorted(human["dataset_id"].unique()):
        training = human[human["dataset_id"] != held_out]
        training_means = contrast_means[contrast_means["dataset_id"] != held_out]
        center = training_means.groupby("metric")["contrast"].mean().reindex(METRICS)
        scale = training.groupby("metric")["contrast"].std().reindex(METRICS).clip(lower=1e-4)
        for contrast_id, values in contrast_means[
            contrast_means["dataset_id"] == held_out
        ].groupby("contrast_id"):
            vector = values.set_index("metric")["contrast"].reindex(METRICS)
            standardized_distance = float(np.sqrt(np.mean(((vector - center) / scale) ** 2)))
            center_values = center.to_numpy(float)
            vector_values = vector.to_numpy(float)
            denominator = np.linalg.norm(center_values) * np.linalg.norm(vector_values)
            records.append(
                {
                    "held_out_dataset": str(held_out),
                    "contrast_id": str(contrast_id),
                    "training_datasets": sorted(
                        str(value) for value in human.loc[human["dataset_id"] != held_out, "dataset_id"].unique()
                    ),
                    "standardized_distance": standardized_distance,
                    "cosine_similarity": (
                        float(np.dot(center_values, vector_values) / denominator)
                        if denominator > 0 else float("nan")
                    ),
                    "same_direction_fraction": float(
                        np.mean(np.sign(vector_values) == np.sign(center_values))
                    ),
                }
            )
    return records


def compare_theories(human: pd.DataFrame, machine: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    discovery = human[human["contrast_id"] == "gabor-0"]
    center = discovery.groupby("metric")["contrast"].mean().reindex(METRICS)
    scale = discovery.groupby("metric")["contrast"].std().reindex(METRICS).clip(lower=1e-4)
    if center.isna().any() or scale.isna().any():
        raise ValueError("Gabor discovery signature is incomplete")
    wide = machine.pivot(index=["architecture", "seed"], columns="metric", values="contrast")
    standardized_error = (wide[list(METRICS)] - center) / scale
    distances = np.sqrt(np.mean(standardized_error**2, axis=1)).rename("rms_distance")
    comparison = distances.reset_index()
    human_vector = center.to_numpy(dtype=float)
    machine_vectors = wide.loc[
        pd.MultiIndex.from_frame(comparison[["architecture", "seed"]])
    ].to_numpy()
    human_norm = np.linalg.norm(human_vector)
    comparison["cosine_similarity"] = [
        (
            float(np.dot(row, human_vector) / (np.linalg.norm(row) * human_norm))
            if np.linalg.norm(row) > 0 and human_norm > 0
            else float("nan")
        )
        for row in machine_vectors
    ]
    comparison["magnitude_ratio"] = [
        float(np.linalg.norm(row) / np.linalg.norm(human_vector))
        for row in machine_vectors
    ]
    ranking = (
        comparison.groupby("architecture", as_index=False)
        .agg(
            mean_rms_distance=("rms_distance", "mean"),
            sd_rms_distance=("rms_distance", "std"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            mean_magnitude_ratio=("magnitude_ratio", "mean"),
            seeds=("seed", "count"),
        )
        .sort_values("mean_rms_distance")
    )
    constrained = wide.loc["shared_workspace"]
    unlimited = wide.loc["unlimited_shared_state"]
    common = constrained.index.intersection(unlimited.index)
    equivalence: dict[str, object] = {}
    all_equivalent = True
    for metric in METRICS:
        difference = (
            constrained.loc[common, metric] - unlimited.loc[common, metric]
        ).to_numpy() / float(scale[metric])
        low, high = _interval(difference)
        finite_difference = difference[np.isfinite(difference)]
        equivalent = low > -0.20 and high < 0.20
        all_equivalent &= equivalent
        equivalence[metric] = {
            "mean_standardized_difference": float(np.mean(finite_difference)),
            "ci90": [low, high],
            "equivalent_within_0.20": equivalent,
        }
    replication: dict[str, object] = {}
    for contrast_id, values in human.groupby("contrast_id"):
        vector = values.groupby("metric")["contrast"].mean().reindex(METRICS)
        replication[contrast_id] = {
            "standardized_distance_from_gabor_discovery": float(
                np.sqrt(np.mean(((vector - center) / scale) ** 2))
            ),
            "same_direction_fraction": float(np.mean(np.sign(vector) == np.sign(center))),
            "participants": int(values["participant_id"].nunique()),
        }
    result = {
        "discovery_contrast": "gabor-0",
        "primary_metrics": list(METRICS),
        "winning_architecture": str(ranking.iloc[0]["architecture"]),
        "architecture_ranking": ranking.to_dict(orient="records"),
        "capacity_equivalence_margin_discovery_human_sd": 0.20,
        "constrained_vs_unlimited": equivalence,
        "all_primary_metrics_equivalent": all_equivalent,
        "capacity_limit_interpretation_falsified": bool(
            all_equivalent
            or ranking.set_index("architecture").loc["unlimited_shared_state", "mean_rms_distance"]
            <= ranking.set_index("architecture").loc["shared_workspace", "mean_rms_distance"]
        ),
        "replication": replication,
        "leave_one_dataset_out": leave_one_dataset_out(human),
    }
    return comparison, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--machine-signatures", type=Path, required=True)
    parser.add_argument("--accuracy-matching", type=Path, required=True)
    parser.add_argument("--fallback-weights", type=Path, required=True)
    parser.add_argument("--dataset-config-root", type=Path, default=Path("configs/datasets"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    human = human_contrasts(pd.read_parquet(args.human), args.dataset_config_root)
    matching = json.loads(args.accuracy_matching.read_text(encoding="utf-8"))
    machine = machine_contrasts(
        pd.read_parquet(args.machine_signatures), matching, args.fallback_weights
    )
    seed_comparison, result = compare_theories(human, machine)
    args.output.mkdir(parents=True, exist_ok=True)
    human.to_csv(args.output / "human-four-metric-contrasts.csv", index=False)
    machine.to_csv(args.output / "machine-four-metric-contrasts.csv", index=False)
    seed_comparison.to_csv(args.output / "architecture-seed-distances.csv", index=False)
    (args.output / "theory-comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
