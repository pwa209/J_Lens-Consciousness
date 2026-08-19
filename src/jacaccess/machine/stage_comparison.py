"""Held-out human geometry evaluation for Experiment 2 machine stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jacaccess.stats.theory_comparison import METRICS, human_contrasts, machine_contrasts


def _reference(
    human: pd.DataFrame,
    participants: list[str],
    contrast_id: str,
) -> tuple[pd.Series, pd.Series]:
    selected = human[
        human["contrast_id"].eq(contrast_id)
        & human["participant_id"].astype(str).isin(participants)
    ]
    center = selected.groupby("metric")["contrast"].mean().reindex(METRICS)
    scale = selected.groupby("metric")["contrast"].std().reindex(METRICS).clip(lower=1e-4)
    if center.isna().any() or scale.isna().any():
        raise ValueError(f"incomplete held-out human reference for {contrast_id}")
    return center, scale


def evaluate_signature(
    signatures: Path,
    *,
    architecture: str,
    seed: int,
    stage: str,
    outer_fold: int,
    center: pd.Series,
    scale: pd.Series,
    accuracy_matching: dict[str, Any],
    fallback_weights: Path,
    human_contrast: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    machine = machine_contrasts(
        pd.read_parquet(signatures), accuracy_matching, fallback_weights
    )
    vector = (
        machine[
            machine["architecture"].eq(architecture) & machine["seed"].eq(seed)
        ]
        .set_index("metric")["contrast"]
        .reindex(METRICS)
    )
    if vector.isna().any():
        raise ValueError(f"incomplete machine geometry for {architecture}/{seed}/{stage}")
    residual = (vector - center) / scale
    human_values = center.to_numpy(float)
    machine_values = vector.to_numpy(float)
    denominator = np.linalg.norm(human_values) * np.linalg.norm(machine_values)
    summary = {
        "architecture": architecture,
        "seed": seed,
        "outer_fold": outer_fold,
        "stage": stage,
        "rms_distance": float(np.sqrt(np.mean(residual.to_numpy(float) ** 2))),
        "cosine_similarity": (
            float(np.dot(human_values, machine_values) / denominator)
            if denominator > 0
            else float("nan")
        ),
        "magnitude_ratio": float(
            np.linalg.norm(machine_values) / max(np.linalg.norm(human_values), 1e-12)
        ),
        "valid": True,
        "invalid_reason": "",
    }
    rows = [
        {
            "architecture": architecture,
            "seed": seed,
            "outer_fold": outer_fold,
            "stage": stage,
            "metric": metric,
            "machine_contrast": float(vector[metric]),
            "human_heldout_contrast": float(center[metric]),
            "standardized_residual": float(residual[metric]),
            "human_contrast_id": human_contrast,
            "valid": True,
            "invalid_reason": "",
        }
        for metric in METRICS
    ]
    return rows, summary


def aggregate_stage_geometry(
    *,
    experiment_root: Path,
    human_table: Path,
    split_manifest: Path,
    dataset_config_root: Path,
    original_machine_root: Path,
    accuracy_matching_path: Path,
    fallback_weights: Path,
    output_root: Path,
) -> dict[str, Any]:
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    human = human_contrasts(pd.read_parquet(human_table), dataset_config_root)
    matching = json.loads(accuracy_matching_path.read_text(encoding="utf-8"))
    geometry_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    external_references: dict[str, tuple[pd.Series, pd.Series]] = {}
    for contrast_id in ("kronemer-0", "kronemer-1", "somato-0", "somato-1"):
        participants = human.loc[
            human["contrast_id"].eq(contrast_id), "participant_id"
        ].astype(str).unique().tolist()
        external_references[contrast_id] = _reference(human, participants, contrast_id)
    for fold_record in split["folds"]:
        fold = int(fold_record["outer_fold"])
        center, scale = _reference(human, fold_record["heldout"], "gabor-0")
        for architecture_dir in sorted((experiment_root / "checkpoints").glob("*")):
            if not architecture_dir.is_dir():
                continue
            architecture = architecture_dir.name
            for seed_dir in sorted(architecture_dir.glob("seed-*")):
                seed = int(seed_dir.name.split("-")[-1])
                repaired_task_trained = (
                    experiment_root
                    / "stage-analysis"
                    / "task_trained"
                    / architecture
                    / seed_dir.name
                    / "jacobian-signatures.parquet"
                )
                original_task_trained = (
                    original_machine_root
                    / architecture
                    / seed_dir.name
                    / "jacobian-signatures.parquet"
                )
                paths = {
                    "random_init": experiment_root
                    / "stage-analysis"
                    / "random_init"
                    / architecture
                    / seed_dir.name
                    / "jacobian-signatures.parquet",
                    "task_trained": (
                        repaired_task_trained
                        if repaired_task_trained.exists()
                        else original_task_trained
                    ),
                    "human_adapted": experiment_root
                    / "stage-analysis"
                    / "human_adapted"
                    / architecture
                    / seed_dir.name
                    / f"fold-{fold}"
                    / "jacobian-signatures.parquet",
                    "sham_adapted": experiment_root
                    / "stage-analysis"
                    / "sham_adapted"
                    / architecture
                    / seed_dir.name
                    / f"fold-{fold}"
                    / "jacobian-signatures.parquet",
                }
                for stage, path in paths.items():
                    if not path.exists():
                        continue
                    rows, summary = evaluate_signature(
                        path,
                        architecture=architecture,
                        seed=seed,
                        stage=stage,
                        outer_fold=fold,
                        center=center,
                        scale=scale,
                        accuracy_matching=matching,
                        fallback_weights=fallback_weights,
                        human_contrast="gabor-0",
                    )
                    performance_path = path.parent / "test-presence-by-bin.parquet"
                    performance = pd.read_parquet(performance_path)
                    selected_bin = int(matching.get("selected_threshold_bin", 2))
                    selected_performance = performance[
                        performance["difficulty_bin"].eq(selected_bin)
                    ]
                    if len(selected_performance) != 1:
                        raise ValueError(
                            f"missing unique task performance for {path} at bin {selected_bin}"
                        )
                    summary["task_accuracy"] = float(
                        selected_performance["presence_accuracy"].iloc[0]
                    )
                    geometry_rows.extend(rows)
                    summaries.append(summary)
                    if stage in {"task_trained", "human_adapted", "sham_adapted"}:
                        vector = pd.Series(
                            {str(row["metric"]): float(row["machine_contrast"]) for row in rows}
                        ).reindex(METRICS)
                        for contrast_id, (external_center, external_scale) in external_references.items():
                            residual = (vector - external_center) / external_scale
                            denominator = np.linalg.norm(vector) * np.linalg.norm(external_center)
                            transfer_rows.append(
                                {
                                    "architecture": architecture,
                                    "seed": seed,
                                    "outer_fold": fold,
                                    "stage": stage,
                                    "adaptation_dataset": "gabor",
                                    "evaluation_contrast": contrast_id,
                                    "rms_distance": float(
                                        np.sqrt(np.mean(residual.to_numpy(float) ** 2))
                                    ),
                                    "cosine_similarity": (
                                        float(np.dot(vector, external_center) / denominator)
                                        if denominator > 0
                                        else float("nan")
                                    ),
                                }
                            )
    geometry = pd.DataFrame(geometry_rows)
    summary = pd.DataFrame(summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(output_root / "stage-geometry.csv", index=False)
    summary.to_csv(output_root / "fold-stage-summary.csv", index=False)
    seed_summary = (
        summary.groupby(["architecture", "seed", "stage"], as_index=False)
        .agg(
            rms_distance=("rms_distance", "mean"),
            cosine_similarity=("cosine_similarity", "mean"),
            magnitude_ratio=("magnitude_ratio", "mean"),
            task_accuracy=("task_accuracy", "mean"),
            valid=("valid", "all"),
        )
    )
    seed_summary["task_loss"] = np.nan
    seed_summary["invalid_reason"] = ""
    seed_summary.to_csv(output_root / "seed-stage-summary.csv", index=False)
    architecture = (
        seed_summary.groupby(["architecture", "stage"], as_index=False)
        .agg(
            mean_rms_distance=("rms_distance", "mean"),
            sd_rms_distance=("rms_distance", "std"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            mean_magnitude_ratio=("magnitude_ratio", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    architecture.to_csv(output_root / "architecture-stage-summary.csv", index=False)
    pd.DataFrame(transfer_rows).to_csv(output_root / "external-transfer.csv", index=False)
    return {
        "stage_geometry_rows": len(geometry),
        "fold_stage_rows": len(summary),
        "seed_stage_rows": len(seed_summary),
        "architectures": int(seed_summary["architecture"].nunique()) if len(seed_summary) else 0,
        "external_transfer_rows": len(transfer_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--human-table", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-config-root", type=Path, default=Path("configs/datasets"))
    parser.add_argument("--original-machine-root", type=Path, required=True)
    parser.add_argument("--accuracy-matching", type=Path, required=True)
    parser.add_argument("--fallback-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate_stage_geometry(
                experiment_root=args.experiment_root,
                human_table=args.human_table,
                split_manifest=args.split_manifest,
                dataset_config_root=args.dataset_config_root,
                original_machine_root=args.original_machine_root,
                accuracy_matching_path=args.accuracy_matching,
                fallback_weights=args.fallback_weights,
                output_root=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
