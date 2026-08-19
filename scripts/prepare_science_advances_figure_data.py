"""Freeze compact, renderer-neutral tables for the Science Advances figures.

The completed analysis remains authoritative.  This script only converts its
parquet/JSON/CSV outputs into small CSV tables that can be consumed natively by
R without requiring the R Arrow package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


ARCHITECTURES = (
    "feedforward",
    "recurrent",
    "private_modules",
    "shared_workspace",
    "unlimited_shared_state",
)
METRICS = ("gain", "broadcast", "persistence", "concentration")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def contrast_specs(config_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in ("gabor", "kronemer", "somato"):
        config = yaml.safe_load((config_root / f"{dataset}.yaml").read_text(encoding="utf-8"))
        for index, contrast in enumerate(config["primary_contrasts"]):
            rows.append({"contrast_id": f"{dataset}-{index}", "dataset": dataset, **contrast})
    return rows


def level(values: pd.Series, target: object) -> pd.Series:
    normalized = values.astype(str).str.replace(r"\.0$", "", regex=True)
    return normalized.eq(str(target).removesuffix(".0"))


def difference_timecourse(table: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    selected = table[table["dataset_id"].eq(spec["dataset"])].copy()
    field = str(spec["condition_field"])
    positive = level(selected[field], spec["positive"])
    negative = level(selected[field], spec["negative"])
    selected = selected[positive | negative].copy()
    selected["condition"] = np.where(positive[positive | negative], "positive", "negative")
    grouped = selected.groupby(["participant_id", "time_seconds", "condition"])["access_index"].mean().unstack()
    return (grouped["positive"] - grouped["negative"]).rename("difference").reset_index()


def write_csv(table: pd.DataFrame, output: Path, name: str, files: list[dict[str, Any]]) -> Path:
    path = output / name
    table.to_csv(path, index=False)
    files.append({"path": name, "rows": int(len(table)), "sha256": sha256(path)})
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_tests(path: Path, key: str = "tests") -> pd.DataFrame:
    value = load_json(path)[key]
    if isinstance(value, dict):
        value = [{"name": name, **row} for name, row in value.items()]
    return pd.json_normalize(value, sep="__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    extension = args.extension_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []

    specs = contrast_specs(root / "configs/datasets")
    human_path = root / "results/aggregate/human.parquet"
    available = set(pq.ParquetFile(human_path).schema.names)
    columns = {
        "dataset_id", "participant_id", "time_seconds", "access_index",
        *[str(item["condition_field"]) for item in specs],
    }
    human = pd.read_parquet(human_path, columns=sorted(columns & available))
    timecourse_rows = []
    participant_time_rows = []
    for spec in specs:
        difference = difference_timecourse(human, spec)
        difference.insert(0, "contrast_id", spec["contrast_id"])
        difference.insert(1, "dataset", spec["dataset"])
        participant_time_rows.append(difference)
        summary = difference.groupby("time_seconds")["difference"].agg(["mean", "sem", "count"]).reset_index()
        summary.insert(0, "contrast_id", spec["contrast_id"])
        summary.insert(1, "dataset", spec["dataset"])
        summary["lower"] = summary["mean"] - 1.96 * summary["sem"]
        summary["upper"] = summary["mean"] + 1.96 * summary["sem"]
        summary["time_ms"] = 1000 * summary["time_seconds"]
        timecourse_rows.append(summary)
    write_csv(pd.concat(timecourse_rows, ignore_index=True), output, "human-timecourses.csv", files)
    write_csv(pd.concat(participant_time_rows, ignore_index=True), output, "human-participant-timecourses.csv", files)

    participant_effects = []
    for spec in specs:
        table = pd.read_csv(root / f"results/statistics/{spec['contrast_id']}/participant-contrasts.csv")
        row = pd.DataFrame({"contrast_id": spec["contrast_id"], "dataset": spec["dataset"], "contrast": table["contrast"]})
        row["participant_index"] = np.arange(len(row))
        participant_effects.append(row)
    write_csv(pd.concat(participant_effects, ignore_index=True), output, "human-participant-effects.csv", files)

    theory_root = root / "results/theory-comparison"
    human_contrasts = pd.read_csv(theory_root / "human-four-metric-contrasts.csv")
    discovery_scale = (
        human_contrasts[human_contrasts["contrast_id"].eq("gabor-0")]
        .groupby("metric")["contrast"].std().reindex(METRICS).clip(lower=1e-4)
    )
    human_profile = human_contrasts.groupby(["contrast_id", "metric"], as_index=False)["contrast"].mean()
    human_profile["standardized_contrast"] = human_profile.apply(
        lambda row: row["contrast"] / discovery_scale.loc[row["metric"]], axis=1
    )
    write_csv(human_profile, output, "human-fingerprints.csv", files)

    prediction = pd.read_csv(root / "results/prediction/nested-cv/fold-results.csv")
    write_csv(prediction, output, "prediction-folds.csv", files)

    machine_root = root / "results/aggregate/machine"
    accuracy = pd.read_csv(machine_root / "accuracy-matching/accuracy-by-bin.csv")
    accuracy_long = []
    for architecture in ARCHITECTURES:
        row = accuracy[["difficulty_bin", "selected_common_bin", f"{architecture}_mean_accuracy"]].copy()
        row["architecture"] = architecture
        row.rename(columns={f"{architecture}_mean_accuracy": "mean_accuracy"}, inplace=True)
        accuracy_long.append(row)
    write_csv(pd.concat(accuracy_long, ignore_index=True), output, "machine-accuracy.csv", files)

    signatures = pd.read_parquet(machine_root / "jacobian-signatures.parquet", columns=["architecture", "seed", "step", "gain"])
    step_seed = signatures.groupby(["architecture", "seed", "step"], as_index=False)["gain"].mean()
    write_csv(step_seed, output, "machine-step-gain-seed.csv", files)
    write_csv(pd.read_csv(machine_root / "interventions.csv"), output, "machine-interventions.csv", files)

    machine_contrasts = pd.read_csv(theory_root / "machine-four-metric-contrasts.csv")
    machine_profile = machine_contrasts.groupby(["architecture", "metric"], as_index=False)["contrast"].mean()
    write_csv(machine_profile, output, "machine-fingerprints.csv", files)

    stages = pd.read_csv(extension / "aggregate/seed-stage-summary.csv")
    write_csv(stages, output, "seed-stage-summary.csv", files)

    geometry_profiles = machine_profile.rename(columns={"architecture": "profile", "contrast": "value"})
    geometry_profiles["profile_type"] = "machine"
    discovery = human_contrasts[human_contrasts["contrast_id"].eq("gabor-0")]
    discovery_mean = discovery.groupby("metric", as_index=False)["contrast"].mean()
    discovery_mean["value"] = discovery_mean.apply(
        lambda row: row["contrast"] / discovery_scale.loc[row["metric"]], axis=1
    )
    discovery_mean["profile"] = "human_discovery"
    discovery_mean["profile_type"] = "human"
    geometry_profiles = pd.concat(
        [geometry_profiles[["profile", "metric", "value", "profile_type"]],
         discovery_mean[["profile", "metric", "value", "profile_type"]]],
        ignore_index=True,
    )
    write_csv(geometry_profiles, output, "geometry-profiles.csv", files)

    theory = load_json(theory_root / "theory-comparison.json")
    equivalence_rows = []
    for metric in METRICS:
        row = theory["constrained_vs_unlimited"][metric]
        equivalence_rows.append({
            "metric": metric,
            "mean": row["mean_standardized_difference"],
            "lower": row["ci90"][0],
            "upper": row["ci90"][1],
        })
    write_csv(pd.DataFrame(equivalence_rows), output, "capacity-equivalence.csv", files)
    write_csv(pd.DataFrame(theory["leave_one_dataset_out"]), output, "lodo-generalization.csv", files)

    aggregate = extension / "aggregate"
    for source, target in (
        ("sham-comparison.csv", "sham-comparison.csv"),
        ("adaptation-cost.csv", "adaptation-cost.csv"),
        ("post-adaptation-interventions.csv", "post-adaptation-interventions.csv"),
    ):
        write_csv(pd.read_csv(aggregate / source), output, target, files)

    sensitivity = load_json(extension / "statistics/retention-sensitivity.json")
    complete_case = pd.DataFrame(
        sensitivity["scenarios"]["complete_case_seeds"]["tests"]["human_specific_gain"]
    )
    complete_case[["lower", "upper"]] = pd.DataFrame(
        complete_case["confidence_interval_95"].tolist(), index=complete_case.index
    )
    complete_case.drop(columns=["confidence_interval_95"], inplace=True)
    write_csv(complete_case, output, "retention-complete-case.csv", files)

    statistics = extension / "statistics"
    write_csv(flatten_tests(statistics / "primary.json"), output, "primary-tests.csv", files)
    write_csv(flatten_tests(statistics / "sham-tests.json"), output, "sham-tests.csv", files)
    write_csv(flatten_tests(statistics / "transfer-tests.json"), output, "transfer-tests.csv", files)
    write_csv(flatten_tests(statistics / "intervention-tests.json"), output, "intervention-tests.csv", files)
    architecture = pd.json_normalize(
        load_json(statistics / "architecture-comparisons.json")["all_prespecified_endpoints"], sep="__"
    )
    write_csv(architecture, output, "architecture-endpoints.csv", files)

    source_manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "renderer-neutral frozen tables for native R figures",
        "project_root": str(root),
        "extension_root": str(extension),
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ready": True, "tables": len(files), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
