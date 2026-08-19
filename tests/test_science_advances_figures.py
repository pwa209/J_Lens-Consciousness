from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = (
    "feedforward", "private_modules", "recurrent", "shared_workspace",
    "unlimited_shared_state",
)
METRICS = ("gain", "broadcast", "persistence", "concentration")


class ScienceAdvancesFigureTests(unittest.TestCase):
    def test_complete_figure_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine = root / "machine"
            theory = root / "theory"
            prediction = root / "prediction"
            statistics = root / "statistics"
            configs = root / "configs"
            output = root / "figures"
            for directory in (machine / "accuracy-matching", theory, prediction, statistics, configs):
                directory.mkdir(parents=True, exist_ok=True)

            specs = [
                ("gabor", "seen", 1, 0, [150, 300]),
                ("kronemer", "perceived", 1, 0, [175, 225]),
                ("kronemer", "perceived", 1, 0, [350, 650]),
                ("somato", "task_relevance", "tactile_relevant", "tactile_irrelevant", [100, 200]),
                ("somato", "report", 1, 0, [300, 600]),
            ]
            dataset_specs: dict[str, list[dict[str, object]]] = {name: [] for name in ("gabor", "kronemer", "somato")}
            for dataset, field, positive, negative, window in specs:
                dataset_specs[dataset].append(
                    {"condition_field": field, "positive": positive, "negative": negative, "window_ms": window}
                )
            for dataset, contrasts in dataset_specs.items():
                (configs / f"{dataset}.yaml").write_text(
                    yaml.safe_dump({"dataset_id": dataset, "primary_contrasts": contrasts}), encoding="utf-8"
                )

            rng = np.random.default_rng(7)
            human_rows = []
            for dataset in dataset_specs:
                for participant in range(4):
                    for trial in range(8):
                        for time in (-0.1, 0.0, 0.1, 0.2, 0.4, 0.6):
                            positive = trial % 2
                            human_rows.append(
                                {
                                    "dataset_id": dataset,
                                    "participant_id": f"{dataset}-{participant}",
                                    "original_trial_id": trial,
                                    "time_seconds": time,
                                    "seen": positive,
                                    "perceived": positive,
                                    "task_relevance": "tactile_relevant" if positive else "tactile_irrelevant",
                                    "report": positive,
                                    "access_index": 0.15 * positive + 0.05 * time + rng.normal(0, 0.03),
                                    "gain": np.exp(rng.normal()),
                                    "broadcast": rng.uniform(0.05, 0.95),
                                    "persistence": rng.uniform(0.05, 0.95),
                                    "concentration": rng.uniform(0.05, 0.95),
                                    "effective_rank": rng.uniform(1, 10),
                                }
                            )
            human_path = root / "human.parquet"
            pd.DataFrame(human_rows).to_parquet(human_path, index=False)

            contrast_ids = []
            dataset_index = {name: 0 for name in dataset_specs}
            human_contrasts = []
            for dataset, field, positive, negative, window in specs:
                contrast_id = f"{dataset}-{dataset_index[dataset]}"
                dataset_index[dataset] += 1
                contrast_ids.append(contrast_id)
                directory = statistics / contrast_id
                directory.mkdir()
                values = pd.DataFrame({"participant_id": [f"{dataset}-{i}" for i in range(4)],
                                       "contrast": [0.10, 0.12, 0.08, 0.11]})
                values.to_csv(directory / "participant-contrasts.csv", index=False)
                (directory / "frequentist.json").write_text(
                    json.dumps({"directional_sign_flip_p": 0.03, "clusters": [
                        {"start_seconds": window[0] / 1000, "stop_seconds": window[1] / 1000,
                         "mass": 4.0, "familywise_p": 0.04}
                    ]}), encoding="utf-8"
                )
                (directory / "bayes-factor.json").write_text(
                    json.dumps({"directional_bayes_factor": 12.0}), encoding="utf-8"
                )
                for participant in range(4):
                    for metric_index, metric in enumerate(METRICS):
                        human_contrasts.append(
                            {"dataset_id": dataset, "contrast_id": contrast_id,
                             "participant_id": f"{dataset}-{participant}", "metric": metric,
                             "contrast": 0.1 + 0.02 * metric_index + rng.normal(0, 0.01)}
                        )
            pd.DataFrame(human_contrasts).to_csv(theory / "human-four-metric-contrasts.csv", index=False)

            machine_contrasts, distances, interventions, signatures = [], [], [], []
            for architecture_index, architecture in enumerate(ARCHITECTURES):
                for seed in range(3):
                    distances.append(
                        {"architecture": architecture, "seed": seed,
                         "rms_distance": 0.4 + architecture_index * 0.1 + seed * 0.01,
                         "cosine_similarity": 0.8 - architecture_index * 0.04,
                         "magnitude_ratio": 0.9 + architecture_index * 0.03}
                    )
                    interventions.append(
                        {"architecture": architecture, "seed": seed,
                         "top_subspace_accuracy_drop": 0.2 + architecture_index * 0.01,
                         "random_drop_mean": 0.05}
                    )
                    for metric_index, metric in enumerate(METRICS):
                        machine_contrasts.append(
                            {"architecture": architecture, "seed": seed, "metric": metric,
                             "contrast": 0.08 + 0.02 * metric_index + 0.01 * architecture_index}
                        )
                    for step in (3, 4, 5):
                        for sample in range(5):
                            signatures.append(
                                {"architecture": architecture, "seed": seed, "step": step,
                                 "gain": 1 + 0.1 * step + 0.02 * architecture_index}
                            )
            pd.DataFrame(machine_contrasts).to_csv(theory / "machine-four-metric-contrasts.csv", index=False)
            pd.DataFrame(distances).to_csv(theory / "architecture-seed-distances.csv", index=False)
            pd.DataFrame(interventions).to_csv(machine / "interventions.csv", index=False)
            pd.DataFrame(signatures).to_parquet(machine / "jacobian-signatures.parquet", index=False)
            pd.DataFrame(interventions).groupby("architecture").mean(numeric_only=True).reset_index().to_csv(
                machine / "architecture-summary.csv", index=False
            )
            accuracy = pd.DataFrame({"difficulty_bin": range(4)})
            for architecture in ARCHITECTURES:
                accuracy[f"{architecture}_mean_accuracy"] = [0.9, 0.8, 0.7, 0.6]
            accuracy["selected_common_bin"] = [False, False, True, False]
            accuracy.to_csv(machine / "accuracy-matching/accuracy-by-bin.csv", index=False)

            replication = {
                cid: {"standardized_distance_from_gabor_discovery": 0.2 + 0.1 * i,
                      "same_direction_fraction": 1.0 - 0.1 * i, "participants": 4}
                for i, cid in enumerate(contrast_ids)
            }
            equivalence = {
                metric: {"mean_standardized_difference": 0.02, "ci90": [-0.08, 0.10],
                         "equivalent_within_0.20": True}
                for metric in METRICS
            }
            lodo = [
                {"held_out_dataset": cid.split("-")[0], "contrast_id": cid,
                 "standardized_distance": 0.3 + 0.05 * index, "cosine_similarity": 0.8,
                 "same_direction_fraction": 0.75}
                for index, cid in enumerate(contrast_ids)
            ]
            (theory / "theory-comparison.json").write_text(
                json.dumps({"replication": replication, "constrained_vs_unlimited": equivalence,
                            "capacity_limit_interpretation_falsified": True,
                            "leave_one_dataset_out": lodo}), encoding="utf-8"
            )

            fold_rows = []
            for fold in range(5):
                fold_rows.extend(
                    [{"fold": fold, "family": "conventional", "auc": 0.65 + fold * 0.01},
                     {"fold": fold, "family": "conventional_plus_jacobian", "auc": 0.70 + fold * 0.01}]
                )
            pd.DataFrame(fold_rows).to_csv(prediction / "fold-results.csv", index=False)
            (prediction / "summary.json").write_text(
                json.dumps({"mean_incremental_auc": 0.05}), encoding="utf-8"
            )
            qc_path = root / "qc.json"
            qc_path.write_text(
                json.dumps({"decisions": [
                    {"dataset_id": dataset, "participant_id": f"{dataset}-{index}", "included": index < 3,
                     "qc": {"bad_channel_fraction": 0.02 * index, "valid_trials": 100 - index,
                            "rejected_trials": index}}
                    for dataset in dataset_specs for index in range(4)
                ]}), encoding="utf-8"
            )

            manifest = output / "figure-manifest.json"
            subprocess.run(
                [sys.executable, "-m", "jacaccess.figures.science_advances",
                 "--human", str(human_path), "--machine-dir", str(machine),
                 "--theory-dir", str(theory), "--prediction-dir", str(prediction),
                 "--statistics-root", str(statistics), "--qc-roster", str(qc_path),
                 "--dataset-config-root", str(configs), "--output", str(output),
                 "--manifest", str(manifest)],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(payload["ready"])
            self.assertEqual(payload["main_figures"], 6)
            self.assertEqual(payload["supplementary_figures"], 3)
            self.assertEqual(len(payload["files"]), 18)
            for item in payload["files"]:
                self.assertTrue(Path(item["path"]).exists())
                self.assertEqual(len(item["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
