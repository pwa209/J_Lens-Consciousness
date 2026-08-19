"""Reproducible main and supplementary figures from aggregate tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(figure: object, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def human_timecourse(table: pd.DataFrame, output: Path) -> None:
    datasets = sorted(table["dataset_id"].unique())
    figure, axes = plt.subplots(len(datasets), 1, figsize=(7, 2.7 * len(datasets)), squeeze=False)
    for axis, dataset in zip(axes[:, 0], datasets, strict=True):
        selected = table[table["dataset_id"] == dataset]
        participant = (
            selected.groupby(["participant_id", "time_seconds"])["access_index"].mean().unstack()
        )
        mean = participant.mean(axis=0)
        sem = participant.sem(axis=0)
        times = mean.index.to_numpy() * 1000
        axis.plot(times, mean, color="#184D77", linewidth=1.8)
        axis.fill_between(times, mean - sem, mean + sem, color="#184D77", alpha=0.2)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        axis.set(title=dataset, ylabel="Access index")
    axes[-1, 0].set_xlabel("Time from stimulus (ms)")
    figure.tight_layout()
    _save(figure, output / "main-human-timecourse")


def machine_summary(machine_directory: Path, output: Path) -> None:
    signature = pd.read_parquet(machine_directory / "jacobian-signatures.parquet")
    interventions = pd.read_csv(machine_directory / "architecture-summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    summary = signature.groupby(["architecture", "step"])["gain"].mean().unstack(0)
    summary.plot(ax=axes[0], marker="o")
    axes[0].set(xlabel="Processing step", ylabel="Jacobian gain", title="Machine signatures")
    x = np.arange(len(interventions))
    axes[1].bar(x - 0.18, interventions["mean_top_drop"], 0.36, label="top subspace")
    axes[1].bar(x + 0.18, interventions["mean_random_drop"], 0.36, label="random")
    axes[1].set_xticks(x, interventions["architecture"], rotation=25, ha="right")
    axes[1].set(ylabel="Accuracy drop", title="Causal intervention")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    _save(figure, output / "main-machine-comparison")


def diagnostics(human: pd.DataFrame, output: Path) -> None:
    metrics = ["gain", "broadcast", "persistence", "concentration", "effective_rank"]
    available = [name for name in metrics if name in human]
    figure, axes = plt.subplots(1, len(available), figsize=(3 * len(available), 2.8))
    axes = np.atleast_1d(axes)
    for axis, metric in zip(axes, available, strict=True):
        values = human[metric].replace([np.inf, -np.inf], np.nan).dropna()
        axis.hist(values, bins=50, color="#858D7E")
        axis.set(title=metric, ylabel="Rows")
    figure.tight_layout()
    _save(figure, output / "supplement-metric-distributions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    human = pd.read_parquet(args.human)
    human_timecourse(human, args.output)
    machine_summary(args.machine, args.output)
    diagnostics(human, args.output)


if __name__ == "__main__":
    main()
