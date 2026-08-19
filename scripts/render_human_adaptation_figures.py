"""Publication figures for the human-neural adaptation extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ARCHITECTURE_LABELS = {
    "feedforward": "Feedforward",
    "private_modules": "Private modules",
    "recurrent": "Recurrent",
    "shared_workspace": "Constrained shared",
    "unlimited_shared_state": "Unlimited shared",
}
STAGE_LABELS = {
    "random_init": "Random init",
    "task_trained": "Task trained",
    "human_adapted": "Human adapted",
    "sham_adapted": "Sham adapted",
}
COLORS = {
    "feedforward": "#C22525",
    "private_modules": "#6F5E56",
    "recurrent": "#3F3A39",
    "shared_workspace": "#C3AB8C",
    "unlimited_shared_state": "#E1D6C7",
}
STUDY_HEATMAP = LinearSegmentedColormap.from_list(
    "study_heatmap", ["#3F3A39", "#6F5E56", "#E1D6C7", "#C3AB8C", "#C22525"]
)


def _save(fig: plt.Figure, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().to_numpy(float)
    mean = float(values.mean())
    ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, ci


def figure_7(root: Path, output: Path) -> None:
    stages = pd.read_csv(root / "aggregate/seed-stage-summary.csv")
    sham = pd.read_csv(root / "aggregate/sham-comparison.csv")
    sham_seed = sham.groupby(["architecture", "seed"], as_index=False)["human_specific_gain"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.axis("off")
    ax.text(
        0.02,
        0.86,
        "Random init  →  Task trained  →  Human adapted\n"
        "                                  ↘  Sham adapted",
        fontsize=13,
        family="monospace",
        va="top",
    )
    ax.text(
        0.02,
        0.45,
        "Train participants: low-level EEG temporal RSM only\n"
        "Validation participants: neural-loss checkpoint selection\n"
        "Held-out participants: four-metric geometry evaluation only",
        fontsize=10,
        va="top",
    )
    ax.text(0.02, 0.08, "Final geometry metrics are never optimization targets.", weight="bold")
    ordered_stages = ["random_init", "task_trained", "human_adapted", "sham_adapted"]
    for metric, ax, title in (
        ("rms_distance", axes[0, 1], "Held-out human RMS distance"),
        ("cosine_similarity", axes[1, 0], "Held-out human cosine similarity"),
    ):
        for architecture, group in stages.groupby("architecture"):
            means, cis = [], []
            for stage in ordered_stages:
                mean, ci = _mean_ci(group.loc[group["stage"].eq(stage), metric])
                means.append(mean)
                cis.append(ci)
            ax.errorbar(
                range(4), means, yerr=cis, marker="o", lw=1.8,
                label=ARCHITECTURE_LABELS[architecture], color=COLORS[architecture]
            )
        ax.set_xticks(range(4), [STAGE_LABELS[value] for value in ordered_stages], rotation=20)
        ax.set_title(title)
        ax.axhline(0, color="0.7", lw=0.8)
    ax = axes[1, 1]
    order = list(ARCHITECTURE_LABELS)
    for index, architecture in enumerate(order):
        values = sham_seed.loc[sham_seed["architecture"].eq(architecture), "human_specific_gain"]
        mean, ci = _mean_ci(values)
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else []
        ax.scatter(np.asarray(jitter) + index, values, s=15, alpha=0.45, color=COLORS[architecture])
        ax.errorbar(index, mean, yerr=ci, marker="D", color="black", capsize=3)
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xticks(range(len(order)), [ARCHITECTURE_LABELS[x] for x in order], rotation=25, ha="right")
    ax.set_ylabel("Sham distance − human-adapted distance")
    ax.set_title("Human-specific alignment gain")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, output, "Fig7-human-adaptation-trajectory")


def figure_8(root: Path, output: Path) -> None:
    stages = pd.read_csv(root / "aggregate/fold-stage-summary.csv")
    costs = pd.read_csv(root / "aggregate/adaptation-cost.csv")
    sham = pd.read_csv(root / "aggregate/sham-comparison.csv")
    interventions = pd.read_csv(root / "aggregate/post-adaptation-interventions.csv")
    h = stages[stages["stage"].eq("human_adapted")].merge(
        costs[costs["condition"].eq("human_adapted")],
        on=["architecture", "seed", "outer_fold"],
    )
    gain = sham.merge(
        costs[costs["condition"].eq("human_adapted")],
        on=["architecture", "seed", "outer_fold"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for architecture, group in h.groupby("architecture"):
        axes[0, 0].scatter(
            group["relative_l2_parameter_displacement"], group["rms_distance"],
            s=18, alpha=0.45, label=ARCHITECTURE_LABELS[architecture], color=COLORS[architecture]
        )
    axes[0, 0].set(xlabel="Relative L2 parameter displacement", ylabel="Final held-out RMS distance")
    axes[0, 0].set_title("Adaptation cost versus final distance")
    for architecture, group in gain.groupby("architecture"):
        axes[0, 1].scatter(
            group["relative_l2_parameter_displacement"], group["alignment_gain"],
            s=18, alpha=0.45, color=COLORS[architecture]
        )
    axes[0, 1].axhline(0, color="0.5", lw=0.8)
    axes[0, 1].set(xlabel="Relative L2 parameter displacement", ylabel="Task-trained − adapted distance")
    axes[0, 1].set_title("Alignment gain versus adaptation cost")
    task = interventions[interventions["stage"].eq("task_trained")]
    adapted = interventions[interventions["stage"].eq("human_adapted")]
    positions = np.arange(len(ARCHITECTURE_LABELS))
    for index, architecture in enumerate(ARCHITECTURE_LABELS):
        task_values = task.loc[task["architecture"].eq(architecture), "causal_specificity"]
        adapted_values = adapted.loc[adapted["architecture"].eq(architecture), "causal_specificity"]
        for offset, values, label in ((-0.14, task_values, "Task"), (0.14, adapted_values, "Human")):
            mean, ci = _mean_ci(values)
            axes[1, 0].errorbar(index + offset, mean, yerr=ci, marker="o", capsize=3,
                                color="0.2" if label == "Task" else COLORS[architecture])
    axes[1, 0].set_xticks(positions, [ARCHITECTURE_LABELS[x] for x in ARCHITECTURE_LABELS], rotation=25, ha="right")
    axes[1, 0].set_ylabel("Targeted drop − random drop")
    axes[1, 0].set_title("Causal specificity before and after adaptation")
    task_seed = task[["architecture", "seed", "causal_specificity"]].rename(
        columns={"causal_specificity": "task_specificity"}
    )
    adapted_seed = adapted.groupby(["architecture", "seed"], as_index=False)["causal_specificity"].mean()
    causal = task_seed.merge(adapted_seed, on=["architecture", "seed"])
    causal["causal_change"] = causal["causal_specificity"] - causal["task_specificity"]
    gain_seed = gain.groupby(["architecture", "seed"], as_index=False)["alignment_gain"].mean()
    causal = causal.merge(gain_seed, on=["architecture", "seed"])
    for architecture, group in causal.groupby("architecture"):
        axes[1, 1].scatter(group["alignment_gain"], group["causal_change"], s=22, alpha=0.65,
                           color=COLORS[architecture])
    axes[1, 1].axhline(0, color="0.5", lw=0.8)
    axes[1, 1].axvline(0, color="0.5", lw=0.8)
    axes[1, 1].set(xlabel="Held-out alignment gain", ylabel="Causal-specificity change")
    axes[1, 1].set_title("Alignment and causal change (descriptive)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    _save(fig, output, "Fig8-adaptation-cost-causality")


def supplementary(root: Path, output: Path) -> None:
    sham = pd.read_csv(root / "aggregate/sham-comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for index, architecture in enumerate(ARCHITECTURE_LABELS):
        group = sham[sham["architecture"].eq(architecture)]
        human = group.groupby("seed")["human_adapted"].mean()
        control = group.groupby("seed")["sham_adapted"].mean()
        for left, right in zip(control, human, strict=True):
            ax.plot([index - 0.12, index + 0.12], [left, right], color="0.75", lw=0.7)
        ax.scatter(np.full(len(control), index - 0.12), control, s=14, color="0.5")
        ax.scatter(np.full(len(human), index + 0.12), human, s=14, color=COLORS[architecture])
    ax.set_xticks(range(len(ARCHITECTURE_LABELS)), [ARCHITECTURE_LABELS[x] for x in ARCHITECTURE_LABELS], rotation=25, ha="right")
    ax.set_ylabel("Held-out RMS distance")
    ax.set_title("Matched sham (gray) versus human adaptation (color)")
    _save(fig, output, "FigS4-human-adaptation-sham")

    transfer = pd.read_csv(root / "aggregate/external-transfer.csv")
    adapted = transfer[transfer["stage"].eq("human_adapted")]
    summary = adapted.groupby(["architecture", "evaluation_contrast"], as_index=False)["rms_distance"].mean()
    matrix = summary.pivot(index="architecture", columns="evaluation_contrast", values="rms_distance").reindex(ARCHITECTURE_LABELS)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap=STUDY_HEATMAP)
    ax.set_yticks(range(len(matrix)), [ARCHITECTURE_LABELS[x] for x in matrix.index])
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    ax.set_title("External held-out distance after Gabor adaptation")
    fig.colorbar(image, ax=ax, label="RMS distance")
    _save(fig, output, "FigS5-human-adaptation-transfer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.experiment_root / "figures"
    figure_7(args.experiment_root, output)
    figure_8(args.experiment_root, output)
    supplementary(args.experiment_root, output)


if __name__ == "__main__":
    main()
