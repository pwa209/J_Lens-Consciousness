"""Publication-grade, deterministic figures for the final article phase."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap


ARCHITECTURES = (
    "feedforward",
    "private_modules",
    "recurrent",
    "shared_workspace",
    "unlimited_shared_state",
)
ARCH_LABELS = {
    "feedforward": "Feedforward",
    "private_modules": "Private modules",
    "recurrent": "Recurrent",
    "shared_workspace": "Capacity-limited\nshared workspace",
    "unlimited_shared_state": "Unlimited\nshared state",
}
ARCH_COLORS = {
    "feedforward": "#184D77",
    "private_modules": "#497987",
    "recurrent": "#858D7E",
    "shared_workspace": "#D48448",
    "unlimited_shared_state": "#BE4A36",
}
DATASET_COLORS = {"gabor": "#184D77", "kronemer": "#D48448", "somato": "#858D7E"}
STUDY_DIVERGING = LinearSegmentedColormap.from_list(
    "study_diverging", ["#184D77", "#E1BD89", "#F7F5F0", "#D48448", "#BE4A36"]
)
METRICS = ("gain", "broadcast", "persistence", "concentration")
METRIC_LABELS = {
    "gain": "Gain",
    "broadcast": "Broadcast",
    "persistence": "Persistence",
    "concentration": "Concentration",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=10)


def _save(figure: plt.Figure, stem: Path, title: str, files: list[dict[str, str]]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    files.extend(
        [
            {"path": str(png), "format": "png", "title": title},
            {"path": str(pdf), "format": "pdf", "title": title},
        ]
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contrast_specs(config_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in ("gabor", "kronemer", "somato"):
        config = yaml.safe_load((config_root / f"{dataset}.yaml").read_text(encoding="utf-8"))
        for index, contrast in enumerate(config["primary_contrasts"]):
            rows.append({"id": f"{dataset}-{index}", "dataset": dataset, **contrast})
    return rows


def _level(values: pd.Series, target: object) -> pd.Series:
    normalized = values.astype(str).str.replace(r"\.0$", "", regex=True)
    wanted = str(target).removesuffix(".0")
    return normalized == wanted


def _human_columns(human_path: Path, specs: list[dict[str, Any]]) -> list[str]:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(human_path).schema.names)
    wanted = {
        "dataset_id", "participant_id", "original_trial_id", "time_seconds",
        "access_index", "gain", "broadcast", "persistence", "concentration",
        "effective_rank", *[str(value["condition_field"]) for value in specs],
    }
    missing = {"dataset_id", "participant_id", "time_seconds", "access_index"} - available
    if missing:
        raise ValueError(f"human aggregate lacks required columns: {sorted(missing)}")
    return sorted(wanted & available)


def figure_1(output: Path, files: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_axis_off()
    boxes = [
        (0.03, 0.70, 0.25, 0.20, "Human arm\n3 EEG datasets\n173 QC-retained participants", "#DCE8F5"),
        (0.03, 0.16, 0.25, 0.38, "Machine arm\n\nFeedforward\nPrivate modules\nRecurrent\nCapacity-limited workspace\nUnlimited shared state", "#F5E4D8"),
        (0.38, 0.61, 0.24, 0.27, "Independent state trajectories\n\nCross-fitted human latents\n20 seeds per architecture", "#E8E8E8"),
        (0.38, 0.18, 0.24, 0.27, "Shared geometry\n\nGain • Broadcast\nPersistence • Concentration", "#E1EFE4"),
        (0.72, 0.61, 0.25, 0.27, "Discovery and replication\n\nGabor discovery\nKronemer + Somato tests", "#E7E2F2"),
        (0.72, 0.18, 0.25, 0.27, "Falsifiable theory ranking\n\nRMS distance • Cosine\nCapacity equivalence", "#F2DFE2"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", linewidth=0.8)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.35)
    arrows = [((0.28, 0.79), (0.38, 0.75)), ((0.28, 0.35), (0.38, 0.69)),
              ((0.50, 0.61), (0.50, 0.45)), ((0.62, 0.31), (0.72, 0.31)),
              ((0.62, 0.74), (0.72, 0.74)), ((0.84, 0.61), (0.84, 0.45))]
    for start, stop in arrows:
        ax.annotate("", xy=stop, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.5, 0.98, "Independent brains and machines, compared in a common geometry",
            ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.03,
            "Human data never train the machine models; similarity is evaluated only after both arms are complete.",
            ha="center", va="bottom", fontsize=8)
    _save(fig, output / "Fig1-study-design", "Study design and falsifiable theory map", files)


def figure_2(machine_dir: Path, theory_dir: Path, output: Path, files: list[dict[str, str]]) -> None:
    accuracy = pd.read_csv(machine_dir / "accuracy-matching/accuracy-by-bin.csv")
    interventions = pd.read_csv(machine_dir / "interventions.csv")
    signatures = pd.read_parquet(
        machine_dir / "jacobian-signatures.parquet",
        columns=["architecture", "seed", "step", "gain"],
    )
    machine = pd.read_csv(theory_dir / "machine-four-metric-contrasts.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    ax = axes[0, 0]
    for architecture in ARCHITECTURES:
        column = f"{architecture}_mean_accuracy"
        ax.plot(accuracy["difficulty_bin"], accuracy[column], marker="o", ms=2.5,
                lw=1.2, color=ARCH_COLORS[architecture], label=ARCH_LABELS[architecture].replace("\n", " "))
    selected = accuracy[accuracy["selected_common_bin"].astype(bool)]
    if not selected.empty:
        ax.axvline(float(selected.iloc[0]["difficulty_bin"]), color="black", ls="--", lw=0.9)
    ax.set(xlabel="Difficulty bin", ylabel="Presence accuracy", title="Accuracy matching")
    ax.legend(frameon=False, ncol=2, loc="best")
    _panel(ax, "a")

    ax = axes[0, 1]
    positions = np.arange(len(ARCHITECTURES))
    rng = np.random.default_rng(20260730)
    for index, architecture in enumerate(ARCHITECTURES):
        group = interventions[interventions["architecture"] == architecture]
        for offset, column, color, label in (
            (-0.16, "top_subspace_accuracy_drop", ARCH_COLORS[architecture], "Targeted"),
            (0.16, "random_drop_mean", "#A8A8A8", "Random"),
        ):
            values = group[column].to_numpy(float)
            jitter = rng.uniform(-0.045, 0.045, len(values))
            ax.scatter(index + offset + jitter, values, s=8, alpha=0.55, color=color, edgecolor="none")
            ax.errorbar(index + offset, values.mean(), yerr=1.96 * values.std(ddof=1) / np.sqrt(len(values)),
                        fmt="o", ms=4, color="black", capsize=2)
    ax.set_xticks(positions, [ARCH_LABELS[a].replace("\n", " ") for a in ARCHITECTURES], rotation=25, ha="right")
    ax.set(ylabel="Accuracy drop", title="Causal subspace intervention")
    ax.text(0.02, 0.98, "colored: targeted   gray: random", transform=ax.transAxes, va="top", fontsize=7)
    _panel(ax, "b")

    ax = axes[1, 0]
    step_gain = signatures.groupby(["architecture", "seed", "step"], as_index=False)["gain"].mean()
    for architecture in ARCHITECTURES:
        group = step_gain[step_gain["architecture"] == architecture]
        summary = group.groupby("step")["gain"].agg(["mean", "sem"])
        ax.plot(summary.index, summary["mean"], marker="o", ms=3, color=ARCH_COLORS[architecture])
        ax.fill_between(summary.index, summary["mean"] - 1.96 * summary["sem"],
                        summary["mean"] + 1.96 * summary["sem"], color=ARCH_COLORS[architecture], alpha=0.12)
    ax.set(xlabel="Processing step", ylabel="Mean Jacobian gain", title="Step-resolved machine dynamics")
    _panel(ax, "c")

    ax = axes[1, 1]
    profile = machine.groupby(["architecture", "metric"])["contrast"].mean().unstack().reindex(ARCHITECTURES)
    image = ax.imshow(profile[list(METRICS)].to_numpy(), cmap=STUDY_DIVERGING, aspect="auto")
    ax.set_xticks(range(len(METRICS)), [METRIC_LABELS[m] for m in METRICS], rotation=25, ha="right")
    ax.set_yticks(range(len(ARCHITECTURES)), [ARCH_LABELS[a].replace("\n", " ") for a in ARCHITECTURES])
    ax.set_title("Accuracy-matched geometry contrast")
    fig.colorbar(image, ax=ax, shrink=0.7, label="Correct − incorrect (z)")
    _panel(ax, "d")
    _save(fig, output / "Fig2-machine-experiment", "Controlled machine experiment", files)


def _difference_timecourse(table: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    selected = table[table["dataset_id"] == spec["dataset"]].copy()
    field = str(spec["condition_field"])
    selected["_positive"] = _level(selected[field], spec["positive"])
    selected["_negative"] = _level(selected[field], spec["negative"])
    selected = selected[selected["_positive"] | selected["_negative"]]
    selected["_condition"] = np.where(selected["_positive"], "positive", "negative")
    grouped = selected.groupby(["participant_id", "time_seconds", "_condition"])["access_index"].mean().unstack()
    difference = (grouped["positive"] - grouped["negative"]).rename("difference").reset_index()
    return difference


def figure_3(human: pd.DataFrame, specs: list[dict[str, Any]], statistics_root: Path,
             output: Path, files: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0), constrained_layout=True)
    labels = "abcde"
    for axis, spec, label in zip(axes.flat, specs, labels, strict=False):
        difference = _difference_timecourse(human, spec)
        participant = difference.pivot(index="participant_id", columns="time_seconds", values="difference")
        mean, sem = participant.mean(), participant.sem()
        times = mean.index.to_numpy(float) * 1000
        color = DATASET_COLORS[spec["dataset"]]
        axis.plot(times, mean, color=color, lw=1.4)
        axis.fill_between(times, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.20)
        low, high = spec["window_ms"]
        axis.axvspan(low, high, color="#777777", alpha=0.10)
        axis.axhline(0, color="black", lw=0.6)
        axis.axvline(0, color="black", lw=0.6)
        frequentist = _load_json(statistics_root / spec["id"] / "frequentist.json")
        ymin, ymax = axis.get_ylim()
        for cluster in frequentist.get("clusters", []):
            if float(cluster["familywise_p"]) < 0.05:
                axis.plot([1000 * cluster["start_seconds"], 1000 * cluster["stop_seconds"]],
                          [ymin + 0.04 * (ymax - ymin)] * 2, color="black", lw=3)
        contrast = f"{spec['positive']} − {spec['negative']}"
        axis.set(title=f"{spec['dataset'].capitalize()}: {contrast}", xlabel="Time (ms)", ylabel="Δ Access Index")
        _panel(axis, label)
    summary_axis = axes.flat[-1]
    rows = []
    for spec in specs:
        values = pd.read_csv(statistics_root / spec["id"] / "participant-contrasts.csv")["contrast"].dropna()
        rows.append((spec["id"], values.mean(), 1.96 * values.sem(), len(values)))
    y = np.arange(len(rows))
    summary_axis.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows], fmt="o", color="#333333", capsize=2)
    summary_axis.axvline(0, color="black", lw=0.7)
    summary_axis.set_yticks(y, [f"{r[0]} (n={r[3]})" for r in rows])
    summary_axis.set(xlabel="Primary-window contrast (95% CI)", title="Participant-level effects")
    _panel(summary_axis, "f")
    _save(fig, output / "Fig3-human-access-geometry", "Human geometry of access", files)


def figure_4(theory_dir: Path, output: Path, files: list[dict[str, str]]) -> None:
    human = pd.read_csv(theory_dir / "human-four-metric-contrasts.csv")
    theory = _load_json(theory_dir / "theory-comparison.json")
    discovery = human[human["contrast_id"] == "gabor-0"]
    scale = discovery.groupby("metric")["contrast"].std().reindex(METRICS).clip(lower=1e-4)
    means = human.groupby(["contrast_id", "metric"])["contrast"].mean().unstack().reindex(columns=METRICS)
    standardized = means.divide(scale, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    axes = axes.flat
    ax = axes[0]
    limit = float(np.nanmax(np.abs(standardized.to_numpy()))) or 1.0
    image = ax.imshow(
        standardized.to_numpy(), cmap=STUDY_DIVERGING, vmin=-limit, vmax=limit, aspect="auto"
    )
    ax.set_xticks(range(len(METRICS)), [METRIC_LABELS[m] for m in METRICS], rotation=30, ha="right")
    ax.set_yticks(range(len(standardized)), standardized.index)
    ax.set_title("Human contrast signatures")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Contrast / discovery SD")
    _panel(ax, "a")

    replication = pd.DataFrame(theory["replication"]).T
    ax = axes[1]
    ax.scatter(replication["standardized_distance_from_gabor_discovery"],
               replication["same_direction_fraction"], s=45,
               c=[DATASET_COLORS[str(index).split("-")[0]] for index in replication.index])
    for index, row in replication.iterrows():
        ax.annotate(index, (row["standardized_distance_from_gabor_discovery"], row["same_direction_fraction"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Distance from Gabor discovery", ylabel="Same-direction fraction", ylim=(-0.05, 1.05),
           title="Replication and specificity")
    _panel(ax, "b")

    ax = axes[2]
    counts = human.groupby("contrast_id")["participant_id"].nunique().reindex(replication.index)
    ax.barh(np.arange(len(counts)), counts, color=[DATASET_COLORS[str(i).split("-")[0]] for i in counts.index])
    ax.set_yticks(np.arange(len(counts)), counts.index)
    ax.invert_yaxis()
    ax.set(xlabel="Participants", title="Independent evidence by contrast")
    _panel(ax, "c")
    ax = axes[3]
    lodo = pd.DataFrame(theory.get("leave_one_dataset_out", []))
    if lodo.empty:
        ax.text(0.5, 0.5, "Leave-one-dataset-out result unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(lodo))
        ax.bar(x, lodo["standardized_distance"],
               color=[DATASET_COLORS[d] for d in lodo["held_out_dataset"]])
        ax.set_xticks(x, lodo["contrast_id"], rotation=30, ha="right")
        ax.set(ylabel="Held-out standardized distance", title="Rotating leave-one-dataset-out test")
    _panel(ax, "d")
    _save(fig, output / "Fig4-cross-dataset-replication", "Cross-dataset replication", files)


def figure_5(theory_dir: Path, output: Path, files: list[dict[str, str]]) -> None:
    distances = pd.read_csv(theory_dir / "architecture-seed-distances.csv")
    machine = pd.read_csv(theory_dir / "machine-four-metric-contrasts.csv")
    human = pd.read_csv(theory_dir / "human-four-metric-contrasts.csv")
    theory = _load_json(theory_dir / "theory-comparison.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), constrained_layout=True)
    ax = axes[0]
    rng = np.random.default_rng(20260730)
    for index, architecture in enumerate(ARCHITECTURES):
        values = distances.loc[distances["architecture"] == architecture, "rms_distance"].to_numpy(float)
        ax.scatter(index + rng.uniform(-0.12, 0.12, len(values)), values, s=10, alpha=0.55,
                   color=ARCH_COLORS[architecture], edgecolor="none")
        ax.errorbar(index, values.mean(), yerr=1.96 * values.std(ddof=1) / np.sqrt(len(values)),
                    fmt="o", color="black", capsize=2)
    ax.set_xticks(range(len(ARCHITECTURES)), [ARCH_LABELS[a].replace("\n", " ") for a in ARCHITECTURES],
                  rotation=30, ha="right")
    ax.set(ylabel="Human–machine RMS distance", title="Competing architecture ranking")
    _panel(ax, "a")

    ax = axes[1]
    human_center = human[human["contrast_id"] == "gabor-0"].groupby("metric")["contrast"].mean().reindex(METRICS)
    x = np.arange(len(METRICS))
    ax.plot(x, human_center, color="black", lw=2.2, marker="o", label="Human discovery")
    for architecture in ARCHITECTURES:
        values = machine[machine["architecture"] == architecture].groupby("metric")["contrast"].mean().reindex(METRICS)
        ax.plot(x, values, color=ARCH_COLORS[architecture], lw=1.1, marker="o", ms=3,
                label=ARCH_LABELS[architecture].replace("\n", " "))
    ax.axhline(0, color="#777777", lw=0.6)
    ax.set_xticks(x, [METRIC_LABELS[m] for m in METRICS], rotation=25, ha="right")
    ax.set(ylabel="Condition contrast", title="Four-metric theory profiles")
    ax.legend(frameon=False, fontsize=6)
    _panel(ax, "b")

    ax = axes[2]
    equivalence = theory["constrained_vs_unlimited"]
    means = [equivalence[m]["mean_standardized_difference"] for m in METRICS]
    lows = [equivalence[m]["ci90"][0] for m in METRICS]
    highs = [equivalence[m]["ci90"][1] for m in METRICS]
    y = np.arange(len(METRICS))
    ax.axvspan(-0.20, 0.20, color="#E1BD89", alpha=0.35)
    ax.errorbar(means, y, xerr=[np.asarray(means) - lows, np.asarray(highs) - means],
                fmt="o", color="#333333", capsize=2)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, [METRIC_LABELS[m] for m in METRICS])
    ax.invert_yaxis()
    status = "Falsified" if theory["capacity_limit_interpretation_falsified"] else "Not falsified"
    ax.set(xlabel="Constrained − unlimited (discovery SD)", title=f"Capacity-limit test: {status}")
    _panel(ax, "c")
    _save(fig, output / "Fig5-human-machine-theory", "Human–machine theory competition", files)


def figure_6(prediction_dir: Path, statistics_root: Path, qc_path: Path,
             specs: list[dict[str, Any]], output: Path, files: list[dict[str, str]]) -> None:
    folds = pd.read_csv(prediction_dir / "fold-results.csv")
    paired = folds.pivot(index="fold", columns="family", values="auc")
    summary = _load_json(prediction_dir / "summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), constrained_layout=True)
    ax = axes[0]
    for _, row in paired.iterrows():
        ax.plot([0, 1], [row["conventional"], row["conventional_plus_jacobian"]],
                color="#999999", lw=0.9, marker="o", ms=4)
    ax.set_xticks([0, 1], ["Conventional", "+ Jacobian"])
    ax.set(ylabel="Held-out AUC", title=f"Nested CV: ΔAUC={summary['mean_incremental_auc']:.3f}")
    _panel(ax, "a")

    ax = axes[1]
    rows = []
    for spec in specs:
        values = pd.read_csv(statistics_root / spec["id"] / "participant-contrasts.csv")["contrast"].dropna()
        frequentist = _load_json(statistics_root / spec["id"] / "frequentist.json")
        bayes = _load_json(statistics_root / spec["id"] / "bayes-factor.json")
        rows.append((spec["id"], values.mean(), 1.96 * values.sem(),
                     frequentist["directional_sign_flip_p"], bayes["directional_bayes_factor"]))
    y = np.arange(len(rows))
    ax.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows], fmt="o", color="#333333", capsize=2)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set(xlabel="Access Index contrast (95% CI)", title="Primary evidence")
    for index, row in enumerate(rows):
        ax.text(ax.get_xlim()[1], index, f"p={row[3]:.3g}; BF={row[4]:.2g}", ha="right", va="bottom", fontsize=6)
    _panel(ax, "b")

    ax = axes[2]
    qc = _load_json(qc_path)
    decisions = pd.DataFrame(qc["decisions"])
    counts = decisions.groupby(["dataset_id", "included"]).size().unstack(fill_value=0)
    counts = counts.reindex(["gabor", "kronemer", "somato"])
    included = counts.get(True, pd.Series(0, index=counts.index))
    excluded = counts.get(False, pd.Series(0, index=counts.index))
    x = np.arange(len(counts))
    ax.bar(x, included, color=[DATASET_COLORS[d] for d in counts.index], label="Included")
    ax.bar(x, excluded, bottom=included, color="#C8C8C8", label="Excluded")
    ax.set_xticks(x, [d.capitalize() for d in counts.index])
    ax.set(ylabel="Participants", title="Frozen analysis sample")
    ax.legend(frameon=False)
    _panel(ax, "c")
    _save(fig, output / "Fig6-prediction-and-evidence", "Incremental prediction and evidence", files)


def supplements(human: pd.DataFrame, theory_dir: Path, qc_path: Path,
                output: Path, files: list[dict[str, str]]) -> None:
    qc = _load_json(qc_path)
    records = []
    for item in qc["decisions"]:
        values = item.get("qc") or {}
        records.append({"dataset_id": item["dataset_id"], "included": item["included"],
                        "bad_channel_fraction": values.get("bad_channel_fraction"),
                        "valid_trials": values.get("valid_trials"),
                        "rejected_trials": values.get("rejected_trials")})
    table = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    for axis, column, title in zip(axes, ("bad_channel_fraction", "valid_trials", "rejected_trials"),
                                   ("Bad-channel fraction", "Valid trials", "Rejected trials"), strict=True):
        for dataset in ("gabor", "kronemer", "somato"):
            values = table.loc[table["dataset_id"] == dataset, column].dropna()
            axis.hist(values, bins=20, histtype="step", lw=1.3, color=DATASET_COLORS[dataset], label=dataset)
        axis.set(title=title, ylabel="Participants")
    axes[0].legend(frameon=False)
    _save(fig, output / "FigS1-preprocessing-qc", "Outcome-blind preprocessing QC", files)

    metrics = [name for name in ("access_index", "gain", "broadcast", "persistence", "concentration", "effective_rank") if name in human]
    sample = human[metrics].replace([np.inf, -np.inf], np.nan)
    if len(sample) > 500_000:
        sample = sample.sample(500_000, random_state=20260730)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=False):
        axis.hist(sample[metric].dropna(), bins=60, color="#858D7E")
        axis.set(title=metric.replace("_", " "), ylabel="Rows")
    for axis in axes.flat[len(metrics):]:
        axis.set_axis_off()
    _save(fig, output / "FigS2-metric-distributions", "Geometry metric distributions", files)

    distances = pd.read_csv(theory_dir / "architecture-seed-distances.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for architecture in ARCHITECTURES:
        group = distances[distances["architecture"] == architecture]
        axes[0].scatter(group["rms_distance"], group["cosine_similarity"], s=12,
                        alpha=0.65, color=ARCH_COLORS[architecture], label=ARCH_LABELS[architecture].replace("\n", " "))
        axes[1].scatter(group["rms_distance"], group["magnitude_ratio"], s=12,
                        alpha=0.65, color=ARCH_COLORS[architecture])
    axes[0].set(xlabel="RMS distance", ylabel="Cosine similarity", title="Shape and distance")
    axes[1].set(xlabel="RMS distance", ylabel="Magnitude ratio", title="Magnitude sensitivity")
    axes[0].legend(frameon=False, fontsize=6)
    _save(fig, output / "FigS3-seed-sensitivity", "Seed-level theory sensitivity", files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--machine-dir", type=Path, required=True)
    parser.add_argument("--theory-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--statistics-root", type=Path, required=True)
    parser.add_argument("--qc-roster", type=Path, required=True)
    parser.add_argument("--dataset-config-root", type=Path, default=Path("configs/datasets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    _style()
    specs = _contrast_specs(args.dataset_config_root)
    human = pd.read_parquet(args.human, columns=_human_columns(args.human, specs))
    files: list[dict[str, str]] = []
    figure_1(args.output, files)
    figure_2(args.machine_dir, args.theory_dir, args.output, files)
    figure_3(human, specs, args.statistics_root, args.output, files)
    figure_4(args.theory_dir, args.output, files)
    figure_5(args.theory_dir, args.output, files)
    figure_6(args.prediction_dir, args.statistics_root, args.qc_roster, specs, args.output, files)
    supplements(human, args.theory_dir, args.qc_roster, args.output, files)
    for item in files:
        item["sha256"] = _sha256(Path(item["path"]))
    manifest = {
        "ready": True,
        "design": "science_advances_six_main_three_supplementary",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "main_figures": 6,
        "supplementary_figures": 3,
        "files": files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ready": True, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
