"""Audit pilot source formats needed to seal dataset-adapter mappings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _six_regions(names: list[str], positions: np.ndarray) -> dict[str, list[int]]:
    """Split scalp positions into left/right by anterior/central/posterior thirds."""

    positions = np.asarray(positions, dtype=float)
    valid = np.isfinite(positions[:, :2]).all(axis=1) & (
        np.linalg.norm(positions, axis=1) > 0
    )
    posterior_cut, anterior_cut = np.quantile(positions[valid, 1], [1 / 3, 2 / 3])
    groups: dict[str, list[int]] = {}
    for side, side_mask in (("left", positions[:, 0] < 0), ("right", positions[:, 0] >= 0)):
        divisions = {
            "anterior": positions[:, 1] > anterior_cut,
            "central": (positions[:, 1] >= posterior_cut)
            & (positions[:, 1] <= anterior_cut),
            "posterior": positions[:, 1] < posterior_cut,
        }
        for region, region_mask in divisions.items():
            groups[f"{side}_{region}"] = np.flatnonzero(
                valid & side_mask & region_mask
            ).astype(int).tolist()
    if any(not indices for indices in groups.values()):
        raise RuntimeError(f"empty scalp region for channels {names}")
    return groups


def inspect_gabor(root: Path) -> dict[str, Any]:
    import mne

    events = pd.read_csv(next(root.rglob("*_events.tsv")), sep="\t", encoding="utf-8-sig")
    raw = mne.io.read_raw_brainvision(next(root.rglob("*.vhdr")), preload=False, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    electrodes = pd.read_csv(next(root.rglob("*_electrodes.tsv")), sep="\t", encoding="utf-8-sig")
    electrode_positions = electrodes.set_index("name")[["x", "y", "z"]]
    positions = np.asarray(
        [electrode_positions.loc[raw.ch_names[index]].to_numpy(dtype=float) for index in picks]
    )
    return {
        "columns": events.columns.tolist(),
        "event_counts": {
            str(key): int(value)
            for key, value in events["trial_type"].value_counts().sort_index().items()
        },
        "eeg_channels": [raw.ch_names[index] for index in picks],
        "output_channel_groups": _six_regions(
            [raw.ch_names[index] for index in picks], positions
        ),
    }


def inspect_kronemer(root: Path) -> dict[str, Any]:
    import mne

    recordings = []
    for path in sorted(root.rglob("*.raw")):
        raw = mne.io.read_raw_egi(path, preload=False, verbose="ERROR")
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        montage_positions = mne.channels.make_standard_montage(
            "GSN-HydroCel-256"
        ).get_positions()["ch_pos"]
        positions = np.asarray(
            [
                montage_positions.get(
                    raw.ch_names[index], np.full(3, np.nan, dtype=float)
                )
                for index in picks
            ]
        )
        annotations = Counter(str(value) for value in raw.annotations.description)
        recordings.append(
            {
                "path": str(path.relative_to(root)),
                "sampling_rate_hz": float(raw.info["sfreq"]),
                "n_times": int(raw.n_times),
                "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
                "channel_count": len(raw.ch_names),
                "channel_names_head": raw.ch_names[:12],
                "eeg_channels": [raw.ch_names[index] for index in picks],
                "output_channel_groups": _six_regions(
                    [raw.ch_names[index] for index in picks], positions
                ),
                "annotation_counts": dict(sorted(annotations.items())),
                "annotations_head": [
                    {"onset": float(onset), "description": str(description)}
                    for onset, description in zip(
                        raw.annotations.onset[:20],
                        raw.annotations.description[:20],
                        strict=True,
                    )
                ],
            }
        )
    behavior = []
    for path in sorted(root.rglob("*.csv")):
        table = pd.read_csv(path, skip_blank_lines=True)
        table = table.dropna(how="all")
        summary: dict[str, object] = {
            "path": str(path.relative_to(root)),
            "rows": len(table),
            "columns": table.columns.tolist(),
        }
        for name in (
            "TRIAL TYPE",
            "QUESTION TYPE",
            "Face shown",
            "Center Face shown",
            "Quadrant Face shown",
            "Perception answer",
            "Task Relevant",
            "Task Paradigm",
        ):
            if name in table:
                summary[f"values:{name}"] = [
                    _json_value(value)
                    for value in table[name].dropna().value_counts().sort_index().index.tolist()
                ]
                summary[f"counts:{name}"] = {
                    str(key): int(value)
                    for key, value in table[name].dropna().value_counts().items()
                }
        for name in (
            "FaceDrawStart",
            "Center Face time",
            "Quadrant Face time",
            "Trial start time",
        ):
            if name in table:
                summary[f"head:{name}"] = [
                    float(value) for value in table[name].dropna().head(10)
                ]
        behavior.append(summary)
    return {"recordings": recordings, "behavior": behavior}


def inspect_somato(root: Path) -> dict[str, Any]:
    import scipy.io

    participants = sorted(path.name for path in root.rglob("sub*") if path.is_dir())
    examples = []
    for path in sorted(root.rglob("EEG_data.mat"))[:8]:
        loaded = scipy.io.loadmat(path, simplify_cells=True)
        arrays = {
            key: list(np.asarray(value).shape)
            for key, value in loaded.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }
        examples.append({"path": str(path.relative_to(root)), "arrays": arrays})
    location_path = next(root.rglob("EEG_locations.mat"))
    locations = scipy.io.loadmat(location_path, simplify_cells=True)
    chanlocs = locations["chanlocs"]
    somato_names = [str(channel["labels"]) for channel in chanlocs]
    # EEGLAB coordinates use +X toward the nose and +Y toward the left ear.
    somato_positions = np.asarray(
        [[-float(channel["Y"]), float(channel["X"]), float(channel["Z"])] for channel in chanlocs]
    )
    location_summary = {
        key: {
            "type": type(value).__name__,
            "shape": list(np.asarray(value).shape),
            "preview": str(value)[:500],
        }
        for key, value in locations.items()
        if not key.startswith("__")
    }
    return {
        "participants": participants,
        "example_arrays": examples,
        "location_file": str(location_path.relative_to(root)),
        "location_metadata": location_summary,
        "eeg_channels": somato_names,
        "output_channel_groups": _six_regions(somato_names, somato_positions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/source-inspection/pilot-adapter-audit.json"),
    )
    args = parser.parse_args()
    report = {
        "gabor": inspect_gabor(args.raw_root / "gabor" / "sub-10"),
        "kronemer_report": inspect_kronemer(args.raw_root / "kronemer" / "223_RP_EEG"),
        "kronemer_no_report": inspect_kronemer(args.raw_root / "kronemer" / "238_NRP"),
        "somato": inspect_somato(args.raw_root / "somato"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
