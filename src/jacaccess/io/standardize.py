"""Declarative source-to-common-contract adapters for the three repositories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from jacaccess.config import load_yaml
from jacaccess.io.events import validate_event_rows
from jacaccess.io.source_inspection import inspect_source_tree


class UnverifiedMappingError(RuntimeError):
    pass


def _require_verified(config: dict[str, Any]) -> None:
    if config.get("adapter_status") != "verified":
        raise UnverifiedMappingError(
            f"{config['dataset_id']} adapter_status is {config.get('adapter_status')!r}; "
            "inspect one participant, fill event_columns, and set status to verified"
        )
    if config.get("dataset_id") == "somato":
        return
    mapping = config.get("event_columns")
    required = {
        "original_trial_id",
        "onset_seconds",
        "event_type",
        *config.get("required_physical_fields", []),
        *config.get("required_condition_fields", []),
    }
    if (
        not isinstance(mapping, dict)
        or required - mapping.keys()
        or any(mapping.get(name) in (None, "") for name in required)
    ):
        raise UnverifiedMappingError("event_columns is incomplete")


def inspect_participant(
    dataset_id: str,
    participant_id: str,
    raw_root: Path,
    output: Path,
) -> None:
    candidate_roots = [
        path
        for path in raw_root.rglob("*")
        if path.is_dir() and participant_id.lower() in path.name.lower()
    ]
    root = min(candidate_roots, key=lambda path: len(path.parts)) if candidate_roots else raw_root
    report = inspect_source_tree(root)
    report.update({"dataset_id": dataset_id, "participant_id": participant_id})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _read_mne_signal(path: Path) -> object:
    import mne

    suffix = path.suffix.lower()
    if suffix == ".set":
        try:
            return mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
        except TypeError:
            return mne.read_epochs_eeglab(path, verbose="ERROR")
    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".vhdr": mne.io.read_raw_brainvision,
        ".cnt": mne.io.read_raw_cnt,
        ".fif": mne.io.read_raw_fif,
    }
    if suffix not in readers:
        raise ValueError(f"unsupported continuous signal format {suffix}")
    return readers[suffix](path, preload=False, verbose="ERROR")


def _largest_mat_array(path: Path) -> np.ndarray:
    try:
        import scipy.io

        loaded = scipy.io.loadmat(path, simplify_cells=True)
        arrays = [
            np.asarray(value)
            for key, value in loaded.items()
            if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 3
        ]
    except NotImplementedError:
        arrays = []
        import h5py

        with h5py.File(path, "r") as handle:
            arrays = [
                np.asarray(handle[key])
                for key in handle
                if hasattr(handle[key], "shape") and len(handle[key].shape) == 3
            ]
    if not arrays:
        raise ValueError(f"no three-dimensional EEG array found in {path}")
    return max(arrays, key=lambda value: value.size)


def _standardize_somato(
    participant_id: str,
    participant_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> None:
    arrays: list[np.ndarray] = []
    event_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    for mat_path in sorted(participant_root.rglob("*.mat")):
        try:
            raw = _largest_mat_array(mat_path)
        except ValueError:
            continue
        # Publication contract is channels x time x trials.
        epochs = np.transpose(raw, (2, 0, 1)).astype(np.float32)
        amplitude_unit = str(config.get("source_amplitude_unit", "volt")).lower()
        if amplitude_unit in {"microvolt", "microvolts", "uv"}:
            epochs *= np.float32(1e-6)
        elif amplitude_unit != "volt":
            raise ValueError(f"unsupported source_amplitude_unit {amplitude_unit!r}")
        folder_text = mat_path.parent.as_posix().upper()
        report = int("/R/" in folder_text or "REPORT" in folder_text)
        instruction = "tactile_relevant" if "_CT" in folder_text else "tactile_irrelevant"
        if not report:
            instruction = "no_report"
        intensity_match = re.search(r"(ST|MT|TACTSS|TACTSM)", folder_text)
        intensity = intensity_match.group(1) if intensity_match else "unverified"
        start_index = sum(value.shape[0] for value in arrays)
        arrays.append(epochs)
        for local_trial in range(epochs.shape[0]):
            trial_id = f"{participant_id}-{start_index + local_trial:05d}"
            event_rows.append(
                {
                    "dataset_id": "somato",
                    "participant_id": participant_id,
                    "original_trial_id": trial_id,
                    "onset_seconds": 0.0,
                    "event_type": "median_nerve_stimulation",
                    "intensity": intensity,
                }
            )
            condition_rows.append(
                {
                    "dataset_id": "somato",
                    "participant_id": participant_id,
                    "original_trial_id": trial_id,
                    "report": report,
                    "task_relevance": instruction,
                }
            )
    if not arrays:
        raise FileNotFoundError(f"no somatosensory epoch arrays found under {participant_root}")
    validate_event_rows(event_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "source_epochs.npy", np.concatenate(arrays, axis=0))
    _write_rows(event_rows, output_root / "physical_events.tsv")
    _write_rows(condition_rows, output_root / "condition_table.tsv")
    import scipy.io

    location_path = _somato_location_file(participant_root)
    chanlocs = scipy.io.loadmat(location_path, simplify_cells=True)["chanlocs"]
    _write_rows(
        [
            {"index": index, "name": str(channel["labels"]), "type": "eeg"}
            for index, channel in enumerate(chanlocs)
        ],
        output_root / "channels.tsv",
    )
    descriptor = {
        "dataset_id": "somato",
        "participant_id": participant_id,
        "signal_kind": "epoched_array",
        "signal_path": "source_epochs.npy",
        "source_sampling_rate_hz": config["source_sampling_rate_hz"],
        "source_epoch_tmin_seconds": config["epoch_ms"][0] / 1000,
        "physical_events": "physical_events.tsv",
        "condition_table": "condition_table.tsv",
        "source_amplitude_unit": config.get("source_amplitude_unit", "volt"),
        "stored_amplitude_unit": "volt",
        "channel_location_source": str(location_path.relative_to(participant_root)),
    }
    (output_root / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_rows(rows: list[dict[str, object]], output: Path) -> None:
    import pandas as pd

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)


def _somato_location_file(participant_root: Path) -> Path:
    candidates = sorted(
        path
        for path in participant_root.rglob("*locations.mat")
        if not path.name.startswith("._")
    )
    if not candidates:
        raise UnverifiedMappingError(
            f"no somatosensory channel-location file under {participant_root}"
        )
    return min(
        candidates,
        key=lambda path: (path.name != "EEG_locations.mat", path.as_posix()),
    )


def _write_continuous_standardization(
    *,
    signal: object,
    dataset_id: str,
    participant_id: str,
    event_rows: list[dict[str, object]],
    condition_rows: list[dict[str, object]],
    output_root: Path,
    source_mapping: list[dict[str, object]] | None = None,
    source_provenance: dict[str, object] | None = None,
) -> None:
    validate_event_rows(event_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    signal_path = output_root / "source-raw.fif"
    signal.save(signal_path, overwrite=True, verbose="ERROR")
    _write_rows(event_rows, output_root / "physical_events.tsv")
    _write_rows(condition_rows, output_root / "condition_table.tsv")
    _write_rows(
        [
            {"index": index, "name": name, "type": kind}
            for index, (name, kind) in enumerate(
                zip(signal.ch_names, signal.get_channel_types(), strict=True)
            )
        ],
        output_root / "channels.tsv",
    )
    descriptor = {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "signal_kind": "continuous_raw",
        "signal_path": signal_path.name,
        "source_sampling_rate_hz": float(signal.info["sfreq"]),
        "source_epoch_tmin_seconds": 0.0,
        "physical_events": "physical_events.tsv",
        "condition_table": "condition_table.tsv",
    }
    if source_mapping is not None:
        descriptor["source_mapping"] = source_mapping
    if source_provenance is not None:
        descriptor["source_provenance"] = source_provenance
    (output_root / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
    )


def _trigger_code(value: object) -> int | None:
    match = re.search(r"S\s*(\d+)\s*$", str(value))
    return None if match is None else int(match.group(1))


def _gabor_interrupted_window(trial_types: object) -> bool:
    """Identify an acquisition discontinuity inside a target-trigger window."""

    return any("new segment" in str(value).lower() for value in trial_types)


def _standardize_gabor(
    participant_id: str,
    participant_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> None:
    import mne
    import pandas as pd

    signal_files = sorted(participant_root.rglob("*.vhdr"))
    event_files = sorted(participant_root.rglob("*_events.tsv"))
    if len(signal_files) != 1 or len(event_files) != 1:
        raise UnverifiedMappingError(
            f"expected one BrainVision file and one events table, found "
            f"{len(signal_files)} and {len(event_files)}"
        )
    signal = _read_mne_signal(signal_files[0])
    eog_channels = [
        name for name in config.get("eog_channels", []) if name in signal.ch_names
    ]
    if eog_channels:
        signal.set_channel_types({name: "eog" for name in eog_channels})
    electrode_files = sorted(participant_root.rglob("*_electrodes.tsv"))
    coordinate_files = sorted(participant_root.rglob("*_coordsystem.json"))
    if len(electrode_files) != 1 or len(coordinate_files) != 1:
        raise UnverifiedMappingError("Gabor BIDS electrodes or coordinate metadata are missing")
    electrodes = pd.read_csv(electrode_files[0], sep="\t", encoding="utf-8-sig")
    coordinates = json.loads(coordinate_files[0].read_text(encoding="utf-8"))
    if coordinates.get("EEGCoordinateUnits") != "m":
        raise UnverifiedMappingError("Gabor electrode coordinates are not expressed in metres")
    channel_positions = {}
    for _, row in electrodes.iterrows():
        name = str(row["name"])
        position = np.asarray([row["x"], row["y"], row["z"]], dtype=float)
        if name in signal.ch_names and np.isfinite(position).all():
            channel_positions[name] = position
    eeg_channels = {
        name
        for name, channel_type in zip(
            signal.ch_names, signal.get_channel_types(), strict=True
        )
        if channel_type == "eeg"
    }
    missing_positions = sorted(eeg_channels - channel_positions.keys())
    if missing_positions == ["FCz"] and {"Fz", "CPz"} <= channel_positions.keys():
        midpoint = 2.0 * channel_positions["Fz"] + channel_positions["CPz"]
        radius = float(
            np.median([np.linalg.norm(value) for value in channel_positions.values()])
        )
        channel_positions["FCz"] = radius * midpoint / np.linalg.norm(midpoint)
    elif missing_positions:
        raise UnverifiedMappingError(
            f"Gabor EEG coordinates are missing for {missing_positions}"
        )
    landmarks = coordinates["AnatomicalLandmarkCoordinates"]
    montage = mne.channels.make_dig_montage(
        ch_pos=channel_positions,
        nasion=np.asarray(landmarks["NAS"], dtype=float),
        lpa=np.asarray(landmarks["LPA"], dtype=float),
        rpa=np.asarray(landmarks["RPA"], dtype=float),
        coord_frame="head",
    )
    signal.set_montage(montage, on_missing="ignore")
    table = pd.read_csv(event_files[0], sep="\t", encoding="utf-8-sig")
    codes = np.asarray([_trigger_code(value) for value in table["trial_type"]], dtype=object)
    target_indices = np.flatnonzero(codes == 10)
    outcomes = {
        55: (1, 1),  # target present, seen
        56: (0, 0),  # target absent, unseen
        57: (0, 1),  # target absent, seen (false alarm)
        59: (1, 0),  # target present, unseen
    }
    tilts = {74: "left", 75: "right"}
    event_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    excluded_trigger_windows: list[dict[str, object]] = []
    for trial_index, start in enumerate(target_indices):
        stop = target_indices[trial_index + 1] if trial_index + 1 < len(target_indices) else len(table)
        window = [int(value) for value in codes[start + 1 : stop] if value is not None]
        block = next((value for value in window if value in {70, 71}), None)
        outcome = next((value for value in window if value in outcomes), None)
        # The authors' preprocessing excludes trials without one of the four
        # valid awareness outcomes. Code 58 is the observed invalid-response
        # marker in the pilot and occurs once.
        if outcome is None and 58 in window:
            continue
        if block is None or outcome is None:
            if _gabor_interrupted_window(table.iloc[start:stop]["trial_type"]):
                excluded_trigger_windows.append(
                    {
                        "trial_index": int(trial_index),
                        "reason": "recording discontinuity before block/outcome triggers",
                    }
                )
                continue
            raise UnverifiedMappingError(
                f"Gabor trial {trial_index} lacks a block or outcome trigger"
            )
        target_present, seen = outcomes[outcome]
        tilt_code = next((value for value in window if value in tilts), None)
        if target_present and tilt_code is None:
            raise UnverifiedMappingError(f"present Gabor trial {trial_index} lacks tilt")
        trial_id = f"{participant_id}-{trial_index:04d}"
        common = {
            "dataset_id": "gabor",
            "participant_id": participant_id,
            "original_trial_id": trial_id,
        }
        event_rows.append(
            {
                **common,
                "onset_seconds": float(table.iloc[start]["onset"]),
                "event_type": "gabor_onset",
                "target_present": target_present,
                "orientation": tilts.get(tilt_code, "absent"),
            }
        )
        condition_rows.append(
            {
                **common,
                "seen": seen,
                "block": "localizer" if block == 70 else "experimental",
            }
        )
    _write_continuous_standardization(
        signal=signal,
        dataset_id="gabor",
        participant_id=participant_id,
        event_rows=event_rows,
        condition_rows=condition_rows,
        output_root=output_root,
        source_provenance={
            "trigger_window_exclusions": excluded_trigger_windows,
        },
    )
    (output_root / "montage-imputations.json").write_text(
        json.dumps(
            {
                "FCz": {
                    "reason": "BIDS coordinate is n/a for the online reference",
                    "method": (
                        "normalized 2:1 spherical interpolation of finite Fz and CPz "
                        "coordinates"
                    ),
                    "position_metres": channel_positions["FCz"].tolist(),
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _align_behavior_clock(
    behavior_times: np.ndarray,
    annotation_onsets: np.ndarray,
) -> np.ndarray:
    """Align a monotonic behavioral clock to its contiguous EGI face triggers."""

    if len(annotation_onsets) < len(behavior_times):
        raise UnverifiedMappingError("fewer EGI face triggers than behavioral face events")
    best: tuple[float, float, list[int]] | None = None
    for first_onset in annotation_onsets:
        offset = float(first_onset - behavior_times[0])
        selected: list[int] = []
        previous = -1
        residuals: list[float] = []
        for behavior_time in behavior_times:
            expected = float(behavior_time + offset)
            position = int(np.searchsorted(annotation_onsets, expected))
            candidates = [
                index
                for index in (position - 1, position)
                if previous < index < len(annotation_onsets)
            ]
            if not candidates:
                selected = []
                break
            chosen = min(
                candidates,
                key=lambda index: abs(float(annotation_onsets[index]) - expected),
            )
            selected.append(chosen)
            residuals.append(abs(float(annotation_onsets[chosen]) - expected))
            previous = chosen
        if len(selected) != len(behavior_times):
            continue
        candidate = (max(residuals), float(np.mean(residuals)), selected)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise UnverifiedMappingError("behavior-to-EGI face-trigger alignment failed")
    selected = np.asarray(best[2], dtype=int)
    slope, intercept = np.polyfit(behavior_times, annotation_onsets[selected], 1)
    if not 0.999 <= float(slope) <= 1.001:
        raise UnverifiedMappingError(f"implausible behavior-to-EGI clock slope: {slope}")
    expected = slope * behavior_times + intercept
    rematched: list[int] = []
    previous = -1
    for value in expected:
        position = int(np.searchsorted(annotation_onsets, value))
        candidates = [
            index
            for index in (position - 1, position)
            if previous < index < len(annotation_onsets)
        ]
        if not candidates:
            raise UnverifiedMappingError("affine EGI alignment could not preserve event order")
        chosen = min(
            candidates,
            key=lambda index: abs(float(annotation_onsets[index]) - float(value)),
        )
        rematched.append(chosen)
        previous = chosen
    selected = np.asarray(rematched, dtype=int)
    slope, intercept = np.polyfit(behavior_times, annotation_onsets[selected], 1)
    residual = np.abs(annotation_onsets[selected] - (slope * behavior_times + intercept))
    if float(np.max(residual)) > 0.010:
        raise UnverifiedMappingError(
            f"behavior-to-EGI affine residual exceeds 10 ms: {float(np.max(residual))}"
        )
    return annotation_onsets[selected]


def _kronemer_behavior_files(
    raw_path: Path,
    participant_root: Path,
) -> tuple[list[Path], str]:
    """Resolve outcome-blind Kronemer EEG-to-behavior file mappings.

    The archive contains a small number of harmless naming irregularities:
    spaces versus underscores, Session 2 files mislabeled Session 1, and early
    report-task recordings stored flat while behavior is grouped by run.  The
    resolver first uses exact structural/session evidence and only falls back
    to within-condition acquisition order when raw and behavioral counts agree.
    """
    relative = raw_path.relative_to(participant_root)
    parts = list(relative.parts)
    index = parts.index("EEG_Data")
    parts[index] = "Behavioral_Data"
    behavior_root = participant_root.joinpath(*parts[: index + 1])
    relative_parent = Path(*parts[index + 1 : -1])
    directory = behavior_root / relative_parent
    candidates = sorted(directory.glob("*.csv"))
    if len(candidates) == 1:
        return candidates, "exact_directory"

    all_candidates = sorted(behavior_root.rglob("*.csv")) if behavior_root.exists() else []
    noncalibration = [
        path for path in all_candidates if "calibration" not in path.as_posix().lower()
    ]

    # Some early report participants store EEG for combined runs in a parent
    # directory while behavior is split into one or two run directories.
    run_numbers = [int(value) for value in re.findall(r"\d+", relative_parent.name)]
    if run_numbers and "run" in relative_parent.name.lower():
        run_candidates = []
        for path in noncalibration:
            group = path.relative_to(behavior_root).parts[0]
            candidate_numbers = [int(value) for value in re.findall(r"\d+", group)]
            if set(run_numbers) & set(candidate_numbers):
                run_candidates.append(path)
        if run_candidates:
            return sorted(run_candidates), "run_directory"

    session = re.search(r"Session[_ ](\d+)", raw_path.name, flags=re.IGNORECASE)
    if session is not None:
        session_candidates = [
            path
            for path in noncalibration
            if re.search(
                rf"Session[_ ]*{re.escape(session.group(1))}(?:_|\b)",
                path.name,
                flags=re.IGNORECASE,
            )
        ]
        if len(session_candidates) == 1:
            return session_candidates, "session_label"

        eeg_root = raw_path.parents[len(relative_parent.parts)]
        sibling_raws = sorted(
            path
            for path in eeg_root.rglob("*.raw")
            if "calibration" not in path.as_posix().lower()
        )
        if len(sibling_raws) == len(noncalibration) and raw_path in sibling_raws:
            position = sibling_raws.index(raw_path)
            return [noncalibration[position]], "within_condition_order"

    # Early report-task archives use a trailing task index: task 1 is the
    # calibration recording, task 2 is runs 1/2, and task 3 is runs 3/4.
    task = re.search(r"(\d)\.raw$", raw_path.name, flags=re.IGNORECASE)
    if task is not None and any(
        "calibration" in path.as_posix().lower() for path in all_candidates
    ):
        task_index = int(task.group(1))
        if task_index == 1:
            return [], "calibration_recording"
        groups: dict[str, list[Path]] = {}
        for path in noncalibration:
            group = path.relative_to(behavior_root).parts[0]
            groups.setdefault(group, []).append(path)
        ordered_groups = sorted(
            groups,
            key=lambda value: [int(number) for number in re.findall(r"\d+", value)]
            or [10**9],
        )
        position = task_index - 2
        if 0 <= position < len(ordered_groups):
            return sorted(groups[ordered_groups[position]]), "report_task_order"

    if not noncalibration:
        return [], "missing_behavior"
    raise UnverifiedMappingError(
        f"could not resolve behavioral CSVs for {raw_path}; found {noncalibration}"
    )


def _standardize_kronemer(
    participant_id: str,
    participant_root: Path,
    output_root: Path,
) -> None:
    import mne
    import pandas as pd

    raw_paths = [
        path
        for path in sorted(participant_root.rglob("*.raw"))
        if "calibration" not in path.as_posix().lower()
        and "/EEG_Session/" in path.as_posix()
    ]
    if not raw_paths:
        raise FileNotFoundError(f"no task EGI recordings under {participant_root}")
    report = "_RP_" in f"_{participant_root.name}_"
    raws = []
    source_mapping: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    cumulative_seconds = 0.0
    trial_index = 0
    for raw_path in raw_paths:
        behavior_paths, mapping_method = _kronemer_behavior_files(
            raw_path, participant_root
        )
        behavior_alignment: list[dict[str, object]] = []
        mapping_record: dict[str, object] = {
            "raw_path": str(raw_path.relative_to(participant_root)),
            "behavior_paths": [
                str(path.relative_to(participant_root)) for path in behavior_paths
            ],
            "mapping_method": mapping_method,
            "status": "included" if behavior_paths else "excluded",
            "behavior_alignment": behavior_alignment,
        }
        source_mapping.append(mapping_record)
        if not behavior_paths:
            continue
        raw = mne.io.read_raw_egi(raw_path, preload=False, verbose="ERROR")
        raw.set_montage(
            mne.channels.make_standard_montage("GSN-HydroCel-256"),
            on_missing="ignore",
        )
        face_onsets = np.asarray(
            [
                onset
                for onset, description in zip(
                    raw.annotations.onset,
                    raw.annotations.description,
                    strict=True,
                )
                if str(description) == "Fac2"
            ],
            dtype=float,
        )
        included_behavior_files = 0
        for behavior_path in behavior_paths:
            table = pd.read_csv(behavior_path).dropna(how="all")
            if report:
                rows = table[table["FaceDrawStart"].notna()].copy()
                records = [
                    (float(row["FaceDrawStart"]), int(index), "Face")
                    for index, row in rows.iterrows()
                ]
            else:
                rows = table[table["Task Relevant"].notna()].copy()
                records = []
                for index, row in rows.iterrows():
                    for role in ("Center", "Quadrant"):
                        value = row.get(f"{role} Face time")
                        if pd.notna(value):
                            records.append((float(value), int(index), role))
            records.sort()
            behavior_times = np.asarray(
                [record[0] for record in records], dtype=float
            )
            try:
                aligned = _align_behavior_clock(behavior_times, face_onsets)
            except UnverifiedMappingError as exc:
                behavior_alignment.append(
                    {
                        "behavior_path": str(
                            behavior_path.relative_to(participant_root)
                        ),
                        "status": "excluded",
                        "reason": str(exc),
                    }
                )
                continue
            behavior_alignment.append(
                {
                    "behavior_path": str(behavior_path.relative_to(participant_root)),
                    "status": "included",
                }
            )
            included_behavior_files += 1
            aligned_by_key = {
                (row_index, role): float(onset)
                for (_, row_index, role), onset in zip(records, aligned, strict=True)
            }
            for index, row in rows.iterrows():
                if report:
                    role = "Face"
                    location = row["Face quadrant"]
                    opacity = row["Face opacity"]
                else:
                    role = str(row["Task Relevant"]).strip().title()
                    location = row[f"{role} Face location"]
                    opacity = row[f"{role} Face opacity"]
                trial_id = f"{participant_id}-{trial_index:04d}"
                common = {
                    "dataset_id": "kronemer",
                    "participant_id": participant_id,
                    "original_trial_id": trial_id,
                }
                event_rows.append(
                    {
                        **common,
                        "onset_seconds": cumulative_seconds
                        + aligned_by_key[(int(index), role)],
                        "event_type": "face_onset",
                        "location": str(location),
                        "opacity": float(opacity),
                    }
                )
                condition_rows.append(
                    {
                        **common,
                        "report": int(report),
                        "perceived": int(float(row["Perception answer"]))
                        if pd.notna(row["Perception answer"])
                        else -1,
                        "task_relevance": "report" if report else role.lower(),
                    }
                )
                trial_index += 1
        if included_behavior_files == 0:
            mapping_record["status"] = "excluded"
            continue
        cumulative_seconds += float(raw.n_times / raw.info["sfreq"])
        raws.append(raw)
    if not raws:
        raise UnverifiedMappingError(
            f"no analyzable EEG-to-behavior pairs under {participant_root}"
        )
    signal = mne.concatenate_raws(raws, preload=False) if len(raws) > 1 else raws[0]
    _write_continuous_standardization(
        signal=signal,
        dataset_id="kronemer",
        participant_id=participant_id,
        event_rows=event_rows,
        condition_rows=condition_rows,
        output_root=output_root,
        source_mapping=source_mapping,
    )


def _source_event_table(signal: object, participant_root: Path, config: dict[str, Any]) -> object:
    import pandas as pd

    event_glob = config.get("event_file_glob")
    candidates = sorted(participant_root.glob(event_glob)) if event_glob else sorted(
        participant_root.rglob("*events.tsv")
    )
    if len(candidates) == 1:
        return pd.read_csv(candidates[0], sep="\t")
    if len(candidates) > 1:
        raise UnverifiedMappingError(f"multiple event tables match: {[str(path) for path in candidates]}")
    annotations = getattr(signal, "annotations", None)
    if annotations is None or len(annotations) == 0:
        raise UnverifiedMappingError("no event TSV and no MNE annotations were found")
    return pd.DataFrame(
        {
            "onset": annotations.onset,
            "duration": annotations.duration,
            "description": annotations.description,
        }
    )


def _mapped_value(row: object, source_name: str, index: int) -> object:
    if source_name == "__row_index__":
        return index
    if source_name.startswith("__literal__:"):
        return source_name.partition(":")[2]
    if source_name not in row:
        raise UnverifiedMappingError(f"source event column {source_name!r} is absent")
    return row[source_name]


def _convert_event_rows(
    table: object,
    *,
    dataset_id: str,
    participant_id: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mapping = config["event_columns"]
    physical_fields = config.get("required_physical_fields", [])
    condition_fields = config.get("required_condition_fields", [])
    event_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    for index, row in table.iterrows():
        trial_id = str(_mapped_value(row, mapping["original_trial_id"], index))
        common = {
            "dataset_id": dataset_id,
            "participant_id": participant_id,
            "original_trial_id": trial_id,
        }
        event_rows.append(
            {
                **common,
                "onset_seconds": float(_mapped_value(row, mapping["onset_seconds"], index)),
                "event_type": str(_mapped_value(row, mapping["event_type"], index)),
                **{
                    name: _mapped_value(row, mapping[name], index)
                    for name in physical_fields
                },
            }
        )
        condition_rows.append(
            {
                **common,
                **{
                    name: _mapped_value(row, mapping[name], index)
                    for name in condition_fields
                },
            }
        )
    validate_event_rows(event_rows)
    return event_rows, condition_rows


def standardize_participant(
    dataset_id: str,
    participant_id: str,
    participant_root: Path,
    output_root: Path,
    config_path: Path,
) -> None:
    config = load_yaml(config_path)
    _require_verified(config)
    if not participant_root.exists():
        matches = [
            path
            for path in participant_root.parent.rglob(participant_id)
            if path.is_dir() and path.name == participant_id
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"could not resolve {participant_id} below {participant_root.parent}"
            )
        participant_root = matches[0]
    if dataset_id == "somato":
        _standardize_somato(participant_id, participant_root, output_root, config)
        return
    if dataset_id == "gabor":
        _standardize_gabor(participant_id, participant_root, output_root, config)
        return
    if dataset_id == "kronemer":
        _standardize_kronemer(participant_id, participant_root, output_root)
        return

    inspection = inspect_source_tree(participant_root)
    candidates = [
        participant_root / record["path"]
        for record in inspection["signal_candidates"]
        if record["suffix"] != ".mat"
    ]
    if len(candidates) != 1:
        raise UnverifiedMappingError(
            f"expected one signal file for {dataset_id}/{participant_id}, found {len(candidates)}"
        )
    signal = _read_mne_signal(candidates[0])
    output_root.mkdir(parents=True, exist_ok=True)
    signal_path = output_root / ("source-epo.fif" if hasattr(signal, "events") else "source-raw.fif")
    signal.save(signal_path, overwrite=True, verbose="ERROR")
    source_events = _source_event_table(signal, participant_root, config)
    event_rows, condition_rows = _convert_event_rows(
        source_events,
        dataset_id=dataset_id,
        participant_id=participant_id,
        config=config,
    )
    _write_rows(event_rows, output_root / "physical_events.tsv")
    _write_rows(condition_rows, output_root / "condition_table.tsv")
    descriptor = {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "signal_kind": "epoched_mne" if hasattr(signal, "events") else "continuous_raw",
        "signal_path": signal_path.name,
        "source_sampling_rate_hz": float(signal.info["sfreq"]),
        "source_epoch_tmin_seconds": float(getattr(signal, "tmin", 0.0)),
        "physical_events": "physical_events.tsv",
        "condition_table": "condition_table.tsv",
    }
    (output_root / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    standardize = subparsers.add_parser("standardize")
    for command in (inspect, standardize):
        command.add_argument("--dataset", required=True)
        command.add_argument("--participant", required=True)
    inspect.add_argument("--raw-root", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    standardize.add_argument("--participant-root", type=Path, required=True)
    standardize.add_argument("--output", type=Path, required=True)
    standardize.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        inspect_participant(args.dataset, args.participant, args.raw_root, args.output)
    else:
        standardize_participant(
            args.dataset,
            args.participant,
            args.participant_root,
            args.output,
            args.config,
        )


if __name__ == "__main__":
    main()
