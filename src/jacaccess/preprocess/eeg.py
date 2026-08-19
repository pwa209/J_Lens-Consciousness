"""MNE/PyPREP/ICLabel outcome-blind preprocessing."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jacaccess.config import load_yaml
from jacaccess.preprocess.artifacts import epoch_artifact_mask
from jacaccess.preprocess.qc import participant_qc


@dataclass(frozen=True)
class PreprocessResult:
    included: bool
    valid_trials: int
    rejected_trials: int
    bad_channel_fraction: float
    removed_ica_fraction: float
    deviations: tuple[str, ...]


def _load_descriptor(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "descriptor.json").read_text(encoding="utf-8"))


def _resample_before_continuous_filters(
    raw: Any,
    target_sampling_rate_hz: float,
    *,
    n_jobs: int,
) -> bool:
    """Anti-alias resample long recordings before memory-intensive FIR filters.

    Kronemer recordings contain about 6.7 million samples across 257 channels.
    Filtering that complete 1000-Hz matrix makes MNE allocate several full-size
    work arrays and exceeds the effective per-process memory ceiling on AutoDL.
    MNE's resampler applies its anti-aliasing low-pass before decimation, so doing
    this first preserves the requested 100-Hz analysis bandwidth while reducing
    all subsequent filter, PyPREP, and ICA arrays by roughly tenfold.
    """
    source_sampling_rate_hz = float(raw.info["sfreq"])
    target_sampling_rate_hz = float(target_sampling_rate_hz)
    if target_sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if target_sampling_rate_hz >= source_sampling_rate_hz:
        return False
    raw.resample(target_sampling_rate_hz, n_jobs=n_jobs, verbose="ERROR")
    return True


def _notch_before_continuous_resampling(
    raw: Any,
    notch_hz: float,
    *,
    quality_factor: float,
) -> bool:
    """Apply a channel-wise, zero-phase IIR line-noise notch before decimation.

    The usual whole-array FIR notch is too memory intensive for the longest
    1000-Hz Kronemer recordings. ``Raw.apply_function`` supplies one channel at
    a time, so the IIR notch has bounded working memory while ensuring line
    noise is removed before it could alias into the 100-Hz representation.
    """
    sampling_rate_hz = float(raw.info["sfreq"])
    notch_hz = float(notch_hz)
    if notch_hz <= 0 or notch_hz >= sampling_rate_hz / 2:
        return False
    if quality_factor <= 0:
        raise ValueError("continuous_notch_q must be positive")
    def apply_notch(values: np.ndarray) -> np.ndarray:
        from scipy.signal import filtfilt, iirnotch

        numerator, denominator = iirnotch(notch_hz, quality_factor, fs=sampling_rate_hz)
        return filtfilt(numerator, denominator, values, axis=-1)

    raw.apply_function(
        apply_notch,
        picks="eeg",
        channel_wise=True,
        n_jobs=1,
        verbose="ERROR",
    )
    return True


def _preprocess_continuous(
    source_directory: Path,
    descriptor: dict[str, Any],
    dataset_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], float, float, list[str]]:
    import mne
    import pandas as pd
    from mne.preprocessing import ICA

    raw = mne.io.read_raw_fif(source_directory / descriptor["signal_path"], preload=True)
    raw.pick(picks="eeg")
    original_channel_count = len(raw.ch_names)
    filter_jobs = int(config.get("continuous_filter_jobs", 1))
    target_sampling_rate_hz = float(config["sampling_rate_hz"])
    notch = float(dataset_config["notch_hz"])
    notch_applied_before_resampling = _notch_before_continuous_resampling(
        raw,
        notch,
        quality_factor=float(config.get("continuous_notch_q", 30.0)),
    )
    resampled_early = bool(config.get("continuous_resample_before_filter", True)) and (
        _resample_before_continuous_filters(
            raw,
            target_sampling_rate_hz,
            n_jobs=filter_jobs,
        )
    )
    if not notch_applied_before_resampling and notch < raw.info["sfreq"] / 2:
        raw.notch_filter([notch], n_jobs=filter_jobs, verbose="ERROR")
    low, high = config["bandpass_hz"]
    raw.filter(
        low,
        high,
        phase="zero",
        fir_design="firwin",
        n_jobs=filter_jobs,
        verbose="ERROR",
    )

    try:
        from pyprep.find_noisy_channels import NoisyChannels

        noisy = NoisyChannels(raw.copy(), random_state=config["random_seed"])
        noisy.find_all_bads(ransac=True)
        raw.info["bads"] = sorted(set(noisy.get_bads()))
        del noisy
        gc.collect()
    except Exception as exc:
        raise RuntimeError(f"PyPREP bad-channel detection failed: {exc}") from exc
    bad_fraction = len(raw.info["bads"]) / max(original_channel_count, 1)

    ica_copy = raw.copy().filter(
        config["ica_highpass_hz"],
        None,
        phase="zero",
        fir_design="firwin",
        n_jobs=filter_jobs,
        verbose="ERROR",
    )
    good_channels = original_channel_count - len(raw.info["bads"])
    components = min(config["ica_components_max"], good_channels - 1)
    ica = ICA(
        n_components=components,
        method="picard",
        random_state=config["random_seed"],
        max_iter="auto",
    )
    ica.fit(ica_copy, picks="eeg", verbose="ERROR")
    labels: dict[str, Any]
    try:
        from mne_icalabel import label_components

        labels = label_components(ica_copy, ica, method="iclabel")
    except Exception as exc:
        raise RuntimeError(f"ICLabel failed: {exc}") from exc
    thresholds = config["iclabel_thresholds"]
    excluded: list[int] = []
    for index, (label, probabilities) in enumerate(
        zip(labels["labels"], labels["y_pred_proba"], strict=True)
    ):
        normalized = str(label).lower().replace(" ", "_")
        threshold = thresholds.get(normalized)
        probability = (
            float(probabilities)
            if np.ndim(probabilities) == 0
            else float(np.max(probabilities))
        )
        if threshold is not None and probability >= threshold:
            excluded.append(index)
    removed_fraction = len(excluded) / max(components, 1)
    ica.exclude = excluded
    ica.apply(raw, verbose="ERROR")
    del ica_copy, ica, labels
    gc.collect()
    raw.interpolate_bads(reset_bads=False, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if not resampled_early and not np.isclose(raw.info["sfreq"], target_sampling_rate_hz):
        raw.resample(target_sampling_rate_hz, n_jobs=filter_jobs, verbose="ERROR")

    events_table = pd.read_csv(source_directory / descriptor["physical_events"], sep="\t")
    samples = (
        np.rint(events_table["onset_seconds"].to_numpy() * raw.info["sfreq"]).astype(int)
        + raw.first_samp
    )
    events = np.column_stack([samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)])
    tmin, tmax = (value / 1000 for value in dataset_config["epoch_ms"])
    baseline = tuple(value / 1000 for value in config["baseline_ms"])
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"stimulus": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    data = epochs.get_data(copy=True)
    trial_ids = (
        events_table.iloc[epochs.selection]["original_trial_id"].astype(str).tolist()
    )
    deviations: list[str] = []
    if notch_applied_before_resampling:
        deviations.append(
            f"zero-phase {notch:g} Hz IIR line-noise notch preceded downsampling "
            "to prevent aliasing with bounded memory"
        )
    if resampled_early:
        deviations.append(
            "anti-alias resampling preceded continuous FIR filtering to keep "
            "long-recording memory bounded"
        )
    return data, epochs.times, trial_ids, bad_fraction, removed_fraction, deviations


def _preprocess_epoched(
    source_directory: Path,
    descriptor: dict[str, Any],
    dataset_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], float, float, list[str]]:
    import mne
    import pandas as pd

    if descriptor["signal_kind"] == "epoched_mne":
        epochs = mne.read_epochs(
            source_directory / descriptor["signal_path"],
            preload=True,
            verbose="ERROR",
        )
        epochs.pick(picks="eeg")
    else:
        values = np.load(source_directory / descriptor["signal_path"], mmap_mode="r")
        info = mne.create_info(
            [f"EEG{index + 1:03d}" for index in range(values.shape[1])],
            sfreq=descriptor["source_sampling_rate_hz"],
            ch_types="eeg",
        )
        epochs = mne.EpochsArray(
            np.asarray(values, dtype=np.float64),
            info,
            tmin=descriptor["source_epoch_tmin_seconds"],
            verbose="ERROR",
        )
    filter_jobs = int(config.get("epoched_filter_jobs", 4))
    notch = float(dataset_config["notch_hz"])
    data = epochs.get_data(copy=True)
    if notch < epochs.info["sfreq"] / 2:
        # Recent MNE releases do not expose EpochsArray.notch_filter. Apply
        # the public array-level filters and rebuild the epochs object while
        # preserving its timing and channel metadata.
        data = mne.filter.notch_filter(
            data,
            epochs.info["sfreq"],
            [notch],
            n_jobs=filter_jobs,
            verbose="ERROR",
        )
    low, high = config["bandpass_hz"]
    data = mne.filter.filter_data(
        data,
        epochs.info["sfreq"],
        low,
        high,
        phase="zero",
        fir_design="firwin",
        n_jobs=filter_jobs,
        verbose="ERROR",
    )
    epochs = mne.EpochsArray(
        data,
        epochs.info.copy(),
        events=epochs.events.copy(),
        tmin=epochs.tmin,
        event_id=epochs.event_id,
        baseline=epochs.baseline,
        metadata=epochs.metadata,
        verbose="ERROR",
    )
    epochs.resample(config["sampling_rate_hz"], n_jobs=filter_jobs, verbose="ERROR")
    epochs.set_eeg_reference("average", projection=False, verbose="ERROR")
    events = pd.read_csv(source_directory / descriptor["physical_events"], sep="\t")
    return (
        epochs.get_data(copy=True),
        epochs.times,
        events["original_trial_id"].astype(str).tolist(),
        0.0,
        0.0,
        ("ICA unavailable because the source provides epoched matrices",),
    )


def preprocess_participant(
    source_directory: Path,
    output_directory: Path,
    dataset_config_path: Path,
    preprocessing_config_path: Path,
) -> PreprocessResult:
    descriptor = _load_descriptor(source_directory)
    dataset_config = load_yaml(dataset_config_path)
    config = load_yaml(preprocessing_config_path)
    if descriptor["signal_kind"] == "continuous_raw":
        loaded = _preprocess_continuous(source_directory, descriptor, dataset_config, config)
    else:
        loaded = _preprocess_epoched(source_directory, descriptor, dataset_config, config)
    epochs, times, trial_ids, bad_fraction, removed_fraction, deviations = loaded
    reject, artifact_scores = epoch_artifact_mask(
        epochs,
        peak_to_peak_uv_max=float(
            dataset_config.get("epoch_peak_to_peak_uv", config["epoch_peak_to_peak_uv"])
        ),
        robust_z_max=config["robust_z_threshold"],
    )
    qc = participant_qc(bad_fraction, removed_fraction)
    valid_trials = int((~reject).sum())
    minimum_valid = int(config.get("minimum_valid_trials", 40))
    trial_deviations: tuple[str, ...] = ()
    if valid_trials < minimum_valid:
        trial_deviations = (
            f"{valid_trials} valid trials is below the minimum {minimum_valid}",
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    np.save(output_directory / "epochs.npy", epochs[~reject].astype(np.float32))
    np.save(output_directory / "time_seconds.npy", np.asarray(times, dtype=np.float64))
    (output_directory / "trial_ids.json").write_text(
        json.dumps(np.asarray(trial_ids)[~reject].tolist(), indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output_directory / "artifact_scores.npz", rejected=reject, **artifact_scores)
    valid_ids = set(np.asarray(trial_ids)[~reject].astype(str))
    for name in ("physical_events.tsv", "condition_table.tsv"):
        source = source_directory / name
        if not source.exists():
            continue
        import pandas as pd

        table = pd.read_csv(source, sep="\t")
        if "original_trial_id" not in table:
            raise ValueError(f"{source} lacks original_trial_id")
        filtered = table[table["original_trial_id"].astype(str).isin(valid_ids)]
        filtered.to_csv(output_directory / name, sep="\t", index=False)
    descriptor_source = source_directory / "descriptor.json"
    if descriptor_source.exists():
        shutil.copy2(descriptor_source, output_directory / "source_descriptor.json")
    result = PreprocessResult(
        included=qc.included and valid_trials >= minimum_valid,
        valid_trials=valid_trials,
        rejected_trials=int(reject.sum()),
        bad_channel_fraction=bad_fraction,
        removed_ica_fraction=removed_fraction,
        deviations=tuple(deviations) + qc.reasons + trial_deviations,
    )
    (output_directory / "qc.json").write_text(
        json.dumps(result.__dict__, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    args = parser.parse_args()
    result = preprocess_participant(
        args.source,
        args.output,
        args.dataset_config,
        args.preprocessing_config,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
