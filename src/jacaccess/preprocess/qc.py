"""Outcome-neutral participant and epoch quality-control decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingThresholds:
    bad_channel_fraction_max: float = 0.15
    removed_ica_fraction_max: float = 0.20
    epoch_peak_to_peak_uv_max: float = 150.0
    robust_z_max: float = 6.0


@dataclass(frozen=True)
class ParticipantQC:
    included: bool
    reasons: tuple[str, ...]


def participant_qc(
    bad_channel_fraction: float,
    removed_ica_fraction: float,
    thresholds: PreprocessingThresholds = PreprocessingThresholds(),
) -> ParticipantQC:
    reasons: list[str] = []
    if bad_channel_fraction >= thresholds.bad_channel_fraction_max:
        reasons.append(
            f"bad-channel fraction {bad_channel_fraction:.3f} is at least "
            f"{thresholds.bad_channel_fraction_max:.3f}"
        )
    if removed_ica_fraction > thresholds.removed_ica_fraction_max:
        reasons.append(
            f"removed-ICA fraction {removed_ica_fraction:.3f} exceeds "
            f"{thresholds.removed_ica_fraction_max:.3f}"
        )
    return ParticipantQC(included=not reasons, reasons=tuple(reasons))

