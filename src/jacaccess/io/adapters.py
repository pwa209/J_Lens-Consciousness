"""Dataset-adapter interface, intentionally gated until source inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class AdapterNotVerifiedError(RuntimeError):
    pass


@dataclass(frozen=True)
class StandardizedParticipantPaths:
    epochs: Path
    events: Path
    channels: Path
    condition_table: Path
    provenance: Path


class DatasetAdapter(Protocol):
    dataset_id: str

    def inspect(self, participant_id: str, raw_root: Path) -> dict[str, object]:
        """Return source-format facts without writing derivatives."""

    def standardize(
        self,
        participant_id: str,
        raw_root: Path,
        output_root: Path,
    ) -> StandardizedParticipantPaths:
        """Write common events/channels and outcome-blind signal inputs."""


class PendingSourceInspectionAdapter:
    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id

    def _raise(self) -> None:
        raise AdapterNotVerifiedError(
            f"{self.dataset_id} adapter requires one downloaded participant and "
            "source-format inspection before implementation"
        )

    def inspect(self, participant_id: str, raw_root: Path) -> dict[str, object]:
        del participant_id, raw_root
        self._raise()

    def standardize(
        self,
        participant_id: str,
        raw_root: Path,
        output_root: Path,
    ) -> StandardizedParticipantPaths:
        del participant_id, raw_root, output_root
        self._raise()


class VerifiedRepositoryAdapter:
    """Adapter backed by the source-verified standardization implementations."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id

    def inspect(self, participant_id: str, raw_root: Path) -> dict[str, object]:
        from jacaccess.io.source_inspection import inspect_source_tree

        participant_root = raw_root / participant_id
        if not participant_root.exists():
            matches = [
                path for path in raw_root.rglob(participant_id) if path.is_dir()
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"could not resolve {participant_id} below {raw_root}"
                )
            participant_root = matches[0]
        return {
            **inspect_source_tree(participant_root),
            "dataset_id": self.dataset_id,
            "participant_id": participant_id,
        }

    def standardize(
        self,
        participant_id: str,
        raw_root: Path,
        output_root: Path,
    ) -> StandardizedParticipantPaths:
        from jacaccess.io.standardize import standardize_participant

        repository_root = Path(__file__).resolve().parents[3]
        standardize_participant(
            self.dataset_id,
            participant_id,
            raw_root / participant_id,
            output_root,
            repository_root / "configs" / "datasets" / f"{self.dataset_id}.yaml",
        )
        signal = (
            output_root / "source_epochs.npy"
            if (output_root / "source_epochs.npy").exists()
            else output_root / "source-raw.fif"
        )
        return StandardizedParticipantPaths(
            epochs=signal,
            events=output_root / "physical_events.tsv",
            channels=output_root / "channels.tsv",
            condition_table=output_root / "condition_table.tsv",
            provenance=output_root / "descriptor.json",
        )

def get_adapter(dataset_id: str) -> DatasetAdapter:
    if dataset_id not in {"kronemer", "gabor", "somato"}:
        raise ValueError(f"unknown dataset {dataset_id!r}")
    return VerifiedRepositoryAdapter(dataset_id)
