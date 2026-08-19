"""Common event-row contract used by repository-specific adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

CORE_EVENT_FIELDS = frozenset(
    {
        "dataset_id",
        "participant_id",
        "original_trial_id",
        "onset_seconds",
        "event_type",
    }
)

PRIMARY_MODEL_FORBIDDEN_FIELDS = frozenset(
    {
        "awareness",
        "perceived",
        "seen",
        "report",
        "task",
        "response",
        "gaze",
        "pupil",
        "eye_classifier_confidence",
    }
)


def validate_event_rows(rows: Iterable[Mapping[str, object]]) -> None:
    keys_seen: set[tuple[str, str, str]] = set()
    count = 0
    for row_index, row in enumerate(rows):
        count += 1
        missing = CORE_EVENT_FIELDS - row.keys()
        if missing:
            raise ValueError(f"event row {row_index} misses {sorted(missing)}")
        key = (
            str(row["dataset_id"]),
            str(row["participant_id"]),
            str(row["original_trial_id"]),
        )
        if key in keys_seen:
            raise ValueError(f"duplicate event key {key}")
        keys_seen.add(key)
    if count == 0:
        raise ValueError("event table is empty")


def assert_primary_model_fields_safe(field_names: Iterable[str]) -> None:
    normalized = {name.strip().lower() for name in field_names}
    leaked = normalized & PRIMARY_MODEL_FORBIDDEN_FIELDS
    if leaked:
        raise ValueError(f"forbidden fields in primary model input: {sorted(leaked)}")

