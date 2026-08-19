"""Deterministic, participant-level stratified cross-fitting."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _hash_integer(*parts: object) -> int:
    text = "".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")


def assign_folds(
    dataset_id: str,
    participant_id: str,
    original_trial_ids: Sequence[str],
    strata: Sequence[str],
    folds: int = 5,
    seed: int = 20260730,
) -> NDArray[np.int_]:
    """Assign trials to deterministic folds while balancing each stratum.

    Strata may use condition labels solely for fold allocation. The returned
    integer assignments are the only information passed onward.
    """

    if folds < 2:
        raise ValueError("folds must be at least two")
    if len(original_trial_ids) != len(strata):
        raise ValueError("trial IDs and strata must have the same length")
    if len(set(original_trial_ids)) != len(original_trial_ids):
        raise ValueError("original trial IDs must be unique within participant")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[str(stratum)].append(index)

    assignments = np.empty(len(original_trial_ids), dtype=int)
    for stratum, indices in sorted(grouped.items()):
        indices.sort(
            key=lambda index: _hash_integer(
                dataset_id,
                participant_id,
                original_trial_ids[index],
                seed,
            )
        )
        offset = _hash_integer(dataset_id, participant_id, stratum, seed) % folds
        for position, index in enumerate(indices):
            assignments[index] = (position + offset) % folds
    return assignments


@dataclass(frozen=True)
class FoldSplit:
    train: NDArray[np.int_]
    validation: NDArray[np.int_]
    test: NDArray[np.int_]


def split_fold(
    dataset_id: str,
    participant_id: str,
    original_trial_ids: Sequence[str],
    assignments: NDArray[np.int_],
    heldout_fold: int,
    validation_fraction: float = 0.10,
    seed: int = 20260730,
) -> FoldSplit:
    """Split one held-out fold and deterministically reserve validation trials."""

    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and one half")
    assignments = np.asarray(assignments)
    if assignments.shape != (len(original_trial_ids),):
        raise ValueError("assignment shape does not match trial IDs")

    test = np.flatnonzero(assignments == heldout_fold)
    candidate_train = np.flatnonzero(assignments != heldout_fold)
    if test.size == 0 or candidate_train.size < 2:
        raise ValueError("fold has insufficient train or test trials")

    ranked = sorted(
        candidate_train.tolist(),
        key=lambda index: _hash_integer(
            dataset_id,
            participant_id,
            original_trial_ids[index],
            seed,
            "validation",
        ),
    )
    validation_count = max(1, math.floor(len(ranked) * validation_fraction))
    validation = np.asarray(ranked[:validation_count], dtype=int)
    train = np.asarray(ranked[validation_count:], dtype=int)
    return FoldSplit(train=train, validation=validation, test=test)

