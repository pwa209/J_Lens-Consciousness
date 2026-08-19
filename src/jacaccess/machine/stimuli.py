"""Deterministic procedural threshold-vision task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

HEAD_NAMES = (
    "presence",
    "orientation",
    "location",
    "contrast_bin",
    "delayed_action",
)


@dataclass(frozen=True)
class StimulusBatch:
    images: FloatArray
    task_cues: FloatArray
    labels: dict[str, IntArray]
    valid_masks: dict[str, NDArray[np.bool_]]
    difficulty_bin: IntArray


def _one_stimulus(
    *,
    index: int,
    seed: int,
    image_size: int,
    target_probability: float,
    levels: int,
) -> tuple[np.ndarray, int, int, int, int]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    present = int(rng.random() < target_probability)
    orientation = int(rng.integers(0, 2))
    location = int(rng.integers(0, 4))
    level = int(rng.integers(0, levels))

    contrast_levels = np.geomspace(0.03, 0.75, levels)
    noise_levels = np.geomspace(0.35, 0.05, levels)
    image = rng.normal(0.0, noise_levels[level], size=(image_size, image_size))
    if present:
        centers = (
            (0.30, 0.30),
            (0.70, 0.30),
            (0.30, 0.70),
            (0.70, 0.70),
        )
        center_x, center_y = centers[location]
        coordinates = np.arange(image_size, dtype=np.float64)
        x, y = np.meshgrid(coordinates, coordinates)
        x = x - center_x * (image_size - 1)
        y = y - center_y * (image_size - 1)
        angle = np.deg2rad(-45.0 if orientation == 0 else 45.0)
        rotated_x = x * np.cos(angle) + y * np.sin(angle)
        sigma = image_size / 10.0
        envelope = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        carrier = np.cos(2.0 * np.pi * 0.12 * rotated_x)
        image = image + contrast_levels[level] * envelope * carrier
    image = np.clip(image, -1.0, 1.0).astype(np.float32)
    return image, present, orientation, location, level


def generate_stimulus_batch(
    indices: IntArray,
    *,
    seed: int = 20260730,
    image_size: int = 64,
    target_probability: float = 2 / 3,
    levels: int = 12,
) -> StimulusBatch:
    """Generate samples by global index, independent of batch order."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0):
        raise ValueError("indices must be a vector of non-negative integers")
    if not 0 < target_probability < 1:
        raise ValueError("target_probability must be between zero and one")

    records = [
        _one_stimulus(
            index=int(index),
            seed=seed,
            image_size=image_size,
            target_probability=target_probability,
            levels=levels,
        )
        for index in indices
    ]
    images = np.stack([record[0] for record in records])[:, None]
    presence = np.asarray([record[1] for record in records], dtype=np.int64)
    orientation = np.asarray([record[2] for record in records], dtype=np.int64)
    location = np.asarray([record[3] for record in records], dtype=np.int64)
    difficulty = np.asarray([record[4] for record in records], dtype=np.int64)

    cue_ids = np.asarray(
        [
            np.random.default_rng(np.random.SeedSequence([seed, int(index), 991])).integers(
                0, len(HEAD_NAMES)
            )
            for index in indices
        ],
        dtype=np.int64,
    )
    cues = np.eye(len(HEAD_NAMES), dtype=np.float32)[cue_ids]
    target_present = presence.astype(bool)
    labels = {
        "presence": presence,
        "orientation": orientation,
        "location": location,
        "contrast_bin": difficulty,
        # The delayed head repeats the presence decision at the final processing
        # step. It is scored only at that step; all other heads are scored at
        # every registered step.
        "delayed_action": presence.copy(),
    }
    masks = {
        "presence": np.ones(len(indices), dtype=bool),
        "orientation": target_present,
        "location": target_present,
        "contrast_bin": target_present,
        "delayed_action": np.ones(len(indices), dtype=bool),
    }
    return StimulusBatch(images, cues, labels, masks, difficulty)

