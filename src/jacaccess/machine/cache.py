"""Reusable memory-mapped procedural stimuli for GPU-saturated training."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from jacaccess.config import load_yaml
from jacaccess.machine.stimuli import StimulusBatch, generate_stimulus_batch

_CACHE_CONFIG: dict[str, object] | None = None
_CACHE_SEED: int | None = None


def _initialize_cache_worker(config: dict[str, object], seed: int) -> None:
    global _CACHE_CONFIG, _CACHE_SEED
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    _CACHE_CONFIG = config
    _CACHE_SEED = seed


def _generate_cache_batch(indices: np.ndarray) -> StimulusBatch:
    if _CACHE_CONFIG is None or _CACHE_SEED is None:
        raise RuntimeError("cache worker was not initialized")
    return generate_stimulus_batch(
        indices,
        seed=_CACHE_SEED,
        image_size=int(_CACHE_CONFIG["image_size"][0]),
        target_probability=float(_CACHE_CONFIG["target_present_probability"]),
        levels=int(_CACHE_CONFIG["contrast_noise_levels"]),
    )


class StimulusCache:
    """Read deterministic batches without regenerating them each epoch."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.images = np.load(directory / "images.npy", mmap_mode="r")
        self.task_cues = np.load(directory / "task_cues.npy", mmap_mode="r")
        self.difficulty_bin = np.load(directory / "difficulty_bin.npy", mmap_mode="r")
        self.labels = {
            name: np.load(directory / f"label-{name}.npy", mmap_mode="r")
            for name in self.manifest["label_names"]
        }
        self.valid_masks = {
            name: np.load(directory / f"mask-{name}.npy", mmap_mode="r")
            for name in self.manifest["label_names"]
        }

    def batch(self, indices: np.ndarray) -> StimulusBatch:
        return StimulusBatch(
            np.asarray(self.images[indices]),
            np.asarray(self.task_cues[indices]),
            {name: np.asarray(values[indices]) for name, values in self.labels.items()},
            {
                name: np.asarray(values[indices])
                for name, values in self.valid_masks.items()
            },
            np.asarray(self.difficulty_bin[indices]),
        )


def _open_array(path: Path, dtype: object, shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def prepare_stimulus_cache(
    config_path: Path,
    seed: int,
    output_directory: Path,
    workers: int,
) -> dict[str, Any]:
    """Materialize every split once, atomically, using exact indexed generation."""

    config = load_yaml(config_path)
    total = sum(int(config[name]) for name in ("train_images", "validation_images", "test_images"))
    image_size = int(config["image_size"][0])
    batch_size = int(config["batch_size"])
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    names = ("presence", "orientation", "location", "contrast_bin", "delayed_action")
    images = _open_array(temporary / "images.npy", np.float32, (total, 1, image_size, image_size))
    cues = _open_array(temporary / "task_cues.npy", np.float32, (total, len(names)))
    difficulty = _open_array(temporary / "difficulty_bin.npy", np.int64, (total,))
    labels = {
        name: _open_array(temporary / f"label-{name}.npy", np.int64, (total,))
        for name in names
    }
    masks = {
        name: _open_array(temporary / f"mask-{name}.npy", np.bool_, (total,))
        for name in names
    }
    batches = [
        np.arange(start, min(start + batch_size, total), dtype=np.int64)
        for start in range(0, total, batch_size)
    ]
    spawn = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=spawn,
        initializer=_initialize_cache_worker,
        initargs=(config, seed),
    ) as executor:
        for indices, generated in zip(
            batches,
            executor.map(_generate_cache_batch, batches, chunksize=1),
            strict=True,
        ):
            images[indices] = generated.images
            cues[indices] = generated.task_cues
            difficulty[indices] = generated.difficulty_bin
            for name in names:
                labels[name][indices] = generated.labels[name]
                masks[name][indices] = generated.valid_masks[name]
    for value in (images, cues, difficulty, *labels.values(), *masks.values()):
        value.flush()
    manifest = {
        "seed": seed,
        "samples": total,
        "image_size": image_size,
        "target_present_probability": float(config["target_present_probability"]),
        "contrast_noise_levels": int(config["contrast_noise_levels"]),
        "label_names": list(names),
        "generation": "exact-indexed-v1",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if output_directory.exists():
        shutil.rmtree(output_directory)
    temporary.replace(output_directory)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_stimulus_cache(args.config, args.seed, args.output, args.workers),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
