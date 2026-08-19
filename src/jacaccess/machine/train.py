"""Resumable mixed-precision training for the procedural machine experiment."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from jacaccess.config import load_yaml

_WORKER_CONFIG: dict[str, object] | None = None
_WORKER_SEED: int | None = None


def _initialize_stimulus_worker(config: dict[str, object], seed: int) -> None:
    """Initialize a spawn-safe CPU generator without nested BLAS thread pools."""

    global _WORKER_CONFIG, _WORKER_SEED
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    _WORKER_CONFIG = config
    _WORKER_SEED = seed


def _generate_batch_cpu(indices: np.ndarray) -> tuple[object, object, object, object]:
    """Generate one deterministic batch in a worker process."""

    from jacaccess.machine.stimuli import generate_stimulus_batch

    if _WORKER_CONFIG is None or _WORKER_SEED is None:
        raise RuntimeError("stimulus worker was not initialized")
    generated = generate_stimulus_batch(
        indices,
        seed=_WORKER_SEED,
        image_size=int(_WORKER_CONFIG["image_size"][0]),
        target_probability=float(_WORKER_CONFIG["target_present_probability"]),
        levels=int(_WORKER_CONFIG["contrast_noise_levels"]),
    )
    return generated.images, generated.task_cues, generated.labels, generated.valid_masks


def _prefetched_batches(
    executor: ProcessPoolExecutor,
    batches: Iterable[np.ndarray],
    prefetch: int,
) -> Iterator[tuple[object, object, object, object]]:
    """Keep a bounded ordered queue of CPU-generated batches."""

    iterator = iter(batches)
    pending: deque[Future[tuple[object, object, object, object]]] = deque()
    for _ in range(prefetch):
        try:
            pending.append(executor.submit(_generate_batch_cpu, next(iterator)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(_generate_batch_cpu, next(iterator)))
        except StopIteration:
            pass


def _to_device(generated: tuple[object, object, object, object], device: object) -> object:
    import torch

    images, cues, labels, masks = generated
    return (
        torch.as_tensor(images, device=device),
        torch.as_tensor(cues, device=device),
        {name: torch.as_tensor(value, device=device) for name, value in labels.items()},
        {name: torch.as_tensor(value, device=device) for name, value in masks.items()},
    )


def _cache_values(generated: object) -> tuple[object, object, object, object]:
    return (
        generated.images,
        generated.task_cues,
        generated.labels,
        generated.valid_masks,
    )


def _batch(
    indices: np.ndarray,
    config: dict[str, object],
    seed: int,
    device: object,
    cache: object | None = None,
) -> object:
    import torch

    from jacaccess.machine.stimuli import generate_stimulus_batch

    generated = (
        cache.batch(indices)
        if cache is not None
        else generate_stimulus_batch(
            indices,
            seed=seed,
            image_size=int(config["image_size"][0]),
            target_probability=float(config["target_present_probability"]),
            levels=int(config["contrast_noise_levels"]),
        )
    )
    return (
        torch.as_tensor(generated.images, device=device),
        torch.as_tensor(generated.task_cues, device=device),
        {name: torch.as_tensor(value, device=device) for name, value in generated.labels.items()},
        {
            name: torch.as_tensor(value, device=device)
            for name, value in generated.valid_masks.items()
        },
    )


def _epoch_indices(count: int, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    values = np.arange(count, dtype=np.int64)
    np.random.default_rng(np.random.SeedSequence([seed, epoch])).shuffle(values)
    return [values[start : start + batch_size] for start in range(0, count, batch_size)]


def _validation_loss(
    model: object,
    config: dict[str, object],
    seed: int,
    device: object,
    cache: object | None = None,
) -> float:
    import torch

    from jacaccess.machine.losses import multihead_cross_entropy

    model.eval()
    total = 0.0
    samples = 0
    count = int(config["validation_images"])
    offset = int(config["train_images"])
    with torch.no_grad():
        for local in range(0, count, int(config["batch_size"])):
            indices = np.arange(offset + local, offset + min(local + int(config["batch_size"]), count))
            images, cues, labels, masks = _batch(indices, config, seed, device, cache)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                _, logits = model(images, cues)
                loss, _ = multihead_cross_entropy(logits, labels, masks, cues)
            total += float(loss) * len(indices)
            samples += len(indices)
    return total / samples


def presence_performance_by_bin(
    model: object,
    config: dict[str, object],
    seed: int,
    device: object,
    *,
    split: str,
    cache: object | None = None,
) -> pd.DataFrame:
    """Evaluate final-step presence accuracy in every fixed difficulty bin."""

    import torch

    from jacaccess.machine.stimuli import generate_stimulus_batch

    if split == "validation":
        offset = int(config["train_images"])
        count = int(config["validation_images"])
    elif split == "test":
        offset = int(config["train_images"]) + int(config["validation_images"])
        count = int(config["test_images"])
    else:
        raise ValueError("split must be validation or test")
    levels = int(config["contrast_noise_levels"])
    correct = np.zeros(levels, dtype=np.int64)
    samples = np.zeros(levels, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for local in range(0, count, int(config["batch_size"])):
            indices = np.arange(
                offset + local,
                offset + min(local + int(config["batch_size"]), count),
            )
            generated = (
                cache.batch(indices)
                if cache is not None
                else generate_stimulus_batch(
                    indices,
                    seed=seed,
                    image_size=int(config["image_size"][0]),
                    target_probability=float(config["target_present_probability"]),
                    levels=levels,
                )
            )
            images = torch.as_tensor(generated.images, device=device)
            cues = torch.as_tensor(generated.task_cues, device=device)
            labels = torch.as_tensor(generated.labels["presence"], device=device)
            _, logits = model(images, cues)
            predicted = logits["presence"][:, -1].argmax(-1)
            is_correct = (predicted == labels).cpu().numpy().astype(np.int64)
            difficulty = generated.difficulty_bin
            samples += np.bincount(difficulty, minlength=levels)
            correct += np.bincount(difficulty, weights=is_correct, minlength=levels).astype(
                np.int64
            )
    if np.any(samples == 0):
        raise RuntimeError(f"{split} split has an empty difficulty bin")
    return pd.DataFrame(
        {
            "architecture": model.architecture_name,
            "seed": seed,
            "split": split,
            "difficulty_bin": np.arange(levels, dtype=np.int64),
            "correct_count": correct,
            "sample_count": samples,
            "presence_accuracy": correct / samples,
        }
    )


def train_machine(
    architecture: str,
    seed: int,
    config_path: Path,
    output_directory: Path,
    device_name: str = "cuda",
    cache_directory: Path | None = None,
) -> dict[str, object]:
    import torch

    from jacaccess.machine.architectures import build_architecture, count_parameters
    from jacaccess.machine.cache import StimulusCache
    from jacaccess.machine.losses import multihead_cross_entropy

    config = load_yaml(config_path)
    cache = StimulusCache(cache_directory) if cache_directory is not None else None
    if cache is not None:
        expected = sum(
            int(config[name])
            for name in ("train_images", "validation_images", "test_images")
        )
        if int(cache.manifest["seed"]) != seed or len(cache.images) != expected:
            raise ValueError("stimulus cache does not match the requested seed and sample count")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_architecture(
        architecture,
        hidden=int(config["hidden_widths"][architecture]),
        state_dimensions=int(
            config.get("architecture_state_dimensions", {}).get(
                architecture, config["integration_state_dimensions"]
            )
        ),
        steps=int(config["internal_steps"]),
        parameter_target=int(config["parameter_target"]),
        tolerance_fraction=float(config["parameter_tolerance_fraction"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = output_directory / "checkpoint.pt"
    start_epoch, best_loss, stale, history = 0, float("inf"), 0, []
    best_state = None
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload["best_loss"])
        stale = int(payload["stale"])
        history = list(payload["history"])
        best_state = payload["best_state"]

    data_workers = int(config.get("data_workers", 6))
    prefetch_batches = int(config.get("prefetch_batches", 12))
    if data_workers < 1 or prefetch_batches < 1:
        raise ValueError("data_workers and prefetch_batches must both be positive")
    spawn = mp.get_context("spawn")
    with ExitStack() as stack:
        executor = None
        if cache is None:
            executor = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=data_workers,
                    mp_context=spawn,
                    initializer=_initialize_stimulus_worker,
                    initargs=(config, seed),
                )
            )
        for epoch in range(start_epoch, int(config["max_epochs"])):
            model.train()
            running, samples = 0.0, 0
            index_batches = _epoch_indices(
                int(config["train_images"]), int(config["batch_size"]), seed, epoch
            )
            generated_batches = (
                (_cache_values(cache.batch(indices)) for indices in index_batches)
                if cache is not None
                else _prefetched_batches(executor, index_batches, prefetch_batches)
            )
            for indices, generated in zip(index_batches, generated_batches, strict=True):
                images, cues, labels, masks = _to_device(generated, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=torch.bfloat16,
                ):
                    _, logits = model(images, cues)
                    loss, _ = multihead_cross_entropy(logits, labels, masks, cues)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.detach()) * len(indices)
                samples += len(indices)
            validation = _validation_loss(model, config, seed, device, cache)
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": running / samples,
                    "validation_loss": validation,
                }
            )
            if validation < best_loss - 1e-7:
                best_loss, stale = validation, 0
                best_state = {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                }
            else:
                stale += 1
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_loss": best_loss,
                    "stale": stale,
                    "history": history,
                    "best_state": best_state,
                    "architecture": architecture,
                    "seed": seed,
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
            if stale >= int(config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("machine training produced no valid checkpoint")
    model.load_state_dict(best_state)
    final_path = output_directory / "model.pt"
    torch.save(
        {
            "model": best_state,
            "architecture": architecture,
            "seed": seed,
            "config": config,
        },
        final_path,
    )
    validation_performance = presence_performance_by_bin(
        model,
        config,
        seed,
        device,
        split="validation",
        cache=cache,
    )
    validation_performance.to_parquet(
        output_directory / "validation-presence-by-bin.parquet", index=False
    )
    summary = {
        "architecture": architecture,
        "seed": seed,
        "parameters": count_parameters(model),
        "best_validation_loss": best_loss,
        "epochs": len(history),
        "device": str(device),
        "data_workers": data_workers,
        "prefetch_batches": prefetch_batches,
        "stimulus_cache": None if cache_directory is None else str(cache_directory),
        "validation_presence_by_bin": "validation-presence-by-bin.parquet",
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stimulus-cache", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            train_machine(
                args.architecture,
                args.seed,
                args.config,
                args.output,
                args.device,
                args.stimulus_cache,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
