"""Non-circular human-neural adaptation of frozen Experiment 1 machines.

The optimization target is a low-level temporal representational-similarity
distribution computed directly from preprocessed EEG.  This module deliberately
does not import the Jacobian geometry or theory-comparison modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jacaccess.config import load_yaml

FORBIDDEN_TARGET_TERMS = {
    "gain",
    "broadcast",
    "persistence",
    "concentration",
    "access_index",
    "rms_distance",
    "cosine_similarity",
    "magnitude_ratio",
}


def performance_gate(delta_accuracy: float, tolerance: float = 0.02) -> bool:
    return abs(float(delta_accuracy)) <= float(tolerance) + 1e-12


def validate_adaptation_config(config: dict[str, Any]) -> None:
    adaptation = config.get("adaptation", {})
    if adaptation.get("uses_final_geometry_in_loss") is not False:
        raise ValueError("uses_final_geometry_in_loss must be explicitly false")
    if adaptation.get("uses_final_geometry_for_checkpoint_selection") is not False:
        raise ValueError(
            "uses_final_geometry_for_checkpoint_selection must be explicitly false"
        )
    serialized = json.dumps(
        {
            "target_type": adaptation.get("target_type"),
            "target_moments": adaptation.get("target_moments"),
            "checkpoint_metric": adaptation.get("checkpoint_metric"),
        }
    ).lower()
    leaked = sorted(term for term in FORBIDDEN_TARGET_TERMS if term in serialized)
    if leaked:
        raise ValueError(f"forbidden final-geometry terms in adaptation target: {leaked}")
    if adaptation.get("checkpoint_metric") != "validation_neural_alignment_loss":
        raise ValueError("checkpoint selection must use validation neural alignment loss")
    if config.get("human_primary_dataset") != "gabor":
        raise ValueError("version 1 supports the frozen Gabor discovery dataset only")
    if int(config.get("outer_subject_folds", 0)) < 2:
        raise ValueError("at least two outer subject folds are required")


def _stable_order(values: list[str], seed: int, label: str) -> list[str]:
    def key(value: str) -> bytes:
        return hashlib.sha256(f"{seed}|{label}|{value}".encode()).digest()

    return sorted(values, key=key)


def subject_splits(
    participants: list[str],
    *,
    folds: int,
    validation_fraction: float,
    seed: int,
) -> list[dict[str, list[str]]]:
    """Create deterministic, balanced, mutually disjoint participant splits."""

    unique = sorted(set(participants))
    if len(unique) != len(participants):
        raise ValueError("participant roster contains duplicates")
    if len(unique) < folds * 2:
        raise ValueError("too few participants for outer participant cross-fitting")
    ordered = _stable_order(unique, seed, "outer")
    assignment = {participant: index % folds for index, participant in enumerate(ordered)}
    result: list[dict[str, list[str]]] = []
    for fold in range(folds):
        heldout = sorted(p for p in unique if assignment[p] == fold)
        available = [p for p in unique if assignment[p] != fold]
        available = _stable_order(available, seed + fold, "inner")
        validation_count = max(1, math.ceil(len(available) * validation_fraction))
        validation = sorted(available[:validation_count])
        train = sorted(available[validation_count:])
        if not train or set(train) & set(validation) or set(train) & set(heldout):
            raise RuntimeError("invalid participant split")
        if set(validation) & set(heldout):
            raise RuntimeError("validation/held-out participant leakage")
        if set(train) | set(validation) | set(heldout) != set(unique):
            raise RuntimeError("participant split does not cover the roster")
        result.append({"train": train, "validation": validation, "heldout": heldout})
    return result


def _rsm_distribution(values: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Return trial-level temporal spatial-cosine RSM mean and variance."""

    if values.ndim != 3:
        raise ValueError("values must have trial, time, feature axes")
    centered = values - values.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(centered, axis=-1, keepdims=True)
    normalized = centered / np.maximum(norms, epsilon)
    trial_rsm = np.einsum("btf,bsf->bts", normalized, normalized, optimize=True)
    return trial_rsm.mean(axis=0), trial_rsm.var(axis=0)


def participant_neural_target(
    directory: Path,
    *,
    window_ms: tuple[float, float],
    steps: int,
) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    epochs = np.load(directory / "epochs.npy", mmap_mode="r")
    times = np.load(directory / "time_seconds.npy")
    requested = np.linspace(window_ms[0] / 1000, window_ms[1] / 1000, steps)
    indices = np.asarray([int(np.argmin(np.abs(times - value))) for value in requested])
    if len(np.unique(indices)) != steps:
        raise ValueError("human sampling grid cannot provide the requested distinct target steps")
    # MNE layout: trial x channel x time.  Work in trial x time x channel.
    selected = np.transpose(np.asarray(epochs[:, :, indices], dtype=np.float64), (0, 2, 1))
    mean, variance = _rsm_distribution(selected)
    return mean, variance, len(selected), times[indices].astype(float).tolist()


def _pool_participants(
    targets: dict[str, tuple[np.ndarray, np.ndarray, int, list[float]]],
    participants: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    # Equal participant weighting prevents high-trial participants dominating.
    means = np.stack([targets[p][0] for p in participants])
    variances = np.stack([targets[p][1] for p in participants])
    pooled_mean = means.mean(axis=0)
    pooled_variance = (variances + (means - pooled_mean) ** 2).mean(axis=0)
    return pooled_mean, pooled_variance, sum(targets[p][2] for p in participants)


def prepare_human_targets(
    *,
    config_path: Path,
    roster_path: Path,
    preprocessed_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_adaptation_config(config)
    roster = pd.read_csv(roster_path, sep="\t", dtype=str).fillna("")
    include = roster["include"].str.lower().isin({"1", "true", "yes"})
    selected = roster[include & roster["dataset_id"].eq("gabor")]
    participants = sorted(selected["participant_id"].tolist())
    split_values = subject_splits(
        participants,
        folds=int(config["outer_subject_folds"]),
        validation_fraction=float(config["validation_fraction_of_nonheldout_subjects"]),
        seed=int(config["participant_split_seed"]),
    )
    adaptation = config["adaptation"]
    window = tuple(float(v) for v in adaptation["target_window_ms"])
    targets = {
        participant: participant_neural_target(
            preprocessed_root / participant,
            window_ms=(window[0], window[1]),
            steps=int(adaptation["target_steps"]),
        )
        for participant in participants
    }
    output_root.mkdir(parents=True, exist_ok=True)
    private_manifest: dict[str, Any] = {"folds": []}
    redacted_manifest: dict[str, Any] = {"folds": []}
    for fold, split in enumerate(split_values):
        directory = output_root / f"fold-{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        train_mean, train_variance, train_trials = _pool_participants(
            targets, split["train"]
        )
        val_mean, val_variance, val_trials = _pool_participants(
            targets, split["validation"]
        )
        np.savez_compressed(
            directory / "neural-targets.npz",
            train_mean=train_mean.astype(np.float32),
            train_variance=train_variance.astype(np.float32),
            validation_mean=val_mean.astype(np.float32),
            validation_variance=val_variance.astype(np.float32),
        )
        private_manifest["folds"].append({"outer_fold": fold, **split})
        redacted_manifest["folds"].append(
            {
                "outer_fold": fold,
                "train_subjects": len(split["train"]),
                "validation_subjects": len(split["validation"]),
                "heldout_subjects": len(split["heldout"]),
                "train_trials": train_trials,
                "validation_trials": val_trials,
                "heldout_subject_hashes": [
                    hashlib.sha256(value.encode()).hexdigest()[:12]
                    for value in split["heldout"]
                ],
            }
        )
    sample_times = next(iter(targets.values()))[3]
    private_manifest.update(
        {
            "target_type": adaptation["target_type"],
            "target_window_ms": list(window),
            "sample_times_seconds": sample_times,
            "participants": len(participants),
        }
    )
    redacted_manifest.update(
        {
            "target_type": adaptation["target_type"],
            "target_window_ms": list(window),
            "sample_times_seconds": sample_times,
            "participants": len(participants),
        }
    )
    (output_root / "split-manifest-private.json").write_text(
        json.dumps(private_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "gabor-outer-fold-manifest-redacted.json").write_text(
        json.dumps(redacted_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return redacted_manifest


def _machine_rsm_distribution(states: Any) -> tuple[Any, Any]:
    import torch

    centered = states.float() - states.float().mean(dim=-1, keepdim=True)
    normalized = centered / torch.linalg.vector_norm(centered, dim=-1, keepdim=True).clamp_min(1e-8)
    trial_rsm = torch.einsum("btf,bsf->bts", normalized, normalized)
    return trial_rsm.mean(dim=0), trial_rsm.var(dim=0, unbiased=False)


def neural_alignment_loss(
    states: Any,
    target_mean: Any,
    target_variance: Any,
    variance_weight: float,
) -> Any:
    import torch

    mean, variance = _machine_rsm_distribution(states)
    steps = int(mean.shape[0])
    mask = ~torch.eye(steps, dtype=torch.bool, device=mean.device)
    return torch.mean((mean[mask] - target_mean[mask]) ** 2) + variance_weight * torch.mean(
        (variance[mask] - target_variance[mask]) ** 2
    )


def _build_model(architecture: str, machine_config: dict[str, Any], device: Any) -> Any:
    from jacaccess.machine.architectures import build_architecture

    return build_architecture(
        architecture,
        hidden=int(machine_config["hidden_widths"][architecture]),
        state_dimensions=int(
            machine_config.get("architecture_state_dimensions", {}).get(
                architecture, machine_config["integration_state_dimensions"]
            )
        ),
        steps=int(machine_config["internal_steps"]),
        parameter_target=int(machine_config["parameter_target"]),
        tolerance_fraction=float(machine_config["parameter_tolerance_fraction"]),
    ).to(device)


def reconstruct_random_model(
    architecture: str, seed: int, machine_config: dict[str, Any], device: Any
) -> Any:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return _build_model(architecture, machine_config, device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trainable_state_parameters(model: Any) -> list[Any]:
    frozen = ("encoder.", "cue_projection.", "heads.")
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(frozen))
        if parameter.requires_grad:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("trainable state-transition scope selected no parameters")
    return selected


def _target_tensors(path: Path, condition: str, fold: int, sham_seed: int, device: Any) -> dict[str, Any]:
    import torch

    values = np.load(path)
    target = {name: np.asarray(values[name]) for name in values.files}
    permutation = np.arange(target["train_mean"].shape[0])
    if condition == "sham_adapted":
        rng = np.random.default_rng(np.random.SeedSequence([sham_seed, fold]))
        permutation = rng.permutation(len(permutation))
        for name in target:
            target[name] = target[name][np.ix_(permutation, permutation)]
    elif condition != "human_adapted":
        raise ValueError("condition must be human_adapted or sham_adapted")
    return {
        **{name: torch.as_tensor(value, device=device) for name, value in target.items()},
        "permutation": permutation.tolist(),
    }


def _fixed_batch(cache: Any, indices: np.ndarray, device: Any) -> tuple[Any, ...]:
    import torch

    generated = cache.batch(indices)
    return (
        torch.as_tensor(generated.images, device=device),
        torch.as_tensor(generated.task_cues, device=device),
        {name: torch.as_tensor(value, device=device) for name, value in generated.labels.items()},
        {name: torch.as_tensor(value, device=device) for name, value in generated.valid_masks.items()},
    )


@dataclass(frozen=True)
class AdaptationResult:
    architecture: str
    seed: int
    outer_fold: int
    condition: str
    adaptation_steps: int
    best_step: int
    neural_alignment_loss_before: float
    neural_alignment_loss_after: float
    validation_neural_alignment_loss: float
    relative_l2_parameter_displacement: float
    trainable_parameters: int
    total_parameters: int
    trainable_fraction: float
    task_accuracy_before: float
    task_accuracy_after: float
    accuracy_change: float
    performance_gate_passed: bool
    wall_time_seconds: float
    parent_checkpoint: str
    parent_checkpoint_sha256: str


def adapt_model(
    *,
    architecture: str,
    seed: int,
    outer_fold: int,
    condition: str,
    extension_config_path: Path,
    machine_config_path: Path,
    target_path: Path,
    parent_checkpoint: Path,
    cache_directory: Path,
    output_directory: Path,
    device_name: str = "cuda",
    max_steps_override: int | None = None,
) -> AdaptationResult:
    import torch

    from jacaccess.machine.cache import StimulusCache
    from jacaccess.machine.losses import multihead_cross_entropy
    from jacaccess.machine.train import presence_performance_by_bin

    started = time.monotonic()
    extension = load_yaml(extension_config_path)
    validate_adaptation_config(extension)
    machine = load_yaml(machine_config_path)
    adaptation = extension["adaptation"]
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    random.seed(seed + 10000 * outer_fold)
    np.random.seed(seed + 10000 * outer_fold)
    torch.manual_seed(seed + 10000 * outer_fold)
    torch.cuda.manual_seed_all(seed + 10000 * outer_fold)
    model = _build_model(architecture, machine, device)
    payload = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != architecture or int(payload.get("seed", -1)) != seed:
        raise ValueError("parent checkpoint lineage mismatch")
    model.load_state_dict(payload["model"])
    parent_state = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainable = _trainable_state_parameters(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(adaptation["learning_rate"]),
        weight_decay=float(adaptation["weight_decay"]),
    )
    targets = _target_tensors(
        target_path,
        condition,
        outer_fold,
        int(extension["sham"]["seed"]),
        device,
    )
    cache = StimulusCache(cache_directory)
    batch_size = int(adaptation["batch_size"])
    max_steps = int(max_steps_override or adaptation["max_steps"])
    evaluation_interval = int(adaptation["evaluation_interval_steps"])
    patience = int(adaptation["early_stopping_patience_evaluations"])
    variance_weight = float(adaptation["variance_loss_weight"])
    validation_offset = int(machine["train_images"])
    validation_indices = np.arange(validation_offset, validation_offset + batch_size)
    validation_batch = _fixed_batch(cache, validation_indices, device)

    def alignment(batch: tuple[Any, ...], prefix: str) -> Any:
        states, _ = model(batch[0], batch[1])
        return neural_alignment_loss(
            states,
            targets[f"{prefix}_mean"],
            targets[f"{prefix}_variance"],
            variance_weight,
        )

    model.eval()
    with torch.no_grad():
        initial_train_loss = float(alignment(validation_batch, "train"))
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    generator = np.random.default_rng(np.random.SeedSequence([seed, outer_fold, 101]))
    for step in range(1, max_steps + 1):
        indices = generator.integers(0, int(machine["train_images"]), size=batch_size)
        batch = _fixed_batch(cache, indices, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        states, logits = model(batch[0], batch[1])
        neural = neural_alignment_loss(
            states,
            targets["train_mean"],
            targets["train_variance"],
            variance_weight,
        )
        task, _ = multihead_cross_entropy(logits, batch[2], batch[3], batch[1])
        anchor_terms = [
            torch.mean((parameter - parent_state[name]) ** 2)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        anchor = torch.stack(anchor_terms).mean()
        loss = neural + float(adaptation["lambda_task"]) * task + float(
            adaptation["lambda_anchor"]
        ) * anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(adaptation["gradient_clip_norm"]))
        optimizer.step()
        if step % evaluation_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                validation_loss = float(alignment(validation_batch, "validation"))
            history.append(
                {
                    "step": step,
                    "objective": float(loss.detach()),
                    "neural_alignment_loss": float(neural.detach()),
                    "task_loss": float(task.detach()),
                    "anchor_loss": float(anchor.detach()),
                    "validation_neural_alignment_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_step = step
                stale = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("adaptation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_neural = float(alignment(validation_batch, "train"))

    original_test = pd.read_parquet(parent_checkpoint.parent / "test-presence-by-bin.parquet")
    difficulty = int(extension["selected_difficulty_bin"])
    before = float(original_test.loc[original_test["difficulty_bin"] == difficulty, "presence_accuracy"].iloc[0])
    after_table = presence_performance_by_bin(
        model, machine, seed, device, split="test", cache=cache
    )
    after = float(after_table.loc[after_table["difficulty_bin"] == difficulty, "presence_accuracy"].iloc[0])
    delta = after - before
    numerator = torch.sqrt(
        sum(
            torch.sum((parameter.detach() - parent_state[name]) ** 2)
            for name, parameter in model.named_parameters()
        )
    )
    denominator = torch.sqrt(
        sum(torch.sum(value**2) for value in parent_state.values())
    ).clamp_min(1e-12)
    displacement = float((numerator / denominator).cpu())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    tolerance = float(extension["retention"]["accuracy_tolerance_absolute"])
    result = AdaptationResult(
        architecture=architecture,
        seed=seed,
        outer_fold=outer_fold,
        condition=condition,
        adaptation_steps=int(history[-1]["step"]),
        best_step=best_step,
        neural_alignment_loss_before=initial_train_loss,
        neural_alignment_loss_after=final_neural,
        validation_neural_alignment_loss=best_loss,
        relative_l2_parameter_displacement=displacement,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        trainable_fraction=trainable_parameters / total_parameters,
        task_accuracy_before=before,
        task_accuracy_after=after,
        accuracy_change=delta,
        performance_gate_passed=performance_gate(delta, tolerance),
        wall_time_seconds=time.monotonic() - started,
        parent_checkpoint=str(parent_checkpoint),
        parent_checkpoint_sha256=_sha256(parent_checkpoint),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "architecture": architecture,
            "seed": seed,
            "outer_fold": outer_fold,
            "stage": condition,
            "parent_checkpoint": str(parent_checkpoint),
            "parent_checkpoint_sha256": result.parent_checkpoint_sha256,
            "target_permutation": targets["permutation"],
            "config": extension,
        },
        output_directory / "model.pt",
    )
    after_table.to_parquet(output_directory / "test-presence-by-bin.parquet", index=False)
    (output_directory / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "summary.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-targets")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--roster", type=Path, required=True)
    prepare.add_argument("--preprocessed-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    reconstruct = subparsers.add_parser("reconstruct-random")
    reconstruct.add_argument("--architecture", required=True)
    reconstruct.add_argument("--seed", type=int, required=True)
    reconstruct.add_argument("--machine-config", type=Path, required=True)
    reconstruct.add_argument("--output", type=Path, required=True)
    reconstruct.add_argument("--device", default="cpu")
    adapt = subparsers.add_parser("adapt")
    adapt.add_argument("--architecture", required=True)
    adapt.add_argument("--seed", type=int, required=True)
    adapt.add_argument("--outer-fold", type=int, required=True)
    adapt.add_argument("--condition", choices=["human_adapted", "sham_adapted"], required=True)
    adapt.add_argument("--config", type=Path, required=True)
    adapt.add_argument("--machine-config", type=Path, required=True)
    adapt.add_argument("--target", type=Path, required=True)
    adapt.add_argument("--parent-checkpoint", type=Path, required=True)
    adapt.add_argument("--stimulus-cache", type=Path, required=True)
    adapt.add_argument("--output", type=Path, required=True)
    adapt.add_argument("--device", default="cuda")
    adapt.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.command == "prepare-targets":
        result = prepare_human_targets(
            config_path=args.config,
            roster_path=args.roster,
            preprocessed_root=args.preprocessed_root,
            output_root=args.output,
        )
    elif args.command == "reconstruct-random":
        import torch

        machine_config = load_yaml(args.machine_config)
        device = torch.device(args.device)
        model = reconstruct_random_model(
            args.architecture, args.seed, machine_config, device
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "architecture": args.architecture,
                "seed": args.seed,
                "stage": "random_init",
                "random_initialization": "exact_training_seed_reconstruction",
                "config": machine_config,
            },
            args.output,
        )
        result = {
            "architecture": args.architecture,
            "seed": args.seed,
            "random_initialization": "exact_training_seed_reconstruction",
            "output": str(args.output),
        }
    else:
        result = asdict(
            adapt_model(
                architecture=args.architecture,
                seed=args.seed,
                outer_fold=args.outer_fold,
                condition=args.condition,
                extension_config_path=args.config,
                machine_config_path=args.machine_config,
                target_path=args.target,
                parent_checkpoint=args.parent_checkpoint,
                cache_directory=args.stimulus_cache,
                output_directory=args.output,
                device_name=args.device,
                max_steps_override=args.max_steps,
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
