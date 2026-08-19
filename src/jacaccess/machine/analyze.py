"""Machine Jacobian signatures and causal state-subspace interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jacaccess.config import load_yaml


def _broadcast(jacobian: object, model: object, epsilon: float = 1e-8) -> object:
    import torch

    # future_logit_vector is time-major, with every head concatenated per step.
    head_sizes = [head.out_features for head in model.heads.values()]
    per_time = sum(head_sizes)
    future_steps = jacobian.shape[-2] // per_time
    cursor = 0
    energies = []
    for size in head_sizes:
        rows = []
        for step in range(future_steps):
            start = step * per_time + cursor
            rows.extend(range(start, start + size))
        energies.append(torch.sum(jacobian[:, rows] ** 2, dim=(-2, -1)))
        cursor += size
    energy = torch.stack(energies, dim=-1)
    probability = energy / torch.clamp(energy.sum(-1, keepdim=True), min=epsilon)
    entropy = -torch.sum(probability * torch.log(torch.clamp(probability, min=epsilon)), dim=-1)
    return entropy / np.log(len(head_sizes))


def _roll_final(model: object, state: object, drive: object, context: object, step: int) -> object:
    current, current_context = state, context
    for future in range(step + 1, model.steps):
        current, current_context = model.advance(current, drive, current_context, future)
    return {name: head(current) for name, head in model.heads.items()}


def _score(logits: dict[str, object], labels: dict[str, object], masks: dict[str, object]) -> float:
    scores = []
    for name, values in logits.items():
        mask = masks[name].bool()
        if mask.any():
            scores.append((values.argmax(-1)[mask] == labels[name][mask]).float().mean())
    return float(sum(scores) / len(scores))


def _repeat_context(context: object, repeats: int) -> object:
    """Repeat architecture context for flattened random-subspace batches."""

    if context is None:
        return None
    if isinstance(context, list):
        return [value.repeat((repeats,) + (1,) * (value.ndim - 1)) for value in context]
    return context.repeat((repeats,) + (1,) * (context.ndim - 1))


def _scores_by_repeat(
    logits: dict[str, object],
    labels: dict[str, object],
    masks: dict[str, object],
    repeats: int,
) -> object:
    """Return one multihead accuracy score per repeated intervention."""

    import torch

    batch = next(iter(labels.values())).shape[0]
    scores = []
    for name, values in logits.items():
        reshaped = values.reshape(repeats, batch, -1)
        mask = masks[name].bool()
        if mask.any():
            correct = reshaped.argmax(-1)[:, mask] == labels[name][mask].unsqueeze(0)
            scores.append(correct.float().mean(dim=1))
    return torch.stack(scores).mean(dim=0)


def analyze_machine(
    architecture: str,
    seed: int,
    config_path: Path,
    model_path: Path,
    output_directory: Path,
    device_name: str = "cuda",
) -> dict[str, object]:
    import torch

    from jacaccess.machine.architectures import build_architecture
    from jacaccess.machine.jacobian import exact_future_logit_jacobians
    from jacaccess.machine.stimuli import generate_stimulus_batch

    config = load_yaml(config_path)
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
    payload = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    output_directory.mkdir(parents=True, exist_ok=True)
    test_start = int(config["train_images"]) + int(config["validation_images"])
    test_count = int(config["test_images"])
    batch_size = min(int(config["batch_size"]), 128)
    signature_rows: list[dict[str, object]] = []
    top_drops: list[float] = []
    levels = int(config["contrast_noise_levels"])
    test_correct = np.zeros(levels, dtype=np.int64)
    test_samples = np.zeros(levels, dtype=np.int64)
    random_drops: list[list[float]] = [
        [] for _ in range(int(config["random_subspaces"]))
    ]
    intervention_step = int(config["intervention_step"]) - 1
    signature_steps = tuple(int(value) - 1 for value in config["signature_steps"])
    generator = torch.Generator(device=device).manual_seed(seed + 8181)
    for local in range(0, test_count, batch_size):
        indices = np.arange(test_start + local, test_start + min(local + batch_size, test_count))
        generated = generate_stimulus_batch(
            indices,
            seed=seed,
            image_size=int(config["image_size"][0]),
            target_probability=float(config["target_present_probability"]),
            levels=int(config["contrast_noise_levels"]),
        )
        images = torch.as_tensor(generated.images, device=device)
        cues = torch.as_tensor(generated.task_cues, device=device)
        labels = {name: torch.as_tensor(value, device=device) for name, value in generated.labels.items()}
        masks = {
            name: torch.as_tensor(value, device=device)
            for name, value in generated.valid_masks.items()
        }
        with torch.no_grad():
            drive = model.initial_drive(images, cues)
            states, contexts = model.trace_from_drive(drive)
            baseline_logits = _roll_final(
                model, states[:, intervention_step], drive, contexts[intervention_step], intervention_step
            )
            baseline_score = _score(baseline_logits, labels, masks)
            presence_correct = (
                baseline_logits["presence"].argmax(-1) == labels["presence"]
            ).cpu().numpy()
        difficulty = generated.difficulty_bin
        test_samples += np.bincount(difficulty, minlength=levels)
        test_correct += np.bincount(
            difficulty,
            weights=presence_correct.astype(np.int64),
            minlength=levels,
        ).astype(np.int64)
        jacobians = exact_future_logit_jacobians(model, images, cues, signature_steps)
        decompositions = {
            step: torch.linalg.svd(jacobian, full_matrices=False)
            for step, jacobian in jacobians.items()
        }
        ordered_steps = sorted(jacobians)
        for step, jacobian in jacobians.items():
            _, singular, vh = decompositions[step]
            squared = singular**2
            total = squared.sum(-1)
            probability = squared / torch.clamp(total[:, None], min=1e-8)
            effective_rank = torch.exp(
                -torch.sum(probability * torch.log(torch.clamp(probability, min=1e-8)), dim=-1)
            )
            concentration = squared[:, :4].sum(-1) / torch.clamp(total, min=1e-8)
            broadcast = _broadcast(jacobian, model)
            next_steps = [candidate for candidate in ordered_steps if candidate > step]
            if next_steps:
                later_vh = decompositions[next_steps[0]][2]
                overlap = vh[:, :4] @ later_vh[:, :4].transpose(-1, -2)
                canonical = torch.linalg.svdvals(overlap)
                persistence = torch.mean(canonical**2, dim=-1)
            else:
                persistence = torch.full_like(total, float("nan"))
            for row, sample_id in enumerate(indices):
                signature_rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "sample_id": int(sample_id),
                        "difficulty_bin": int(generated.difficulty_bin[row]),
                        "presence_correct": bool(presence_correct[row]),
                        "step": step + 1,
                        "gain": float(total[row] / jacobian.shape[-2]),
                        "broadcast": float(broadcast[row]),
                        "persistence": float(persistence[row]),
                        "concentration": float(concentration[row]),
                        "effective_rank": float(effective_rank[row]),
                    }
                )
        _, _, vh = decompositions[intervention_step]
        top_basis = vh[:, :4]
        state = states[:, intervention_step]
        projected = torch.einsum("bkd,bd->bk", top_basis, state)
        ablated = state - torch.einsum("bkd,bk->bd", top_basis, projected)
        with torch.no_grad():
            score = _score(
                _roll_final(model, ablated, drive, contexts[intervention_step], intervention_step),
                labels,
                masks,
            )
        top_drops.append(baseline_score - score)
        random_count = int(config["random_subspaces"])
        random_chunk = int(config.get("random_intervention_chunk", 10))
        if random_chunk < 1:
            raise ValueError("random_intervention_chunk must be positive")
        for random_start in range(0, random_count, random_chunk):
            count = min(random_chunk, random_count - random_start)
            random_matrices = torch.randn(
                count,
                state.shape[-1],
                4,
                generator=generator,
                device=device,
            )
            random_bases = torch.linalg.qr(random_matrices, mode="reduced").Q.transpose(-1, -2)
            projected = torch.einsum("kqd,bd->kbq", random_bases, state)
            random_ablated = state.unsqueeze(0) - torch.einsum(
                "kqd,kbq->kbd", random_bases, projected
            )
            flattened_state = random_ablated.reshape(count * len(state), state.shape[-1])
            flattened_drive = drive.repeat((count,) + (1,) * (drive.ndim - 1))
            flattened_context = _repeat_context(contexts[intervention_step], count)
            with torch.no_grad():
                random_scores = _scores_by_repeat(
                    _roll_final(
                        model,
                        flattened_state,
                        flattened_drive,
                        flattened_context,
                        intervention_step,
                    ),
                    labels,
                    masks,
                    count,
                )
            for offset, random_score in enumerate(random_scores.tolist()):
                random_drops[random_start + offset].append(baseline_score - random_score)

    pd.DataFrame(signature_rows).to_parquet(
        output_directory / "jacobian-signatures.parquet", index=False
    )
    if np.any(test_samples == 0):
        raise RuntimeError("test split has an empty difficulty bin")
    pd.DataFrame(
        {
            "architecture": architecture,
            "seed": seed,
            "split": "test",
            "difficulty_bin": np.arange(levels, dtype=np.int64),
            "correct_count": test_correct,
            "sample_count": test_samples,
            "presence_accuracy": test_correct / test_samples,
        }
    ).to_parquet(output_directory / "test-presence-by-bin.parquet", index=False)
    top_drop = float(np.mean(top_drops))
    random_distribution = np.asarray([np.mean(values) for values in random_drops])
    percentile = float(np.mean(random_distribution <= top_drop))
    result = {
        "architecture": architecture,
        "seed": seed,
        "top_subspace_accuracy_drop": top_drop,
        "random_drop_mean": float(random_distribution.mean()),
        "random_drop_95_percentile": float(np.quantile(random_distribution, 0.95)),
        "top_subspace_percentile": percentile,
        "passes_intervention_criterion": percentile >= float(config["random_percentile_required"]),
        "test_presence_by_bin": "test-presence-by-bin.parquet",
    }
    np.save(output_directory / "random-intervention-drops.npy", random_distribution)
    (output_directory / "intervention.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_machine(
                args.architecture,
                args.seed,
                args.config,
                args.model,
                args.output,
                args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
