"""Deterministic training, early stopping and checkpoint recovery."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("jacaccess.latent.train requires PyTorch") from exc

from jacaccess.latent.dynamics import ResidualDynamics
from jacaccess.latent.losses import LossComponents, multi_horizon_loss


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_transitions: int = 4096
    max_epochs: int = 200
    patience: int = 15
    gradient_clip_norm: float = 1.0
    rollout_4_weight: float = 0.5
    rollout_8_weight: float = 0.25
    stability_weight: float = 0.001
    stability_threshold: float = 1.5
    power_iterations: int = 3
    seed: int = 20260730


@dataclass(frozen=True)
class ModelQC:
    one_step_mse: float
    heldout_r2: float
    persistence_mse: float
    improvement_over_persistence: float


@dataclass
class TrainingResult:
    model: ResidualDynamics
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    history: list[dict[str, float]]


class TrajectoryWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, states: Tensor, physical_inputs: Tensor, horizon: int = 8) -> None:
        if states.ndim != 3 or physical_inputs.ndim != 3:
            raise ValueError("states and inputs must have [trial, time, feature] shape")
        if states.shape[:2] != physical_inputs.shape[:2]:
            raise ValueError("states and inputs must share trial and time axes")
        if states.shape[1] <= horizon:
            raise ValueError("trajectories are too short for requested horizon")
        self.states = states
        self.inputs = physical_inputs
        self.horizon = horizon
        self.starts_per_trial = states.shape[1] - horizon

    def __len__(self) -> int:
        return self.states.shape[0] * self.starts_per_trial

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        trial = index // self.starts_per_trial
        start = index % self.starts_per_trial
        return (
            self.states[trial, start : start + self.horizon + 1],
            self.inputs[trial, start : start + self.horizon],
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loss_kwargs(config: TrainingConfig) -> dict[str, float | int]:
    return {
        "rollout_4_weight": config.rollout_4_weight,
        "rollout_8_weight": config.rollout_8_weight,
        "stability_weight": config.stability_weight,
        "stability_threshold": config.stability_threshold,
        "power_iterations": config.power_iterations,
    }


def _evaluate_loader(
    model: ResidualDynamics,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    config: TrainingConfig,
) -> tuple[float, dict[str, float]]:
    model.eval()
    totals: dict[str, float] = {
        "total": 0.0,
        "one_step": 0.0,
        "rollout_4": 0.0,
        "rollout_8": 0.0,
        "stability": 0.0,
    }
    samples = 0
    with torch.no_grad():
        for state_window, input_window in loader:
            state_window = state_window.to(device)
            input_window = input_window.to(device)
            components = multi_horizon_loss(
                model,
                state_window,
                input_window,
                **_loss_kwargs(config),
            )
            batch = state_window.shape[0]
            samples += batch
            totals["total"] += float(components.total) * batch
            totals["one_step"] += float(components.one_step_mse) * batch
            totals["rollout_4"] += float(components.rollout_4_mse) * batch
            totals["rollout_8"] += float(components.rollout_8_mse) * batch
            totals["stability"] += float(components.stability_penalty) * batch
    means = {name: value / samples for name, value in totals.items()}
    return means["total"], means


def _atomic_checkpoint(payload: dict[str, Any], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)


def fit_residual_dynamics(
    *,
    training_states: Tensor,
    training_inputs: Tensor,
    validation_states: Tensor,
    validation_inputs: Tensor,
    hidden_dimensions: int = 64,
    alpha: float = 0.1,
    config: TrainingConfig = TrainingConfig(),
    checkpoint_path: Path | None = None,
    resume: bool = True,
    device: str | torch.device | None = None,
) -> TrainingResult:
    """Fit one fold and recover safely from the latest epoch checkpoint."""

    _seed_everything(config.seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ResidualDynamics(
        state_dimensions=training_states.shape[-1],
        hidden_dimensions=hidden_dimensions,
        input_dimensions=training_inputs.shape[-1],
        alpha=alpha,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    transition_batch = max(1, config.batch_transitions // 8)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        TrajectoryWindowDataset(training_states, training_inputs),
        batch_size=transition_batch,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TrajectoryWindowDataset(validation_states, validation_inputs),
        batch_size=transition_batch,
        shuffle=False,
    )

    start_epoch = 0
    best_epoch = -1
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    best_state: dict[str, Tensor] | None = None
    if checkpoint_path and resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload["best_epoch"])
        best_loss = float(payload["best_validation_loss"])
        stale_epochs = int(payload["stale_epochs"])
        history = list(payload["history"])
        best_state = payload["best_model_state"]

    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        running_loss = 0.0
        samples = 0
        for state_window, input_window in train_loader:
            state_window = state_window.to(selected_device)
            input_window = input_window.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            components: LossComponents = multi_horizon_loss(
                model,
                state_window,
                input_window,
                **_loss_kwargs(config),
            )
            components.total.backward()
            clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch = state_window.shape[0]
            samples += batch
            running_loss += float(components.total.detach()) * batch

        validation_loss, validation_parts = _evaluate_loader(
            model,
            validation_loader,
            selected_device,
            config,
        )
        record = {
            "epoch": float(epoch),
            "training_loss": running_loss / samples,
            "validation_loss": validation_loss,
            **{f"validation_{name}": value for name, value in validation_parts.items()},
        }
        history.append(record)
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
        else:
            stale_epochs += 1

        if checkpoint_path:
            _atomic_checkpoint(
                {
                    "schema_version": 1,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "stale_epochs": stale_epochs,
                    "history": history,
                    "model_state": model.state_dict(),
                    "best_model_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "training_config": asdict(config),
                },
                checkpoint_path,
            )
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(selected_device)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        epochs_completed=len(history),
        history=history,
    )


def evaluate_model_qc(
    model: ResidualDynamics,
    heldout_states: Tensor,
    heldout_inputs: Tensor,
    device: str | torch.device | None = None,
) -> ModelQC:
    selected_device = torch.device(device or next(model.parameters()).device)
    states = heldout_states.to(selected_device)
    inputs = heldout_inputs.to(selected_device)
    model.eval()
    with torch.no_grad():
        prediction = model.step(states[:, :-1], inputs[:, :-1])
        target = states[:, 1:]
        model_mse = torch.mean((prediction - target) ** 2)
        persistence_mse = torch.mean((states[:, :-1] - target) ** 2)
        variance = torch.mean((target - target.mean(dim=(0, 1), keepdim=True)) ** 2)
        r_squared = 1.0 - model_mse / torch.clamp(variance, min=1e-12)
        improvement = (persistence_mse - model_mse) / torch.clamp(persistence_mse, min=1e-12)
    return ModelQC(
        one_step_mse=float(model_mse),
        heldout_r2=float(r_squared),
        persistence_mse=float(persistence_mse),
        improvement_over_persistence=float(improvement),
    )

