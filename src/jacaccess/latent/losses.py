"""Prespecified multi-horizon loss for residual human dynamics."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("jacaccess.latent.losses requires PyTorch") from exc


@dataclass(frozen=True)
class LossComponents:
    total: Tensor
    one_step_mse: Tensor
    rollout_4_mse: Tensor
    rollout_8_mse: Tensor
    stability_penalty: Tensor


def endpoint_rollout_mse(
    model: object,
    state_windows: Tensor,
    input_windows: Tensor,
    horizon: int,
) -> Tensor:
    if state_windows.shape[1] < horizon + 1:
        raise ValueError("state window is shorter than requested rollout")
    current = state_windows[:, 0]
    for step in range(horizon):
        current = model.step(current, input_windows[:, step])
    return torch.mean((current - state_windows[:, horizon]) ** 2)


def estimated_spectral_norm(
    jacobians: Tensor,
    iterations: int = 3,
    epsilon: float = 1e-8,
) -> Tensor:
    """Differentiable fixed-start power iteration for batched matrices."""

    if jacobians.shape[-1] != jacobians.shape[-2]:
        raise ValueError("jacobians must be square")
    vector = torch.ones(
        (*jacobians.shape[:-2], jacobians.shape[-1], 1),
        device=jacobians.device,
        dtype=jacobians.dtype,
    )
    vector = vector / torch.linalg.vector_norm(vector, dim=-2, keepdim=True)
    for _ in range(iterations):
        left = jacobians @ vector
        left = left / torch.clamp(
            torch.linalg.vector_norm(left, dim=-2, keepdim=True),
            min=epsilon,
        )
        vector = jacobians.transpose(-1, -2) @ left
        vector = vector / torch.clamp(
            torch.linalg.vector_norm(vector, dim=-2, keepdim=True),
            min=epsilon,
        )
    return torch.linalg.vector_norm(jacobians @ vector, dim=(-2, -1))


def multi_horizon_loss(
    model: object,
    state_windows: Tensor,
    input_windows: Tensor,
    *,
    rollout_4_weight: float = 0.5,
    rollout_8_weight: float = 0.25,
    stability_weight: float = 0.001,
    stability_threshold: float = 1.5,
    power_iterations: int = 3,
) -> LossComponents:
    """Calculate one-step, endpoint rollouts and local stability penalty."""

    if state_windows.ndim != 3 or input_windows.ndim != 3:
        raise ValueError("windows must have shape [batch, time, feature]")
    if state_windows.shape[:2] != (
        input_windows.shape[0],
        input_windows.shape[1] + 1,
    ):
        raise ValueError("input windows need one fewer time sample than states")
    if state_windows.shape[1] < 9:
        raise ValueError("eight-step loss requires at least nine state samples")

    predicted_next = model.step(state_windows[:, :-1], input_windows)
    one_step = torch.mean((predicted_next - state_windows[:, 1:]) ** 2)
    rollout_4 = endpoint_rollout_mse(model, state_windows, input_windows, 4)
    rollout_8 = endpoint_rollout_mse(model, state_windows, input_windows, 8)

    jacobians = model.analytic_jacobian(state_windows[:, :-1], input_windows)
    spectral = estimated_spectral_norm(jacobians, power_iterations)
    stability = torch.mean(torch.relu(spectral - stability_threshold) ** 2)
    total = (
        one_step
        + rollout_4_weight * rollout_4
        + rollout_8_weight * rollout_8
        + stability_weight * stability
    )
    return LossComponents(total, one_step, rollout_4, rollout_8, stability)

