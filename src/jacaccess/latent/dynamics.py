"""Residual low-rank nonlinear dynamics used for human latent trajectories."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised on the GPU environment
    raise ImportError(
        "jacaccess.latent.dynamics requires PyTorch; run the AutoDL bootstrap first"
    ) from exc


class ResidualDynamics(nn.Module):
    """Thirty-two-dimensional residual dynamics with an analytic Jacobian."""

    def __init__(
        self,
        state_dimensions: int = 32,
        hidden_dimensions: int = 64,
        input_dimensions: int = 0,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if state_dimensions < 1 or hidden_dimensions < 1 or input_dimensions < 0:
            raise ValueError("model dimensions are invalid")
        self.state_dimensions = state_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.input_dimensions = input_dimensions
        self.alpha = alpha

        self.A = nn.Parameter(torch.zeros(state_dimensions, state_dimensions))
        self.U = nn.Parameter(torch.empty(state_dimensions, hidden_dimensions))
        self.V = nn.Parameter(torch.empty(hidden_dimensions, state_dimensions))
        self.B = nn.Parameter(torch.empty(hidden_dimensions, input_dimensions))
        self.b = nn.Parameter(torch.zeros(hidden_dimensions))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        if input_dimensions:
            nn.init.xavier_uniform_(self.B)

    def step(self, state: Tensor, physical_input: Tensor | None = None) -> Tensor:
        if physical_input is None:
            physical_input = state.new_zeros((*state.shape[:-1], self.input_dimensions))
        q = state @ self.V.T + physical_input @ self.B.T + self.b
        return state + self.alpha * (state @ self.A.T + torch.tanh(q) @ self.U.T)

    def forward(self, state: Tensor, physical_input: Tensor | None = None) -> Tensor:
        return self.step(state, physical_input)

    def analytic_jacobian(
        self,
        state: Tensor,
        physical_input: Tensor | None = None,
    ) -> Tensor:
        if physical_input is None:
            physical_input = state.new_zeros((*state.shape[:-1], self.input_dimensions))
        q = state @ self.V.T + physical_input @ self.B.T + self.b
        slope = 1.0 - torch.tanh(q).square()
        nonlinear = torch.einsum("dh,...h,he->...de", self.U, slope, self.V)
        identity = torch.eye(
            self.state_dimensions,
            device=state.device,
            dtype=state.dtype,
        )
        return identity + self.alpha * (self.A + nonlinear)

