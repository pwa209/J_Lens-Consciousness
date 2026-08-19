"""Five parameter-matched systems testing distinct integration claims."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("jacaccess.machine.architectures requires PyTorch") from exc

HEAD_DIMENSIONS = {
    "presence": 2,
    "orientation": 2,
    "location": 4,
    "contrast_bin": 12,
    "delayed_action": 2,
}

DEFAULT_HIDDEN_WIDTHS = {
    "feedforward": 2200,
    "recurrent": 6650,
    "shared_workspace": 1460,
    "private_modules": 4350,
    "unlimited_shared_state": 710,
}

DEFAULT_STATE_DIMENSIONS = {
    "feedforward": 32,
    "recurrent": 32,
    "shared_workspace": 32,
    "private_modules": 32,
    # Equal to the combined capacity of four 32-D specialist states.
    "unlimited_shared_state": 128,
}


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class SharedEncoder(nn.Module):
    def __init__(self, state_dimensions: int = 32) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.features = nn.Linear(128 * 8 * 8, 128)
        self.to_state = nn.Linear(128, state_dimensions)

    def forward(self, images: Tensor) -> Tensor:
        features = self.convolution(images)
        if features.shape[-2:] != (8, 8):
            raise ValueError("the common encoder expects 64 x 64 images")
        return self.to_state(torch.nn.functional.gelu(self.features(features.flatten(1))))


class ResidualMLP(nn.Module):
    def __init__(self, input_dimensions: int, output_dimensions: int, hidden: int) -> None:
        super().__init__()
        self.first = nn.Linear(input_dimensions, hidden)
        self.second = nn.Linear(hidden, output_dimensions)

    def forward(self, values: Tensor) -> Tensor:
        return self.second(torch.nn.functional.gelu(self.first(values)))


class VisionSystem(nn.Module):
    architecture_name = "base"

    def __init__(self, state_dimensions: int = 32, steps: int = 6) -> None:
        super().__init__()
        self.state_dimensions = state_dimensions
        self.steps = steps
        self.encoder = SharedEncoder(state_dimensions)
        self.cue_projection = nn.Linear(5, state_dimensions)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(state_dimensions, size) for name, size in HEAD_DIMENSIONS.items()}
        )

    def initial_drive(self, images: Tensor, task_cues: Tensor) -> Tensor:
        return torch.tanh(self.encoder(images) + self.cue_projection(task_cues))

    def initialize(self, drive: Tensor) -> tuple[Tensor, Any]:
        raise NotImplementedError

    def advance(
        self,
        state: Tensor,
        drive: Tensor,
        context: Any,
        step: int,
    ) -> tuple[Tensor, Any]:
        raise NotImplementedError

    def trace_from_drive(self, drive: Tensor) -> tuple[Tensor, list[Any]]:
        state, context = self.initialize(drive)
        states: list[Tensor] = []
        contexts: list[Any] = []
        for step in range(self.steps):
            state, context = self.advance(state, drive, context, step)
            states.append(state)
            contexts.append(context)
        return torch.stack(states, dim=1), contexts

    def future_logit_vector(
        self,
        state: Tensor,
        drive: Tensor,
        start_step: int,
        context: Any,
    ) -> Tensor:
        """Map a 32-D state to current and future logits, holding context fixed."""

        if not 0 <= start_step < self.steps:
            raise ValueError("start_step is outside the processing sequence")
        vectors = [torch.cat([head(state) for head in self.heads.values()], dim=-1)]
        current = state
        current_context = context
        for step in range(start_step + 1, self.steps):
            current, current_context = self.advance(
                current,
                drive,
                current_context,
                step,
            )
            vectors.append(torch.cat([head(current) for head in self.heads.values()], dim=-1))
        return torch.cat(vectors, dim=-1)

    def forward(self, images: Tensor, task_cues: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if task_cues.shape != (images.shape[0], 5):
            raise ValueError("task cues must have shape [batch, 5]")
        states, _ = self.trace_from_drive(self.initial_drive(images, task_cues))
        logits = {
            name: torch.stack([head(states[:, step]) for step in range(self.steps)], dim=1)
            for name, head in self.heads.items()
        }
        return states, logits


class FeedforwardSystem(VisionSystem):
    architecture_name = "feedforward"

    def __init__(self, hidden: int, state_dimensions: int = 32, steps: int = 6) -> None:
        super().__init__(state_dimensions, steps)
        self.blocks = nn.ModuleList(
            [ResidualMLP(state_dimensions, state_dimensions, hidden) for _ in range(steps)]
        )

    def initialize(self, drive: Tensor) -> tuple[Tensor, None]:
        return drive, None

    def advance(
        self,
        state: Tensor,
        drive: Tensor,
        context: None,
        step: int,
    ) -> tuple[Tensor, None]:
        return state + 0.1 * self.blocks[step](state), None


class RecurrentSystem(VisionSystem):
    architecture_name = "recurrent"

    def __init__(self, hidden: int, state_dimensions: int = 32, steps: int = 6) -> None:
        super().__init__(state_dimensions, steps)
        self.cell = ResidualMLP(2 * state_dimensions, 2 * state_dimensions, hidden)

    def initialize(self, drive: Tensor) -> tuple[Tensor, None]:
        return drive, None

    def advance(
        self,
        state: Tensor,
        drive: Tensor,
        context: None,
        step: int,
    ) -> tuple[Tensor, None]:
        del step
        candidate, gate = self.cell(torch.cat([state, drive], dim=-1)).chunk(2, dim=-1)
        return state + torch.sigmoid(gate) * torch.tanh(candidate), None


class SharedWorkspaceSystem(VisionSystem):
    architecture_name = "shared_workspace"

    def __init__(self, hidden: int, state_dimensions: int = 32, steps: int = 6) -> None:
        super().__init__(state_dimensions, steps)
        self.specialist_initializers = nn.ModuleList(
            [nn.Linear(state_dimensions, state_dimensions) for _ in range(4)]
        )
        self.specialist_updates = nn.ModuleList(
            [ResidualMLP(2 * state_dimensions, state_dimensions, hidden) for _ in range(4)]
        )
        self.workspace_update = ResidualMLP(
            5 * state_dimensions,
            state_dimensions,
            hidden,
        )

    def initialize(self, drive: Tensor) -> tuple[Tensor, list[Tensor]]:
        specialists = [torch.tanh(initializer(drive)) for initializer in self.specialist_initializers]
        return drive, specialists

    def advance(
        self,
        state: Tensor,
        drive: Tensor,
        context: list[Tensor],
        step: int,
    ) -> tuple[Tensor, list[Tensor]]:
        del drive, step
        specialists = [
            specialist + 0.1 * update(torch.cat([specialist, state], dim=-1))
            for specialist, update in zip(context, self.specialist_updates, strict=True)
        ]
        workspace = state + 0.1 * self.workspace_update(torch.cat([*specialists, state], dim=-1))
        return workspace, specialists


class UnlimitedSharedStateSystem(VisionSystem):
    """Workspace control whose shared state can carry all specialist states.

    The constrained workspace has a 32-D shared state receiving four 32-D
    specialist states.  Here the shared state is 128-D, equal to their combined
    capacity.  Specialists, update topology, processing steps, tasks, and the
    approximate trainable-parameter budget are otherwise retained.
    """

    architecture_name = "unlimited_shared_state"

    def __init__(self, hidden: int, state_dimensions: int = 128, steps: int = 6) -> None:
        if state_dimensions % 4:
            raise ValueError("unlimited shared state must split evenly across four specialists")
        super().__init__(state_dimensions, steps)
        specialist_dimensions = state_dimensions // 4
        self.specialist_initializers = nn.ModuleList(
            [nn.Linear(state_dimensions, specialist_dimensions) for _ in range(4)]
        )
        self.specialist_updates = nn.ModuleList(
            [
                ResidualMLP(
                    state_dimensions + specialist_dimensions,
                    specialist_dimensions,
                    hidden,
                )
                for _ in range(4)
            ]
        )
        self.workspace_update = ResidualMLP(
            2 * state_dimensions,
            state_dimensions,
            hidden,
        )

    def initialize(self, drive: Tensor) -> tuple[Tensor, list[Tensor]]:
        specialists = [torch.tanh(initializer(drive)) for initializer in self.specialist_initializers]
        return drive, specialists

    def advance(
        self,
        state: Tensor,
        drive: Tensor,
        context: list[Tensor],
        step: int,
    ) -> tuple[Tensor, list[Tensor]]:
        del drive, step
        specialists = [
            specialist + 0.1 * update(torch.cat([specialist, state], dim=-1))
            for specialist, update in zip(context, self.specialist_updates, strict=True)
        ]
        workspace = state + 0.1 * self.workspace_update(torch.cat([*specialists, state], dim=-1))
        return workspace, specialists


class PrivateModulesSystem(VisionSystem):
    architecture_name = "private_modules"

    def __init__(self, hidden: int, state_dimensions: int = 32, steps: int = 6) -> None:
        if state_dimensions % 4:
            raise ValueError("private state must split evenly into four modules")
        super().__init__(state_dimensions, steps)
        private_dimensions = state_dimensions // 4
        self.private_initializers = nn.ModuleList(
            [nn.Linear(state_dimensions, private_dimensions) for _ in range(4)]
        )
        self.private_updates = nn.ModuleList(
            [
                ResidualMLP(
                    state_dimensions + private_dimensions,
                    private_dimensions,
                    hidden,
                )
                for _ in range(4)
            ]
        )

    def initialize(self, drive: Tensor) -> tuple[Tensor, None]:
        private = [torch.tanh(initializer(drive)) for initializer in self.private_initializers]
        return torch.cat(private, dim=-1), None

    def advance(
        self,
        state: Tensor,
        drive: Tensor,
        context: None,
        step: int,
    ) -> tuple[Tensor, None]:
        del step
        private = state.chunk(4, dim=-1)
        updated = [
            value + 0.1 * update(torch.cat([value, drive], dim=-1))
            for value, update in zip(private, self.private_updates, strict=True)
        ]
        return torch.cat(updated, dim=-1), None


def build_architecture(
    name: str,
    *,
    hidden: int | None = None,
    state_dimensions: int | None = None,
    steps: int = 6,
    parameter_target: int = 2_000_000,
    tolerance_fraction: float = 0.10,
) -> VisionSystem:
    width = hidden or DEFAULT_HIDDEN_WIDTHS[name]
    classes: Mapping[str, type[VisionSystem]] = {
        "feedforward": FeedforwardSystem,
        "recurrent": RecurrentSystem,
        "shared_workspace": SharedWorkspaceSystem,
        "private_modules": PrivateModulesSystem,
        "unlimited_shared_state": UnlimitedSharedStateSystem,
    }
    if name not in classes:
        raise ValueError(f"unknown architecture {name!r}")
    state_dimensions = state_dimensions or DEFAULT_STATE_DIMENSIONS[name]
    model = classes[name](width, state_dimensions, steps)
    parameters = count_parameters(model)
    relative_error = abs(parameters - parameter_target) / parameter_target
    if relative_error > tolerance_fraction:
        raise ValueError(
            f"{name} has {parameters:,} parameters, outside "
            f"{tolerance_fraction:.1%} of {parameter_target:,}"
        )
    return model
