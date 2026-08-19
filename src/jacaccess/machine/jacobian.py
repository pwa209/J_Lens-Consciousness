"""Exact future-logit Jacobians for the common machine integration state."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("jacaccess.machine.jacobian requires PyTorch") from exc


def exact_future_logit_jacobians(
    model: object,
    images: Tensor,
    task_cues: Tensor,
    steps: tuple[int, ...] = (2, 3, 4),
) -> dict[int, Tensor]:
    """Differentiate current/future logits with respect to each selected state.

    Step indices are zero based. Shared-workspace specialist states are treated
    as fixed local context at the differentiation point; subsequent specialist
    updates remain functions of the perturbed workspace.
    """

    drive = model.initial_drive(images, task_cues)
    states, contexts = model.trace_from_drive(drive)
    output: dict[int, Tensor] = {}
    for step in steps:
        if not 0 <= step < model.steps:
            raise ValueError(f"step {step} is outside the model trace")
        selected_states = states[:, step].detach()
        selected_drive = drive.detach()
        raw_context = contexts[step]
        if isinstance(raw_context, list):
            context_tensors = tuple(value.detach() for value in raw_context)

            def future(value: Tensor, sample_drive: Tensor, *context: Tensor) -> Tensor:
                return model.future_logit_vector(
                    value,
                    sample_drive,
                    step,
                    list(context),
                )

            derivative = torch.func.jacrev(future, argnums=0)
            output[step] = torch.func.vmap(derivative)(
                selected_states,
                selected_drive,
                *context_tensors,
            )
        else:
            derivative = torch.func.jacrev(
                lambda value, sample_drive: model.future_logit_vector(
                    value,
                    sample_drive,
                    step,
                    None,
                ),
                argnums=0,
            )
            output[step] = torch.func.vmap(derivative)(selected_states, selected_drive)
    return output
