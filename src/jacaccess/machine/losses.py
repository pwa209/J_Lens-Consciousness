"""Masked multi-head loss for the controlled machine task."""

from __future__ import annotations

from collections.abc import Mapping

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("jacaccess.machine.losses requires PyTorch") from exc


def multihead_cross_entropy(
    logits: Mapping[str, Tensor],
    labels: Mapping[str, Tensor],
    valid_masks: Mapping[str, Tensor],
    task_cues: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Score perceptual heads at all steps and delayed action at the final step."""

    losses: dict[str, Tensor] = {}
    head_names = tuple(logits)
    for head_index, name in enumerate(head_names):
        values = logits[name]
        target = labels[name]
        mask = valid_masks[name].bool()
        if name == "delayed_action":
            selected = values[:, -1]
            base_loss = torch.nn.functional.cross_entropy(selected[mask], target[mask])
        else:
            repeated_target = target[:, None].expand(-1, values.shape[1])
            repeated_mask = mask[:, None].expand_as(repeated_target)
            base_loss = torch.nn.functional.cross_entropy(
                values[repeated_mask],
                repeated_target[repeated_mask],
            )
        cue_weight = 1.0 + task_cues[:, head_index].mean()
        losses[name] = cue_weight * base_loss
    return torch.stack(tuple(losses.values())).mean(), losses

