"""Small GPU/CPU smoke run for one architecture and seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from jacaccess.machine.architectures import build_architecture, count_parameters
    from jacaccess.machine.stimuli import generate_stimulus_batch

    torch.manual_seed(args.seed)
    batch = generate_stimulus_batch(np.arange(4), seed=args.seed)
    model = build_architecture(args.architecture)
    images = torch.from_numpy(batch.images)
    cues = torch.from_numpy(batch.task_cues)
    states, logits = model(images, cues)
    summary = {
        "architecture": args.architecture,
        "seed": args.seed,
        "parameters": count_parameters(model),
        "state_shape": list(states.shape),
        "head_shapes": {name: list(values.shape) for name, values in logits.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

