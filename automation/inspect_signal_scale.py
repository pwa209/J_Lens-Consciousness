"""Print bounded scale diagnostics for a NumPy EEG array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    values = np.load(args.path, mmap_mode="r")
    payload = {
        "shape": list(values.shape),
        "minimum": float(np.nanmin(values)),
        "maximum": float(np.nanmax(values)),
        "median_trial_channel_peak_to_peak": float(np.nanmedian(np.ptp(values, axis=-1))),
        "median_absolute": float(np.nanmedian(np.abs(values))),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

