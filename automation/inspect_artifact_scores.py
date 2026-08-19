"""Summarize outcome-blind epoch artifact scores for pipeline QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    loaded = np.load(args.path)
    payload = {}
    for name in loaded.files:
        values = np.asarray(loaded[name])
        if values.dtype == bool:
            payload[name] = {"true": int(values.sum()), "total": int(values.size)}
        else:
            finite = values[np.isfinite(values)]
            payload[name] = {
                "minimum": float(np.min(finite)),
                "median": float(np.median(finite)),
                "p75": float(np.quantile(finite, 0.75)),
                "p90": float(np.quantile(finite, 0.90)),
                "p95": float(np.quantile(finite, 0.95)),
                "maximum": float(np.max(finite)),
            }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

