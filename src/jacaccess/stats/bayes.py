"""Directional one-sample JZS Bayes factor without an R runtime dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats


def directional_jzs_bayes_factor(
    values: np.ndarray, *, rscale: float = 0.5
) -> float:
    """Return BF+0 for a positive standardized effect with a half-Cauchy prior.

    Conditional on standardized effect ``delta``, the one-sample t statistic has
    a noncentral-t distribution with noncentrality ``delta * sqrt(n)``.  The
    JZS alternative places a Cauchy prior on delta; restricting and
    renormalizing it to delta > 0 matches the directional alternative used by
    ``BayesFactor::ttestBF(nullInterval=c(0, Inf), rscale=...)``.
    """

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        raise ValueError("at least two finite contrasts are required")
    if not np.isfinite(rscale) or rscale <= 0:
        raise ValueError("rscale must be positive and finite")
    standard_deviation = float(np.std(finite, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("directional Bayes factor requires nonzero sample variance")
    n = len(finite)
    degrees_of_freedom = n - 1
    t_value = float(np.mean(finite) / (standard_deviation / np.sqrt(n)))
    log_null = float(stats.t.logpdf(t_value, degrees_of_freedom))

    def integrand(delta: float) -> float:
        log_likelihood_ratio = (
            stats.nct.logpdf(
                t_value, degrees_of_freedom, delta * np.sqrt(n)
            )
            - log_null
        )
        if not np.isfinite(log_likelihood_ratio):
            return 0.0
        prior = 2.0 / (np.pi * rscale * (1.0 + (delta / rscale) ** 2))
        return float(np.exp(log_likelihood_ratio) * prior)

    value, error = integrate.quad(
        integrand, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=300
    )
    if not np.isfinite(value) or value <= 0 or not np.isfinite(error):
        raise ValueError("directional JZS Bayes-factor integration failed")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rscale", type=float, default=0.5)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    if "contrast" not in table:
        raise ValueError("participant contrast table lacks a contrast column")
    values = table["contrast"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    result = {
        "participants": int(len(finite)),
        "mean_contrast": float(np.mean(finite)),
        "directional_bayes_factor": directional_jzs_bayes_factor(
            finite, rscale=args.rscale
        ),
        "rscale": float(args.rscale),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
