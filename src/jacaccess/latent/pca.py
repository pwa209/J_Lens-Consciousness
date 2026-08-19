"""Training-fold PCA and whitening for EEG latent states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class PCAWhitening:
    mean: FloatArray
    components: FloatArray
    eigenvalues: FloatArray
    epsilon: float = 1e-7

    def transform(self, values: FloatArray) -> FloatArray:
        x = np.asarray(values)
        if x.shape[-1] != self.mean.shape[0]:
            raise ValueError("input feature dimension does not match fitted PCA")
        projected = (x - self.mean) @ self.components.T
        return projected / np.sqrt(self.eigenvalues + self.epsilon)

    def inverse_transform(self, latent: FloatArray) -> FloatArray:
        z = np.asarray(latent)
        if z.shape[-1] != self.components.shape[0]:
            raise ValueError("latent dimension does not match fitted PCA")
        unwhitened = z * np.sqrt(self.eigenvalues + self.epsilon)
        return unwhitened @ self.components + self.mean


def fit_pca_whitening(
    training_values: FloatArray,
    components: int = 32,
    epsilon: float = 1e-7,
) -> PCAWhitening:
    """Fit unit-invariant PCA whitening using training samples only.

    Leading dimensions are treated as sample axes and flattened. The final
    dimension is the sensor/feature axis. ``epsilon`` is a relative fraction
    of the leading eigenvalue, so volts and microvolts produce the same fit.
    """

    values = np.asarray(training_values)
    if values.ndim < 2:
        raise ValueError("training_values needs sample and feature dimensions")
    flattened = values.reshape(-1, values.shape[-1]).astype(np.float64, copy=False)
    if components < 1 or components > min(flattened.shape):
        raise ValueError("invalid number of PCA components")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be a relative fraction between zero and one")
    if not np.isfinite(flattened).all():
        raise ValueError("training_values contains non-finite values")
    mean = flattened.mean(axis=0)
    centered = flattened - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    denominator = max(flattened.shape[0] - 1, 1)
    eigenvalues = singular_values[:components] ** 2 / denominator
    leading_eigenvalue = float(singular_values[0] ** 2 / denominator)
    whitening_epsilon = epsilon * leading_eigenvalue
    if leading_eigenvalue <= 0 or np.any(eigenvalues <= whitening_epsilon):
        raise ValueError("training data have insufficient rank for requested PCA")
    return PCAWhitening(
        mean=mean,
        components=vh[:components],
        eigenvalues=eigenvalues,
        epsilon=whitening_epsilon,
    )
