"""Gaussian CRPS for heteroscedastic predictions.

Closed-form CRPS for a Gaussian predictive distribution N(mu, sigma) and
realized observation y:

    CRPS(N(mu, sigma), y) = sigma * [ z * (2*Phi(z) - 1)
                                       + 2*phi(z)
                                       - 1/sqrt(pi) ]
    where z = (y - mu) / sigma.

A proper scoring rule (lower is better). Combines bulk fit, calibration, and
sharpness in one number; not gameable by inflating sigma. See Gneiting &
Raftery (2007) "Strictly Proper Scoring Rules, Prediction, and Estimation."
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def gaussian_crps(y, mu, sigma):
    """Mean Gaussian CRPS over a 1-D batch.

    Args:
        y, mu, sigma: 1-D arrays of equal length. sigma must be > 0.
    Returns:
        float — mean CRPS across the batch.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    if not mask.any():
        return float("nan")
    y, mu, sigma = y[mask], mu[mask], sigma[mask]
    z = (y - mu) / sigma
    return float(
        np.mean(
            sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        )
    )
