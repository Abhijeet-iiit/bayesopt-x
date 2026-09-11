"""Acquisition functions (written for maximization).

If the user asked to minimize, the optimizer negates y before the GP sees it,
so these formulas stay the same.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from .validation import as_finite_array, validate_finite_scalar

Array = NDArray[np.float64]

_ZERO_STD = 1e-12


def _prepare_mean_std(mean: Array, std: Array) -> tuple[Array, Array]:
    mean_arr = as_finite_array(mean, name="mean")
    std_arr = as_finite_array(std, name="std")
    if mean_arr.shape != std_arr.shape:
        raise ValueError(
            f"mean and std must have the same shape, got {mean_arr.shape} and {std_arr.shape}."
        )
    if np.any(std_arr < -1e-12):
        raise ValueError("std must be non-negative.")
    return mean_arr, np.maximum(std_arr, 0.0)


def upper_confidence_bound(
    mean: Array,
    std: Array,
    beta: float = 2.0,
) -> Array:
    """UCB(x) = mean(x) + beta * std(x). Larger beta explores more."""
    mean_arr, std_arr = _prepare_mean_std(mean, std)
    beta_value = validate_finite_scalar(beta, name="beta")
    if beta_value < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta_value}.")

    scores = mean_arr + beta_value * std_arr
    if not np.all(np.isfinite(scores)):
        raise ValueError("UCB produced a non-finite score.")
    return scores


def expected_improvement(
    mean: Array,
    std: Array,
    best_value: float,
    xi: float = 0.01,
) -> Array:
    """How much a point is expected to beat the current best.

    EI is 0 when std is numerically zero. Always non-negative.
    """
    mean_arr, std_arr = _prepare_mean_std(mean, std)
    best = validate_finite_scalar(best_value, name="best_value")
    xi_value = validate_finite_scalar(xi, name="xi")
    if xi_value < 0.0:
        raise ValueError(f"xi must be >= 0, got {xi_value}.")

    improvement = mean_arr - best - xi_value
    ei = np.zeros_like(mean_arr, dtype=float)

    explore = std_arr > _ZERO_STD
    if np.any(explore):
        sigma = std_arr[explore]
        z = improvement[explore] / sigma
        ei[explore] = improvement[explore] * norm.cdf(z) + sigma * norm.pdf(z)

    ei = np.maximum(ei, 0.0)
    if not np.all(np.isfinite(ei)):
        raise ValueError("Expected Improvement produced a non-finite score.")
    return ei
