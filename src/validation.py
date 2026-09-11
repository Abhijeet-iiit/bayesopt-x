"""Shared input checks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def require_callable(value: object, name: str = "objective") -> Callable[..., object]:
    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value).__name__}.")
    return value


def as_1d_finite_vector(x: object, name: str = "x") -> Array:
    """Turn x into a non-empty 1-D float vector. Scalars become length 1."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def as_finite_array(x: object, name: str = "array") -> Array:
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def validate_bounds(bounds: object) -> Array:
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"bounds must have shape (n_dimensions, 2), got {arr.shape}."
        )
    if arr.shape[0] < 1:
        raise ValueError("bounds must contain at least one dimension.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds contains NaN or infinite values.")
    if np.any(arr[:, 0] >= arr[:, 1]):
        raise ValueError(
            "Each lower bound must be strictly smaller than its upper bound."
        )
    return arr


def validate_positive_int(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    ivalue = int(value)
    if ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {ivalue}.")
    return ivalue


def validate_finite_scalar(value: object, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite real number, got {value!r}."
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number, got {result}.")
    return result


def validate_random_state(random_state: object) -> int:
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError(
            f"random_state must be an integer, got {type(random_state).__name__}."
        )
    return int(random_state)


def clip_to_bounds(x: Array, bounds: Array) -> Array:
    return np.clip(x, bounds[:, 0], bounds[:, 1])
