"""Benchmark functions used to test the optimizer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .validation import as_1d_finite_vector

Array = NDArray[np.float64]
ObjectiveFn = Callable[[Array], float]


def sphere(x: Array) -> float:
    """f(x) = sum(x^2). Minimum at the origin, f(0) = 0. Works in any dimension."""
    x = as_1d_finite_vector(x, name="x")
    return float(np.sum(x**2))


def rosenbrock(x: Array) -> float:
    """Classic banana valley. Needs at least 2 dimensions. Minimum f(1, ..., 1) = 0."""
    x = as_1d_finite_vector(x, name="x")
    if x.size < 2:
        raise ValueError(
            f"Rosenbrock requires at least 2 dimensions, got length {x.size}."
        )
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    )


def rastrigin(x: Array) -> float:
    """Many local minima. Minimum f(0, ..., 0) = 0. Works in any dimension."""
    x = as_1d_finite_vector(x, name="x")
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


BENCHMARK_FUNCTIONS: dict[str, ObjectiveFn] = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
}
