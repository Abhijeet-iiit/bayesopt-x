"""Tests for benchmark objective functions."""

import numpy as np
import pytest

from src.black_box import BENCHMARK_FUNCTIONS, rastrigin, rosenbrock, sphere


def test_sphere_global_minimum() -> None:
    x = np.array([0.0, 0.0])
    assert sphere(x) == 0.0


def test_sphere_known_value() -> None:
    x = np.array([3.0, 4.0])
    assert sphere(x) == 25.0


def test_sphere_multidimensional() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert sphere(x) == 30.0


def test_rosenbrock_global_minimum() -> None:
    x = np.array([1.0, 1.0])
    assert rosenbrock(x) == 0.0


def test_rosenbrock_three_dimensions() -> None:
    x = np.array([1.0, 1.0, 1.0])
    assert rosenbrock(x) == 0.0


def test_rosenbrock_rejects_scalar_dimension() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        rosenbrock(np.array([1.0]))


def test_rastrigin_global_minimum() -> None:
    x = np.array([0.0, 0.0])
    assert rastrigin(x) == 0.0


def test_rastrigin_known_value() -> None:
    x = np.array([0.0])
    assert rastrigin(x) == 0.0


def test_functions_reject_non_finite_input() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        sphere(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError, match="NaN or infinite"):
        rastrigin(np.array([np.inf]))


def test_functions_reject_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D array"):
        sphere(np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_benchmark_registry() -> None:
    assert set(BENCHMARK_FUNCTIONS) == {"sphere", "rosenbrock", "rastrigin"}
    assert BENCHMARK_FUNCTIONS["sphere"] is sphere
