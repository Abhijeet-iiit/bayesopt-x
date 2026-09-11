"""Shared paths, defaults, and benchmark definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.black_box import rastrigin, rosenbrock, sphere

Array = NDArray[np.float64]
ObjectiveFn = Callable[[Array], float]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

DEFAULT_SEEDS = [42, 7, 21, 100, 2026]
DEFAULT_N_INITIAL = 5
DEFAULT_N_ITERATIONS = 20

BENCHMARK_CHOICES = ("sphere", "rosenbrock", "rastrigin")
ACQUISITION_CHOICES = ("ei", "ucb")


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    function: ObjectiveFn
    bounds: Array


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "sphere": BenchmarkSpec(
        name="Sphere",
        function=sphere,
        bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
    ),
    "rosenbrock": BenchmarkSpec(
        name="Rosenbrock",
        function=rosenbrock,
        bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
    ),
    "rastrigin": BenchmarkSpec(
        name="Rastrigin",
        function=rastrigin,
        bounds=np.array([[-5.12, 5.12], [-5.12, 5.12]]),
    ),
}


def resolve_benchmarks(name: str) -> list[BenchmarkSpec]:
    key = name.lower()
    if key == "all":
        return list(BENCHMARKS.values())
    if key not in BENCHMARKS:
        valid = ", ".join(BENCHMARKS)
        raise ValueError(f"Unknown benchmark {name!r}. Choose from: {valid}, all.")
    return [BENCHMARKS[key]]


def total_evaluations(n_initial: int, n_iterations: int) -> int:
    return n_initial + n_iterations
