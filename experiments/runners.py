"""Run Bayesian Optimization or Random Search for the experiment scripts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from src.optimizer import BayesianOptimizer
from src.random_search import RandomSearch

Array = NDArray[np.float64]
ObjectiveFn = Callable[[Array], float]


def run_bayesian_optimization(
    objective: ObjectiveFn,
    bounds: Array,
    seed: int,
    acquisition: str = "ei",
    n_initial: int = 5,
    n_iterations: int = 20,
    minimize: bool = True,
) -> tuple[Array, float]:
    optimizer = BayesianOptimizer(
        objective=objective,
        bounds=bounds,
        acquisition=acquisition,
        minimize=minimize,
        random_state=seed,
    )
    return optimizer.optimize(n_initial=n_initial, n_iterations=n_iterations)


def run_random_search(
    objective: ObjectiveFn,
    bounds: Array,
    seed: int,
    n_iterations: int,
    minimize: bool = True,
) -> tuple[Array, float]:
    optimizer = RandomSearch(
        objective=objective,
        bounds=bounds,
        minimize=minimize,
        random_state=seed,
    )
    return optimizer.optimize(n_iterations=n_iterations)
