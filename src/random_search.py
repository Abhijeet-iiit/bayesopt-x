"""Uniform random search, used as a baseline."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .validation import (
    as_1d_finite_vector,
    clip_to_bounds,
    require_callable,
    validate_bounds,
    validate_finite_scalar,
    validate_positive_int,
    validate_random_state,
)

Array = NDArray[np.float64]
ObjectiveFn = Callable[[Array], float]

logger = logging.getLogger(__name__)


class RandomSearch:
    """Sample the box uniformly and keep the best point."""

    def __init__(
        self,
        objective: ObjectiveFn,
        bounds: Array,
        minimize: bool = True,
        random_state: int = 42,
    ) -> None:
        self.objective = require_callable(objective, name="objective")
        self.bounds = validate_bounds(bounds)
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")
        self.minimize = minimize
        self.random_state = validate_random_state(random_state)
        self.dimension = int(self.bounds.shape[0])
        self.rng = np.random.default_rng(self.random_state)

        self.X: list[Array] = []
        self.y: list[float] = []
        self.history: list[float] = []
        self.best_x: Array | None = None
        self.best_y: float | None = None

        logger.info(
            "RandomSearch dim=%d minimize=%s seed=%d",
            self.dimension,
            self.minimize,
            self.random_state,
        )

    def reset(self) -> None:
        self.X = []
        self.y = []
        self.history = []
        self.best_x = None
        self.best_y = None
        self.rng = np.random.default_rng(self.random_state)

    def _sample_point(self) -> Array:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return clip_to_bounds(self.rng.uniform(lower, upper), self.bounds)

    def _evaluate_one(self, x: Array) -> float:
        x_vec = as_1d_finite_vector(x, name="x")
        if x_vec.size != self.dimension:
            raise ValueError(
                f"Objective input must have length {self.dimension}, got {x_vec.size}."
            )
        return validate_finite_scalar(self.objective(x_vec), name="objective value")

    def _update_best(self) -> None:
        best_index = int(np.argmin(self.y) if self.minimize else np.argmax(self.y))
        self.best_x = np.asarray(self.X[best_index], dtype=float)
        self.best_y = float(self.y[best_index])
        self.history.append(self.best_y)

    def optimize(self, n_iterations: int = 25) -> tuple[Array, float]:
        """Run from a clean state. Same seed => same result."""
        n_iterations = validate_positive_int(n_iterations, name="n_iterations")
        self.reset()
        logger.info("Starting Random Search: n_iterations=%d", n_iterations)

        for _ in range(n_iterations):
            x = self._sample_point()
            y = self._evaluate_one(x)
            self.X.append(np.asarray(x, dtype=float))
            self.y.append(y)
            self._update_best()

        assert self.best_x is not None
        assert self.best_y is not None
        logger.info("Done after %d evaluations. Best y=%.6g", n_iterations, self.best_y)
        return self.best_x, self.best_y
