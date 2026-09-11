"""Bayesian Optimization loop.

Fit a GP, maximize an acquisition function inside the bounds, evaluate
the true objective, repeat. Acquisition functions assume maximization, so
when minimize=True we train the GP on -y.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .acquisition import expected_improvement, upper_confidence_bound
from .gp_model import GaussianProcessSurrogate
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

_VALID_ACQUISITIONS = frozenset({"ei", "ucb"})


class BayesianOptimizer:
    """Bayesian optimizer for box-constrained black-box functions."""

    def __init__(
        self,
        objective: ObjectiveFn,
        bounds: Array,
        acquisition: str = "ei",
        minimize: bool = True,
        random_state: int = 42,
        xi: float = 0.01,
        beta: float = 2.0,
        n_candidates: int = 1000,
        n_acquisition_restarts: int = 8,
    ) -> None:
        self.objective = require_callable(objective, name="objective")
        self.bounds = validate_bounds(bounds)
        if not isinstance(acquisition, str):
            raise TypeError(
                f"acquisition must be a string, got {type(acquisition).__name__}."
            )
        self.acquisition = acquisition.lower()
        if self.acquisition not in _VALID_ACQUISITIONS:
            valid = ", ".join(sorted(_VALID_ACQUISITIONS))
            raise ValueError(
                f"acquisition must be one of {{{valid}}}, got {acquisition!r}."
            )
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        self.minimize = minimize
        self.random_state = validate_random_state(random_state)
        self.xi = validate_finite_scalar(xi, name="xi")
        if self.xi < 0.0:
            raise ValueError(f"xi must be >= 0, got {self.xi}.")
        self.beta = validate_finite_scalar(beta, name="beta")
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}.")
        self.n_candidates = validate_positive_int(n_candidates, name="n_candidates")
        self.n_acquisition_restarts = validate_positive_int(
            n_acquisition_restarts,
            name="n_acquisition_restarts",
            minimum=0,
        )

        self.dimension = int(self.bounds.shape[0])
        self.rng = np.random.default_rng(self.random_state)
        self.gp = GaussianProcessSurrogate(random_state=self.random_state)

        self.X: list[Array] = []
        self.y: list[float] = []
        self.history: list[float] = []
        self.best_x: Array | None = None
        self.best_y: float | None = None

        logger.info(
            "BayesianOptimizer dim=%d acquisition=%s minimize=%s seed=%d",
            self.dimension,
            self.acquisition,
            self.minimize,
            self.random_state,
        )

    def reset(self) -> None:
        """Drop previous observations so a new run starts clean."""
        self.X = []
        self.y = []
        self.history = []
        self.best_x = None
        self.best_y = None
        self.gp = GaussianProcessSurrogate(random_state=self.random_state)
        self.rng = np.random.default_rng(self.random_state)

    def _sample_random_points(self, n_points: int) -> Array:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return self.rng.uniform(lower, upper, size=(n_points, self.dimension))

    def _evaluate_one(self, x: Array) -> float:
        x_vec = as_1d_finite_vector(x, name="x")
        if x_vec.size != self.dimension:
            raise ValueError(
                f"Objective input must have length {self.dimension}, got {x_vec.size}."
            )
        x_vec = clip_to_bounds(x_vec, self.bounds)
        return validate_finite_scalar(self.objective(x_vec), name="objective value")

    def _evaluate(self, X: Array) -> Array:
        return np.array([self._evaluate_one(x) for x in X], dtype=float)

    def _transformed_values(self) -> Array:
        y = np.asarray(self.y, dtype=float)
        return -y if self.minimize else y

    def _acquisition_scores(self, candidates: Array) -> Array:
        mean, std = self.gp.predict(candidates)
        assert std is not None
        if self.acquisition == "ucb":
            return upper_confidence_bound(mean, std, beta=self.beta)
        best_value = float(np.max(self._transformed_values()))
        return expected_improvement(mean, std, best_value=best_value, xi=self.xi)

    def _select_next_point(self) -> Array:
        """Random multi-start, then L-BFGS-B inside the box."""
        candidates = self._sample_random_points(self.n_candidates)
        scores = self._acquisition_scores(candidates)
        best_index = int(np.argmax(scores))
        best_x = candidates[best_index].copy()
        best_score = float(scores[best_index])

        n_restarts = min(self.n_acquisition_restarts, self.n_candidates)
        if n_restarts == 0:
            return clip_to_bounds(best_x, self.bounds)

        start_indices = np.argsort(scores)[-n_restarts:]
        scipy_bounds = [(float(lo), float(hi)) for lo, hi in self.bounds]

        def negative_acquisition(x: Array) -> float:
            value = float(self._acquisition_scores(np.atleast_2d(x))[0])
            return -value if np.isfinite(value) else np.inf

        for index in start_indices:
            result = minimize(
                negative_acquisition,
                x0=candidates[index],
                method="L-BFGS-B",
                bounds=scipy_bounds,
            )
            if not np.isfinite(result.fun):
                continue
            x_opt = clip_to_bounds(np.asarray(result.x, dtype=float), self.bounds)
            score = float(self._acquisition_scores(np.atleast_2d(x_opt))[0])
            if score > best_score:
                best_score = score
                best_x = x_opt

        return clip_to_bounds(best_x, self.bounds)

    def _update_best(self) -> None:
        if not self.y:
            self.best_x = None
            self.best_y = None
            return
        best_index = int(np.argmin(self.y) if self.minimize else np.argmax(self.y))
        self.best_x = np.asarray(self.X[best_index], dtype=float)
        self.best_y = float(self.y[best_index])
        self.history.append(self.best_y)

    def initialize(self, n_initial: int = 5) -> None:
        n_initial = validate_positive_int(n_initial, name="n_initial")
        if self.X:
            raise RuntimeError(
                "Optimizer already has observations. Call reset() first, "
                "or use optimize() which starts a fresh run."
            )

        X_initial = self._sample_random_points(n_initial)
        y_initial = self._evaluate(X_initial)
        for x, y in zip(X_initial, y_initial, strict=True):
            self.X.append(np.asarray(x, dtype=float))
            self.y.append(float(y))
            self._update_best()

        logger.info("Initialized with %d points. Best y=%.6g", n_initial, self.best_y)

    def step(self) -> None:
        if not self.X:
            raise RuntimeError("Call initialize() before step().")

        self.gp.fit(np.asarray(self.X, dtype=float), self._transformed_values())
        next_x = self._select_next_point()
        next_y = self._evaluate_one(next_x)
        self.X.append(np.asarray(next_x, dtype=float))
        self.y.append(float(next_y))
        self._update_best()
        logger.debug("step %d: y=%.6g best_y=%.6g", len(self.y), next_y, self.best_y)

    def optimize(
        self,
        n_initial: int = 5,
        n_iterations: int = 20,
    ) -> tuple[Array, float]:
        """Run a full optimization. Always resets first (same seed => same result)."""
        n_initial = validate_positive_int(n_initial, name="n_initial")
        n_iterations = validate_positive_int(n_iterations, name="n_iterations")

        self.reset()
        self.initialize(n_initial)
        logger.info(
            "Starting BO: n_initial=%d n_iterations=%d", n_initial, n_iterations
        )
        for iteration in range(1, n_iterations + 1):
            self.step()
            logger.debug("iter %d/%d best_y=%.6g", iteration, n_iterations, self.best_y)

        assert self.best_x is not None
        assert self.best_y is not None
        logger.info("Done after %d evaluations. Best y=%.6g", len(self.y), self.best_y)
        return self.best_x, self.best_y
