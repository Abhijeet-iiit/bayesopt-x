"""Gaussian Process surrogate.

Gives a predicted mean and uncertainty for any candidate. Inputs and
outputs are standardized so the Matern kernel stays on a sensible scale.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

from .validation import as_finite_array, validate_positive_int, validate_random_state

Array = NDArray[np.float64]

logger = logging.getLogger(__name__)


def _optimize_hyperparameters(
    obj_func: Callable[..., tuple[float, np.ndarray]],
    initial_theta: np.ndarray,
    bounds: object,
) -> tuple[np.ndarray, float]:
    """Fit kernel hyperparameters with a slightly larger L-BFGS-B budget."""
    result = minimize(
        obj_func,
        np.asarray(initial_theta, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 200},
    )
    if not result.success:
        logger.debug("GP hyperparameter fit did not converge: %s", result.message)
    return result.x, result.fun


class GaussianProcessSurrogate:
    """Matern GP with standardized X and y."""

    def __init__(
        self,
        random_state: int = 42,
        n_restarts_optimizer: int = 5,
        alpha: float = 1e-6,
    ) -> None:
        self.random_state = validate_random_state(random_state)
        self.n_restarts_optimizer = validate_positive_int(
            n_restarts_optimizer,
            name="n_restarts_optimizer",
            minimum=0,
        )
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(f"alpha must be a positive finite number, got {alpha}.")
        self.alpha = float(alpha)

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # After scaling, length-scales around 1 are typical.
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e4)) * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-3, 1e2),
            nu=2.5,
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            optimizer=_optimize_hyperparameters,
        )
        self.is_fitted = False
        self.n_features_: int | None = None

    def fit(self, X: Array, y: Array) -> None:
        X_arr, y_arr = self._validate_training_data(X, y)
        logger.debug(
            "Fitting GP on %d samples, %d features.",
            X_arr.shape[0],
            X_arr.shape[1],
        )
        X_scaled = self.x_scaler.fit_transform(X_arr)
        y_scaled = self.y_scaler.fit_transform(y_arr.reshape(-1, 1)).ravel()
        self.model.fit(X_scaled, y_scaled)
        self.n_features_ = X_arr.shape[1]
        self.is_fitted = True

    def predict(
        self,
        X: Array,
        return_std: bool = True,
    ) -> tuple[Array, Array | None]:
        if not self.is_fitted or self.n_features_ is None:
            raise RuntimeError("The Gaussian Process must be fitted before calling predict().")

        X_scaled = self.x_scaler.transform(self._validate_predict_X(X))

        if return_std:
            mean_scaled, std_scaled = self.model.predict(X_scaled, return_std=True)
            mean = self._unscale_mean(mean_scaled)
            std = np.asarray(std_scaled, dtype=float) * float(self.y_scaler.scale_[0])
            return mean, np.maximum(std, 0.0)

        mean_scaled = self.model.predict(X_scaled, return_std=False)
        return self._unscale_mean(mean_scaled), None

    def _unscale_mean(self, mean_scaled: Array) -> Array:
        return self.y_scaler.inverse_transform(
            np.asarray(mean_scaled, dtype=float).reshape(-1, 1)
        ).ravel()

    def _validate_training_data(self, X: object, y: object) -> tuple[Array, Array]:
        X_arr = as_finite_array(X, name="X")
        y_arr = as_finite_array(y, name="y").reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {X_arr.shape}."
            )
        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X_arr.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if y_arr.size != X_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X_arr.shape[0]} vs {y_arr.size}."
            )
        return X_arr, y_arr

    def _validate_predict_X(self, X: object) -> Array:
        assert self.n_features_ is not None
        X_arr = as_finite_array(X, name="X")
        if X_arr.ndim == 1:
            if X_arr.size != self.n_features_:
                raise ValueError(
                    f"1-D input must have length {self.n_features_}, got {X_arr.size}."
                )
            return X_arr.reshape(1, -1)
        if X_arr.ndim == 2:
            if X_arr.shape[0] == 0:
                raise ValueError("X must contain at least one sample.")
            if X_arr.shape[1] != self.n_features_:
                raise ValueError(
                    f"X must have {self.n_features_} features, got {X_arr.shape[1]}."
                )
            return X_arr
        raise ValueError(f"X must be 1-D or 2-D, got shape {X_arr.shape}.")
