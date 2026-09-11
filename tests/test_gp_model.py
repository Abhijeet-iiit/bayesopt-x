"""Tests for the Gaussian Process surrogate."""

import numpy as np
import pytest

from src.gp_model import GaussianProcessSurrogate


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([1.0, 0.0, 1.0])
    return X, y


def test_gp_can_fit_and_predict() -> None:
    X, y = _toy_data()
    gp = GaussianProcessSurrogate(random_state=42, n_restarts_optimizer=1)
    gp.fit(X, y)
    mean, std = gp.predict(np.array([[0.0], [0.5]]))
    assert mean.shape == (2,)
    assert std is not None
    assert std.shape == (2,)
    assert np.all(std >= 0.0)


def test_gp_predicts_training_point_reasonably() -> None:
    X, y = _toy_data()
    gp = GaussianProcessSurrogate(random_state=42, n_restarts_optimizer=1)
    gp.fit(X, y)
    mean, _ = gp.predict(np.array([[0.0]]))
    assert abs(mean[0]) < 0.2


def test_gp_unfitted_prediction_raises() -> None:
    gp = GaussianProcessSurrogate(random_state=42)
    with pytest.raises(RuntimeError, match="fitted"):
        gp.predict(np.array([[0.0]]))


def test_gp_rejects_1d_training_X() -> None:
    gp = GaussianProcessSurrogate(random_state=42)
    with pytest.raises(ValueError, match="2-D"):
        gp.fit(np.array([0.0, 1.0]), np.array([1.0, 2.0]))


def test_gp_rejects_mismatched_lengths() -> None:
    gp = GaussianProcessSurrogate(random_state=42)
    with pytest.raises(ValueError, match="length mismatch"):
        gp.fit(np.array([[0.0], [1.0]]), np.array([1.0]))


def test_gp_rejects_non_finite_training_data() -> None:
    gp = GaussianProcessSurrogate(random_state=42)
    with pytest.raises(ValueError, match="NaN or infinite"):
        gp.fit(np.array([[np.nan], [1.0]]), np.array([1.0, 2.0]))


def test_gp_rejects_wrong_feature_count() -> None:
    gp = GaussianProcessSurrogate(random_state=42, n_restarts_optimizer=0)
    gp.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="features"):
        gp.predict(np.array([[0.0, 1.0]]))


def test_gp_predict_without_std() -> None:
    X, y = _toy_data()
    gp = GaussianProcessSurrogate(random_state=42, n_restarts_optimizer=0)
    gp.fit(X, y)
    mean, std = gp.predict(np.array([[0.0]]), return_std=False)
    assert mean.shape == (1,)
    assert std is None


def test_gp_1d_predict_shape() -> None:
    X, y = _toy_data()
    gp = GaussianProcessSurrogate(random_state=42, n_restarts_optimizer=0)
    gp.fit(X, y)
    mean, std = gp.predict(np.array([0.0]))
    assert mean.shape == (1,)
    assert std is not None
    assert std.shape == (1,)
