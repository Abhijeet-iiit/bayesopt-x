"""Tests for acquisition functions."""

import numpy as np
import pytest

from src.acquisition import expected_improvement, upper_confidence_bound


def test_ucb_output_shape() -> None:
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    result = upper_confidence_bound(mean, std, beta=2.0)
    assert result.shape == mean.shape


def test_ucb_scalar_input() -> None:
    result = upper_confidence_bound(1.0, 0.5, beta=2.0)
    assert float(result) == pytest.approx(2.0)


def test_ucb_increases_with_uncertainty() -> None:
    mean = np.array([1.0, 1.0])
    low = upper_confidence_bound(mean, np.array([0.1, 0.1]), beta=2.0)
    high = upper_confidence_bound(mean, np.array([1.0, 1.0]), beta=2.0)
    assert np.all(high > low)


def test_ucb_beta_zero_is_mean() -> None:
    mean = np.array([1.5, -0.2])
    std = np.array([4.0, 3.0])
    result = upper_confidence_bound(mean, std, beta=0.0)
    np.testing.assert_allclose(result, mean)


def test_ucb_rejects_negative_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        upper_confidence_bound(np.array([1.0]), np.array([0.1]), beta=-1.0)


def test_ucb_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        upper_confidence_bound(np.array([1.0, 2.0]), np.array([0.1]))


def test_expected_improvement_output_shape() -> None:
    mean = np.array([0.0, 1.0, 2.0])
    std = np.array([0.1, 0.2, 0.3])
    result = expected_improvement(mean, std, best_value=1.0, xi=0.01)
    assert result.shape == mean.shape


def test_expected_improvement_is_non_negative() -> None:
    mean = np.array([0.0, 1.0, 2.0])
    std = np.array([0.1, 0.2, 0.3])
    result = expected_improvement(mean, std, best_value=1.0, xi=0.01)
    assert np.all(result >= 0.0)


def test_expected_improvement_zero_standard_deviation() -> None:
    mean = np.array([5.0, -1.0])
    std = np.array([0.0, 0.0])
    result = expected_improvement(mean, std, best_value=0.0, xi=0.0)
    np.testing.assert_allclose(result, 0.0)


def test_expected_improvement_prefers_better_mean() -> None:
    std = np.array([0.2, 0.2])
    result = expected_improvement(
        np.array([0.0, 1.0]),
        std,
        best_value=0.0,
        xi=0.0,
    )
    assert result[1] > result[0]


def test_expected_improvement_rejects_negative_xi() -> None:
    with pytest.raises(ValueError, match="xi"):
        expected_improvement(
            np.array([1.0]),
            np.array([0.1]),
            best_value=0.0,
            xi=-0.1,
        )


def test_acquisition_rejects_non_finite_mean() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        upper_confidence_bound(np.array([np.nan]), np.array([0.1]))
