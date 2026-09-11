"""Tests for BayesianOptimizer."""

import numpy as np
import pytest

from src.black_box import sphere
from src.optimizer import BayesianOptimizer


def _optimizer(**kwargs: object) -> BayesianOptimizer:
    defaults: dict[str, object] = {
        "objective": sphere,
        "bounds": np.array([[-5.0, 5.0]]),
        "acquisition": "ei",
        "minimize": True,
        "random_state": 42,
        "n_candidates": 48,
        "n_acquisition_restarts": 2,
    }
    defaults.update(kwargs)
    return BayesianOptimizer(**defaults)  # type: ignore[arg-type]


def test_bayesian_optimizer_returns_result() -> None:
    optimizer = _optimizer()
    best_x, best_y = optimizer.optimize(n_initial=3, n_iterations=4)
    assert best_x.shape == (1,)
    assert isinstance(best_y, float)


def test_bayesian_optimizer_stays_inside_bounds() -> None:
    bounds = np.array([[-2.0, 2.0], [-1.0, 1.0]])
    optimizer = _optimizer(bounds=bounds)
    optimizer.optimize(n_initial=3, n_iterations=4)
    X = np.asarray(optimizer.X)
    assert np.all(X[:, 0] >= -2.0 - 1e-12)
    assert np.all(X[:, 0] <= 2.0 + 1e-12)
    assert np.all(X[:, 1] >= -1.0 - 1e-12)
    assert np.all(X[:, 1] <= 1.0 + 1e-12)


def test_history_length_and_evaluation_count() -> None:
    optimizer = _optimizer()
    optimizer.optimize(n_initial=3, n_iterations=4)
    assert len(optimizer.y) == 7
    assert len(optimizer.history) == 7
    assert optimizer.history[-1] == optimizer.best_y
    assert optimizer.best_y == min(optimizer.y)


def test_minimize_mode_tracks_minimum() -> None:
    optimizer = _optimizer(minimize=True)
    _, best_y = optimizer.optimize(n_initial=3, n_iterations=3)
    assert best_y == min(optimizer.y)
    assert np.all(np.diff(optimizer.history) <= 1e-12)


def test_maximize_mode_tracks_maximum() -> None:
    optimizer = _optimizer(minimize=False)
    _, best_y = optimizer.optimize(n_initial=3, n_iterations=3)
    assert best_y == max(optimizer.y)
    assert np.all(np.diff(optimizer.history) >= -1e-12)


def test_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="shape"):
        BayesianOptimizer(objective=sphere, bounds=np.array([-5.0, 5.0]))
    with pytest.raises(ValueError, match="lower bound"):
        BayesianOptimizer(
            objective=sphere,
            bounds=np.array([[5.0, -5.0]]),
        )


def test_invalid_acquisition() -> None:
    with pytest.raises(ValueError, match="acquisition"):
        _optimizer(acquisition="thompson")


def test_invalid_iteration_counts() -> None:
    optimizer = _optimizer()
    with pytest.raises(ValueError, match="n_initial"):
        optimizer.optimize(n_initial=0, n_iterations=3)
    with pytest.raises(ValueError, match="n_iterations"):
        optimizer.optimize(n_initial=3, n_iterations=0)


def test_reproducibility_with_same_seed() -> None:
    first = _optimizer(random_state=7)
    second = _optimizer(random_state=7)
    x1, y1 = first.optimize(n_initial=3, n_iterations=3)
    x2, y2 = second.optimize(n_initial=3, n_iterations=3)
    np.testing.assert_allclose(x1, x2)
    assert y1 == y2
    np.testing.assert_allclose(first.history, second.history)


def test_optimize_resets_state() -> None:
    optimizer = _optimizer(random_state=7)
    _, y1 = optimizer.optimize(n_initial=3, n_iterations=3)
    history_first = list(optimizer.history)
    _, y2 = optimizer.optimize(n_initial=3, n_iterations=3)
    assert y1 == y2
    assert history_first == optimizer.history
    assert len(optimizer.y) == 6


def test_different_seeds_can_differ() -> None:
    first = _optimizer(random_state=1)
    second = _optimizer(random_state=2)
    first.optimize(n_initial=3, n_iterations=3)
    second.optimize(n_initial=3, n_iterations=3)
    assert not np.allclose(np.asarray(first.X), np.asarray(second.X))


def test_objective_evaluation_count() -> None:
    calls = {"n": 0}

    def counted_sphere(x: np.ndarray) -> float:
        calls["n"] += 1
        return sphere(x)

    optimizer = _optimizer(objective=counted_sphere)
    optimizer.optimize(n_initial=4, n_iterations=5)
    assert calls["n"] == 9


def test_non_finite_objective_raises() -> None:
    def exploding(_x: np.ndarray) -> float:
        return float("nan")

    optimizer = _optimizer(objective=exploding)
    with pytest.raises(ValueError, match="finite"):
        optimizer.optimize(n_initial=2, n_iterations=1)


def test_step_before_initialize_raises() -> None:
    optimizer = _optimizer()
    with pytest.raises(RuntimeError, match="initialize"):
        optimizer.step()


def test_acquisition_local_optimization_stays_in_bounds() -> None:
    bounds = np.array([[-1.5, 1.5]])
    optimizer = _optimizer(
        bounds=bounds,
        n_candidates=32,
        n_acquisition_restarts=4,
    )
    best_x, _ = optimizer.optimize(n_initial=3, n_iterations=3)
    assert -1.5 <= float(best_x[0]) <= 1.5
