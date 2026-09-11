"""Tests for the Random Search baseline."""

import numpy as np
import pytest

from src.black_box import sphere
from src.random_search import RandomSearch


def test_random_search_returns_result() -> None:
    optimizer = RandomSearch(
        objective=sphere,
        bounds=np.array([[-5.0, 5.0]]),
        minimize=True,
        random_state=42,
    )
    best_x, best_y = optimizer.optimize(n_iterations=10)
    assert best_x.shape == (1,)
    assert isinstance(best_y, float)


def test_random_search_stays_inside_bounds() -> None:
    bounds = np.array([[-5.0, 5.0], [-1.0, 2.0]])
    optimizer = RandomSearch(
        objective=sphere,
        bounds=bounds,
        minimize=True,
        random_state=42,
    )
    optimizer.optimize(n_iterations=20)
    X = np.asarray(optimizer.X)
    assert np.all(X[:, 0] >= -5.0)
    assert np.all(X[:, 0] <= 5.0)
    assert np.all(X[:, 1] >= -1.0)
    assert np.all(X[:, 1] <= 2.0)


def test_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="shape"):
        RandomSearch(objective=sphere, bounds=np.array([0.0, 1.0]))


def test_invalid_iterations() -> None:
    optimizer = RandomSearch(
        objective=sphere,
        bounds=np.array([[-1.0, 1.0]]),
        random_state=0,
    )
    with pytest.raises(ValueError, match="n_iterations"):
        optimizer.optimize(n_iterations=0)


def test_reproducibility() -> None:
    bounds = np.array([[-5.0, 5.0]])
    first = RandomSearch(sphere, bounds, random_state=11)
    second = RandomSearch(sphere, bounds, random_state=11)
    x1, y1 = first.optimize(n_iterations=12)
    x2, y2 = second.optimize(n_iterations=12)
    np.testing.assert_allclose(x1, x2)
    assert y1 == y2
    np.testing.assert_allclose(first.history, second.history)


def test_history_length() -> None:
    optimizer = RandomSearch(
        objective=sphere,
        bounds=np.array([[-5.0, 5.0]]),
        random_state=3,
    )
    optimizer.optimize(n_iterations=9)
    assert len(optimizer.history) == 9
    assert len(optimizer.y) == 9
    assert optimizer.history[-1] == optimizer.best_y
    assert optimizer.best_y == min(optimizer.y)


def test_optimize_resets_between_runs() -> None:
    optimizer = RandomSearch(
        objective=sphere,
        bounds=np.array([[-5.0, 5.0]]),
        random_state=11,
    )
    _, y1 = optimizer.optimize(n_iterations=8)
    _, y2 = optimizer.optimize(n_iterations=8)
    assert y1 == y2
    assert len(optimizer.y) == 8
