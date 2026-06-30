from __future__ import annotations

import numpy as np
import pytest

from saferai_budget_recovery.distances import (
    empirical_quantiles,
    quantile_grid,
    squared_wasserstein2_from_quantiles,
    squared_wasserstein2_from_samples,
)


def test_quantile_grid_shape_and_endpoints() -> None:
    grid = quantile_grid(11)
    assert grid.shape == (11,)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0


def test_empirical_quantiles_on_simple_array() -> None:
    samples = np.array([0.0, 1.0, 2.0, 3.0])
    grid = np.array([0.0, 0.5, 1.0])
    assert np.allclose(empirical_quantiles(samples, grid), np.array([0.0, 1.5, 3.0]))


def test_squared_w2_zero_for_identical_quantiles_and_samples() -> None:
    q = np.array([0.0, 0.5, 1.0])
    grid = np.array([0.0, 0.5, 1.0])
    assert squared_wasserstein2_from_quantiles(q, q, grid) == pytest.approx(0.0)
    samples = np.array([0.1, 0.2, 0.3])
    assert squared_wasserstein2_from_samples(samples, samples) == pytest.approx(0.0)


def test_squared_w2_positive_for_shifted_samples() -> None:
    samples_a = np.array([0.0, 0.0, 0.0])
    samples_b = np.array([1.0, 1.0, 1.0])
    assert squared_wasserstein2_from_samples(samples_a, samples_b) > 0


def test_non_finite_samples_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="samples must contain only finite values"):
        empirical_quantiles(np.array([0.0, np.nan]), quantile_grid())

