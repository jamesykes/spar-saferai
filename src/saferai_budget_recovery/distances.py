"""One-dimensional empirical distribution summaries and distances."""

from __future__ import annotations

import numpy as np


def quantile_grid(n_grid: int = 1001) -> np.ndarray:
    """Return an evenly spaced quantile grid on [0, 1], including endpoints."""

    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    return np.linspace(0.0, 1.0, n_grid)


def empirical_quantiles(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate empirical quantiles on the provided grid."""

    sample_arr = _finite_1d_array(samples, "samples")
    grid_arr = _finite_1d_array(grid, "grid")
    if np.any((grid_arr < 0.0) | (grid_arr > 1.0)):
        raise ValueError("grid values must lie in [0, 1].")
    return np.quantile(sample_arr, grid_arr)


def squared_wasserstein2_from_quantiles(
    q_a: np.ndarray,
    q_b: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Approximate one-dimensional W2^2 by trapezoidal integration of squared quantile gaps."""

    q_a_arr = _finite_1d_array(q_a, "q_a")
    q_b_arr = _finite_1d_array(q_b, "q_b")
    if q_a_arr.shape != q_b_arr.shape:
        raise ValueError("q_a and q_b must have the same shape.")
    if grid is None:
        grid_arr = quantile_grid(len(q_a_arr))
    else:
        grid_arr = _finite_1d_array(grid, "grid")
    if grid_arr.shape != q_a_arr.shape:
        raise ValueError("grid must have the same shape as q_a and q_b.")
    if np.any((grid_arr < 0.0) | (grid_arr > 1.0)):
        raise ValueError("grid values must lie in [0, 1].")

    squared_gap = (q_a_arr - q_b_arr) ** 2
    distance = float(np.trapezoid(squared_gap, grid_arr))
    return max(0.0, distance)


def squared_wasserstein2_from_samples(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_grid: int = 1001,
) -> float:
    """Compute W2^2 between empirical samples using a shared quantile grid."""

    grid = quantile_grid(n_grid)
    q_a = empirical_quantiles(samples_a, grid)
    q_b = empirical_quantiles(samples_b, grid)
    return squared_wasserstein2_from_quantiles(q_a, q_b, grid)


def _finite_1d_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if len(arr) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr

