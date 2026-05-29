"""Leave-one-out output fragility primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from saferai_budget_recovery import config
from saferai_budget_recovery.distances import (
    empirical_quantiles,
    quantile_grid as make_quantile_grid,
    squared_wasserstein2_from_quantiles,
)
from saferai_budget_recovery.reveal import FITTED_ROW_UID_COLUMN, usable_fit_rows
from saferai_budget_recovery.sampling import sample_p_success_from_revealed


def loo_perturbed_revealed_df(
    revealed_df: pd.DataFrame,
    step_label: str,
    row_id: str,
) -> pd.DataFrame:
    """Return revealed rows with exactly one row removed for one MITRE-step label."""

    revealed = _ensure_stable_row_id(revealed_df)
    target_mask = revealed["step_name"].eq(step_label) & revealed[FITTED_ROW_UID_COLUMN].eq(row_id)
    n_targets = int(target_mask.sum())
    if n_targets != 1:
        raise ValueError(
            "LOO perturbation must identify exactly one revealed fitted row. "
            f"Matched {n_targets} rows for step={step_label!r}, row_id={row_id!r}."
        )
    return revealed.loc[~target_mask].copy().reset_index(drop=True)


def compute_current_output_quantiles(
    revealed_df: pd.DataFrame,
    n_samples: int,
    quantile_grid: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sample the current revealed model and return empirical output quantiles."""

    sample_df = sample_p_success_from_revealed(revealed_df, n_samples=n_samples, seed=seed)
    return empirical_quantiles(sample_df["p_success"].to_numpy(dtype=float), quantile_grid)


def compute_loo_fragility_scores(
    revealed_df: pd.DataFrame,
    n_samples: int = 5000,
    n_grid: int = 501,
    seed: int = 12345,
    common_random_numbers: bool = True,
) -> pd.DataFrame:
    """Compute LOO output fragility for each expected MITRE-step input."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    revealed = usable_fit_rows(revealed_df)
    grid = make_quantile_grid(n_grid)
    current_quantiles = compute_current_output_quantiles(
        revealed, n_samples=n_samples, quantile_grid=grid, seed=seed
    )

    rows: list[dict] = []
    for step_index, step_label in enumerate(config.EXPECTED_MITRE_STEP_LABELS):
        step_rows = revealed.loc[revealed["step_name"].eq(step_label)].copy()
        distances: list[float] = []
        n_failed = 0

        if len(step_rows) >= 2:
            for loo_index, row in enumerate(step_rows.to_dict(orient="records")):
                try:
                    perturbed = loo_perturbed_revealed_df(
                        revealed, step_label=step_label, row_id=str(row[FITTED_ROW_UID_COLUMN])
                    )
                    perturbed_seed = _loo_seed(
                        seed=seed,
                        step_index=step_index,
                        loo_index=loo_index,
                        common_random_numbers=common_random_numbers,
                    )
                    perturbed_quantiles = compute_current_output_quantiles(
                        perturbed,
                        n_samples=n_samples,
                        quantile_grid=grid,
                        seed=perturbed_seed,
                    )
                    distance = squared_wasserstein2_from_quantiles(
                        perturbed_quantiles, current_quantiles, grid
                    )
                    distances.append(float(distance))
                except Exception:
                    n_failed += 1

        finite_distances = np.asarray(distances, dtype=float)
        finite_distances = finite_distances[np.isfinite(finite_distances)]
        if len(finite_distances) == 0:
            mean_distance = np.nan
            median_distance = np.nan
            min_distance = np.nan
            max_distance = np.nan
        else:
            mean_distance = float(np.mean(finite_distances))
            median_distance = float(np.median(finite_distances))
            min_distance = float(np.min(finite_distances))
            max_distance = float(np.max(finite_distances))

        rows.append(
            {
                "step_name": step_label,
                "n_revealed": int(len(step_rows)),
                "loo_fragility": mean_distance,
                "mean_loo_distance": mean_distance,
                "median_loo_distance": median_distance,
                "max_loo_distance": max_distance,
                "min_loo_distance": min_distance,
                "n_loo_terms": int(len(finite_distances)),
                "n_failed_loo_terms": int(n_failed),
                "sample_seed": int(seed),
                "n_samples": int(n_samples),
                "n_grid": int(n_grid),
            }
        )
    return pd.DataFrame(rows)


def _ensure_stable_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if FITTED_ROW_UID_COLUMN in df.columns:
        return df.copy()
    return usable_fit_rows(df)


def _loo_seed(seed: int, step_index: int, loo_index: int, common_random_numbers: bool) -> int:
    if common_random_numbers:
        return int(seed + (step_index + 1) * 100_000 + loo_index)
    return int(seed + (step_index + 1) * 10_000_000 + loo_index * 101)

