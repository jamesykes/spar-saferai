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
    max_loo_terms_per_step: int | None = None,
    loo_subsample_seed: int | None = None,
) -> pd.DataFrame:
    """Compute LOO output fragility for each expected MITRE-step input."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    if max_loo_terms_per_step is not None and max_loo_terms_per_step <= 0:
        raise ValueError("max_loo_terms_per_step must be positive when provided.")

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
        n_available = int(len(step_rows)) if len(step_rows) >= 2 else 0
        loo_rows = _select_loo_rows(
            step_rows=step_rows,
            step_index=step_index,
            seed=seed,
            max_loo_terms_per_step=max_loo_terms_per_step,
            loo_subsample_seed=loo_subsample_seed,
        )
        n_used = int(len(loo_rows))
        loo_subsampled = bool(n_available > n_used)

        if len(step_rows) >= 2:
            for loo_index, row in enumerate(loo_rows.to_dict(orient="records")):
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
                "n_loo_terms_available": n_available,
                "n_loo_terms_used": n_used,
                "loo_subsampled": loo_subsampled,
                "max_loo_terms_per_step": max_loo_terms_per_step,
                "n_failed_loo_terms": int(n_failed),
                "sample_seed": int(seed),
                "n_samples": int(n_samples),
                "n_grid": int(n_grid),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["total_loo_terms_available"] = int(out["n_loo_terms_available"].sum())
    out.attrs["total_loo_terms_used"] = int(out["n_loo_terms_used"].sum())
    out.attrs["any_loo_subsampled"] = bool(out["loo_subsampled"].any())
    out.attrs["max_loo_terms_per_step"] = max_loo_terms_per_step
    return out


def _ensure_stable_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if FITTED_ROW_UID_COLUMN in df.columns:
        return df.copy()
    return usable_fit_rows(df)


def _loo_seed(seed: int, step_index: int, loo_index: int, common_random_numbers: bool) -> int:
    if common_random_numbers:
        return int(seed + (step_index + 1) * 100_000 + loo_index)
    return int(seed + (step_index + 1) * 10_000_000 + loo_index * 101)


def _select_loo_rows(
    step_rows: pd.DataFrame,
    step_index: int,
    seed: int,
    max_loo_terms_per_step: int | None,
    loo_subsample_seed: int | None,
) -> pd.DataFrame:
    if len(step_rows) < 2:
        return step_rows.iloc[0:0].copy()
    if max_loo_terms_per_step is None or len(step_rows) <= max_loo_terms_per_step:
        return step_rows.copy()

    base_seed = seed if loo_subsample_seed is None else loo_subsample_seed
    rng = np.random.default_rng(int(base_seed + (step_index + 1) * 1_000_003))
    chosen_positions = rng.choice(len(step_rows), size=max_loo_terms_per_step, replace=False)
    return step_rows.iloc[np.sort(chosen_positions)].copy()
