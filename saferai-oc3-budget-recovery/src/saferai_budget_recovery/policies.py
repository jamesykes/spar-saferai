"""Policy primitives for pilot budget-recovery runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from saferai_budget_recovery.fragility import compute_loo_fragility_scores
from saferai_budget_recovery.reveal import FITTED_ROW_UID_COLUMN, usable_fit_rows


def choose_next_uniform_row_random(
    unrevealed_df: pd.DataFrame,
    rng: np.random.Generator,
) -> str:
    """Diagnostic policy: choose one unrevealed fitted-row ID uniformly at random."""

    unrevealed = _ensure_stable_row_id(unrevealed_df)
    if unrevealed.empty:
        raise ValueError("Cannot choose a row-random uniform reveal from an empty unrevealed set.")
    row_index = int(rng.integers(0, len(unrevealed)))
    return str(unrevealed.iloc[row_index][FITTED_ROW_UID_COLUMN])


def choose_next_uniform_step_balanced(
    revealed_df: pd.DataFrame,
    unrevealed_df: pd.DataFrame,
    rng: np.random.Generator,
) -> str:
    """V8 baseline: choose from currently under-sampled steps, then uniformly within that step."""

    revealed = _ensure_stable_row_id(revealed_df)
    unrevealed = _ensure_stable_row_id(unrevealed_df)
    if unrevealed.empty:
        raise ValueError("Cannot choose a step-balanced uniform reveal from an empty unrevealed set.")

    available_steps = sorted(set(unrevealed["step_name"].dropna().astype(str)))
    revealed_counts = revealed.groupby("step_name").size().to_dict()
    counts_for_available = {
        step: int(revealed_counts.get(step, 0))
        for step in available_steps
    }
    min_count = min(counts_for_available.values())
    candidate_steps = [
        step for step in available_steps if counts_for_available[step] == min_count
    ]
    selected_step = str(candidate_steps[int(rng.integers(0, len(candidate_steps)))])
    candidate_rows = unrevealed.loc[unrevealed["step_name"].eq(selected_step)]
    row_index = int(rng.integers(0, len(candidate_rows)))
    return str(candidate_rows.iloc[row_index][FITTED_ROW_UID_COLUMN])


def choose_next_greedy_fragility(
    revealed_df: pd.DataFrame,
    unrevealed_df: pd.DataFrame,
    rng: np.random.Generator,
    fragility_kwargs: dict | None = None,
    fragility_scores: pd.DataFrame | None = None,
) -> tuple[str, pd.DataFrame]:
    """Choose from the step with largest finite LOO fragility, falling back to uniform."""

    revealed = _ensure_stable_row_id(revealed_df)
    unrevealed = _ensure_stable_row_id(unrevealed_df)
    if unrevealed.empty:
        raise ValueError("Cannot choose a greedy-fragility reveal from an empty unrevealed set.")

    if fragility_scores is None:
        kwargs = fragility_kwargs or {}
        fragility_scores = compute_loo_fragility_scores(revealed, **kwargs)
    else:
        fragility_scores = fragility_scores.copy()

    selected_step = _select_fragile_step_with_available_rows(fragility_scores, unrevealed)
    if selected_step is None:
        row_id = choose_next_uniform_row_random(unrevealed, rng)
        fragility_scores.attrs["selection_fallback"] = "uniform_all_scores_nan_or_no_available_step"
        fragility_scores.attrs["selected_step"] = None
        return row_id, fragility_scores

    candidate_rows = unrevealed.loc[unrevealed["step_name"].eq(selected_step)].copy()
    if candidate_rows.empty:
        row_id = choose_next_uniform_row_random(unrevealed, rng)
        fragility_scores.attrs["selection_fallback"] = "uniform_selected_step_empty"
        fragility_scores.attrs["selected_step"] = None
        return row_id, fragility_scores

    row_index = int(rng.integers(0, len(candidate_rows)))
    row_id = str(candidate_rows.iloc[row_index][FITTED_ROW_UID_COLUMN])
    fragility_scores.attrs["selection_fallback"] = None
    fragility_scores.attrs["selected_step"] = selected_step
    return row_id, fragility_scores


def _select_fragile_step_with_available_rows(
    fragility_scores: pd.DataFrame,
    unrevealed_df: pd.DataFrame,
) -> str | None:
    required = {"step_name", "loo_fragility"}
    missing = required - set(fragility_scores.columns)
    if missing:
        raise ValueError(f"Fragility score table is missing columns: {sorted(missing)}")

    available_steps = set(unrevealed_df["step_name"].dropna().astype(str))
    finite_scores = fragility_scores.loc[np.isfinite(fragility_scores["loo_fragility"])].copy()
    finite_scores = finite_scores.loc[finite_scores["step_name"].astype(str).isin(available_steps)]
    if finite_scores.empty:
        return None

    ranked = finite_scores.sort_values(["loo_fragility", "step_name"], ascending=[False, True])
    return str(ranked.iloc[0]["step_name"])


def _ensure_stable_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if FITTED_ROW_UID_COLUMN in df.columns:
        return df.copy()
    return usable_fit_rows(df)
