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
        row_id = _choose_row_from_under_sampled_steps(revealed, unrevealed, rng)
        fragility_scores.attrs["selection_fallback"] = "under_sampled_all_scores_nan_or_no_available_step"
        fragility_scores.attrs["selected_step"] = _step_for_row(unrevealed, row_id)
        fragility_scores.attrs["decision_type"] = "fallback"
        return row_id, fragility_scores

    row_id = _choose_row_from_step(unrevealed, selected_step, rng)
    fragility_scores.attrs["selection_fallback"] = None
    fragility_scores.attrs["selected_step"] = selected_step
    fragility_scores.attrs["decision_type"] = "exploit"
    return row_id, fragility_scores


def choose_next_epsilon_greedy_fragility(
    revealed_df: pd.DataFrame,
    unrevealed_df: pd.DataFrame,
    rng: np.random.Generator,
    epsilon: float = 0.2,
    fragility_kwargs: dict | None = None,
    fragility_scores: pd.DataFrame | None = None,
) -> tuple[str, pd.DataFrame]:
    """Choose by epsilon-greedy LOO fragility with under-sampled exploration."""

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1].")
    revealed = _ensure_stable_row_id(revealed_df)
    unrevealed = _ensure_stable_row_id(unrevealed_df)
    if unrevealed.empty:
        raise ValueError("Cannot choose an epsilon-greedy reveal from an empty unrevealed set.")

    scores = _get_fragility_scores(revealed, fragility_kwargs, fragility_scores)
    if float(rng.random()) < epsilon:
        row_id = _choose_row_from_under_sampled_steps(revealed, unrevealed, rng)
        scores.attrs["selection_fallback"] = None
        scores.attrs["selected_step"] = _step_for_row(unrevealed, row_id)
        scores.attrs["decision_type"] = "explore"
        scores.attrs["epsilon"] = float(epsilon)
        return row_id, scores

    selected_step = _select_fragile_step_with_available_rows(scores, unrevealed)
    if selected_step is None:
        row_id = _choose_row_from_under_sampled_steps(revealed, unrevealed, rng)
        scores.attrs["selection_fallback"] = "under_sampled_all_scores_nan_or_no_available_step"
        scores.attrs["selected_step"] = _step_for_row(unrevealed, row_id)
        scores.attrs["decision_type"] = "fallback"
        scores.attrs["epsilon"] = float(epsilon)
        return row_id, scores

    row_id = _choose_row_from_step(unrevealed, selected_step, rng)
    scores.attrs["selection_fallback"] = None
    scores.attrs["selected_step"] = selected_step
    scores.attrs["decision_type"] = "exploit"
    scores.attrs["epsilon"] = float(epsilon)
    return row_id, scores


def choose_next_exploration_bonus_fragility(
    revealed_df: pd.DataFrame,
    unrevealed_df: pd.DataFrame,
    rng: np.random.Generator,
    c: float = 0.5,
    fragility_kwargs: dict | None = None,
    fragility_scores: pd.DataFrame | None = None,
) -> tuple[str, pd.DataFrame]:
    """Choose by LOO fragility plus c-scaled under-sampling bonus."""

    if c < 0:
        raise ValueError("c must be non-negative.")
    revealed = _ensure_stable_row_id(revealed_df)
    unrevealed = _ensure_stable_row_id(unrevealed_df)
    if unrevealed.empty:
        raise ValueError("Cannot choose an exploration-bonus reveal from an empty unrevealed set.")

    scores = _get_fragility_scores(revealed, fragility_kwargs, fragility_scores)
    eligible = _eligible_score_rows(scores, unrevealed)
    positive = eligible.loc[np.isfinite(eligible["loo_fragility"]) & eligible["loo_fragility"].gt(0)]
    if positive.empty:
        row_id = _choose_row_from_under_sampled_steps(revealed, unrevealed, rng)
        scores.attrs["selection_fallback"] = "under_sampled_no_positive_fragility_scale"
        scores.attrs["selected_step"] = _step_for_row(unrevealed, row_id)
        scores.attrs["decision_type"] = "fallback"
        scores.attrs["exploration_bonus_c"] = float(c)
        scores.attrs["exploration_bonus_lambda"] = None
        return row_id, scores

    scale = float(np.median(positive["loo_fragility"].to_numpy(dtype=float)))
    lambda_value = float(c) * scale
    revealed_counts = revealed.groupby("step_name").size().to_dict()
    scored = eligible.copy()
    scored["n_step"] = scored["step_name"].map(lambda step: max(int(revealed_counts.get(step, 0)), 1))
    scored["fragility_for_score"] = np.where(
        np.isfinite(scored["loo_fragility"]),
        scored["loo_fragility"].to_numpy(dtype=float),
        0.0,
    )
    scored["exploration_bonus"] = 1.0 / np.sqrt(scored["n_step"].to_numpy(dtype=float))
    scored["acquisition_score"] = (
        scored["fragility_for_score"] + lambda_value * scored["exploration_bonus"]
    )
    ranked = scored.sort_values(["acquisition_score", "step_name"], ascending=[False, True])
    selected = ranked.iloc[0]
    selected_step = str(selected["step_name"])
    row_id = _choose_row_from_step(unrevealed, selected_step, rng)

    scores.attrs["selection_fallback"] = None
    scores.attrs["selected_step"] = selected_step
    scores.attrs["decision_type"] = "bonus"
    scores.attrs["exploration_bonus_c"] = float(c)
    scores.attrs["exploration_bonus_lambda"] = lambda_value
    scores.attrs["selected_fragility"] = float(selected["fragility_for_score"])
    scores.attrs["selected_bonus"] = float(selected["exploration_bonus"])
    scores.attrs["selected_acquisition_score"] = float(selected["acquisition_score"])
    return row_id, scores


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


def _get_fragility_scores(
    revealed_df: pd.DataFrame,
    fragility_kwargs: dict | None,
    fragility_scores: pd.DataFrame | None,
) -> pd.DataFrame:
    if fragility_scores is None:
        return compute_loo_fragility_scores(revealed_df, **(fragility_kwargs or {}))
    return fragility_scores.copy()


def _eligible_score_rows(
    fragility_scores: pd.DataFrame,
    unrevealed_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {"step_name", "loo_fragility"}
    missing = required - set(fragility_scores.columns)
    if missing:
        raise ValueError(f"Fragility score table is missing columns: {sorted(missing)}")
    available_steps = set(unrevealed_df["step_name"].dropna().astype(str))
    out = fragility_scores.loc[fragility_scores["step_name"].astype(str).isin(available_steps)].copy()
    if out.empty:
        raise ValueError("No eligible steps have unrevealed rows.")
    return out


def _choose_row_from_under_sampled_steps(
    revealed_df: pd.DataFrame,
    unrevealed_df: pd.DataFrame,
    rng: np.random.Generator,
) -> str:
    revealed = _ensure_stable_row_id(revealed_df)
    unrevealed = _ensure_stable_row_id(unrevealed_df)
    if unrevealed.empty:
        raise ValueError("Cannot choose from an empty unrevealed set.")
    available_steps = sorted(set(unrevealed["step_name"].dropna().astype(str)))
    revealed_counts = revealed.groupby("step_name").size().to_dict()
    min_count = min(int(revealed_counts.get(step, 0)) for step in available_steps)
    candidate_steps = [
        step for step in available_steps if int(revealed_counts.get(step, 0)) == min_count
    ]
    selected_step = str(candidate_steps[int(rng.integers(0, len(candidate_steps)))])
    return _choose_row_from_step(unrevealed, selected_step, rng)


def _choose_row_from_step(
    unrevealed_df: pd.DataFrame,
    selected_step: str,
    rng: np.random.Generator,
) -> str:
    candidate_rows = unrevealed_df.loc[unrevealed_df["step_name"].eq(selected_step)]
    if candidate_rows.empty:
        raise ValueError(f"Selected step has no unrevealed rows: {selected_step}")
    row_index = int(rng.integers(0, len(candidate_rows)))
    return str(candidate_rows.iloc[row_index][FITTED_ROW_UID_COLUMN])


def _step_for_row(unrevealed_df: pd.DataFrame, row_id: str) -> str:
    match = unrevealed_df.loc[unrevealed_df[FITTED_ROW_UID_COLUMN].eq(row_id), "step_name"]
    if len(match) != 1:
        raise ValueError(f"Cannot identify selected step for row ID: {row_id}")
    return str(match.iloc[0])


def _ensure_stable_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if FITTED_ROW_UID_COLUMN in df.columns:
        return df.copy()
    return usable_fit_rows(df)
