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


def choose_step_uniform_row_random(
    available_steps: set[str],
    available_row_counts_by_step: dict[str, int],
    rng: np.random.Generator,
) -> str:
    """Diagnostic policy: choose a step with probability proportional to unrevealed rows."""

    candidates = sorted(str(step) for step in available_steps)
    if not candidates:
        raise ValueError("Cannot choose a row-random step from an empty available step set.")
    weights = np.asarray([available_row_counts_by_step.get(step, 0) for step in candidates], dtype=float)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Available row counts must be positive for at least one available step.")
    probabilities = weights / weights.sum()
    return str(candidates[int(rng.choice(len(candidates), p=probabilities))])


def choose_step_uniform_step_balanced(
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
) -> str:
    """V8 baseline: choose among available steps with the smallest revealed count."""

    revealed = _ensure_stable_row_id(revealed_df)
    candidates = sorted(str(step) for step in available_steps)
    if not candidates:
        raise ValueError("Cannot choose a step-balanced reveal from an empty available step set.")
    revealed_counts = revealed.groupby("step_name").size().to_dict()
    min_count = min(int(revealed_counts.get(step, 0)) for step in candidates)
    under_sampled = [
        step for step in candidates if int(revealed_counts.get(step, 0)) == min_count
    ]
    return str(under_sampled[int(rng.integers(0, len(under_sampled)))])


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

    selected_step = choose_step_uniform_step_balanced(
        revealed, set(unrevealed["step_name"].dropna().astype(str)), rng
    )
    return _choose_row_from_step(unrevealed, selected_step, rng)


def choose_step_greedy_fragility(
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
    fragility_scores: pd.DataFrame,
) -> str:
    """Choose the available step with largest finite fragility, falling back to uniform."""

    selected_step = _select_fragile_step_from_available(fragility_scores, available_steps)
    if selected_step is None:
        selected_step = choose_step_uniform_step_balanced(revealed_df, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = "under_sampled_all_scores_nan_or_no_available_step"
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "fallback"
        return selected_step
    fragility_scores.attrs["selection_fallback"] = None
    fragility_scores.attrs["selected_step"] = selected_step
    fragility_scores.attrs["decision_type"] = "exploit"
    return selected_step


def choose_step_stochastic_normalized_fragility(
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
    fragility_scores: pd.DataFrame,
) -> str:
    """Sample an available step with probability proportional to finite positive fragility."""

    revealed = _ensure_stable_row_id(revealed_df)
    eligible = _eligible_score_rows_for_steps(fragility_scores, available_steps)
    scored = eligible.copy()
    scored["fragility_for_probability"] = np.where(
        np.isfinite(scored["loo_fragility"]) & scored["loo_fragility"].gt(0),
        scored["loo_fragility"].to_numpy(dtype=float),
        0.0,
    )
    total = float(scored["fragility_for_probability"].sum())
    if total <= 0:
        selected_step = choose_step_uniform_step_balanced(revealed, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = "under_sampled_no_positive_fragility_mass"
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "fallback"
        fragility_scores.attrs["normalized_fragility_total"] = total
        return selected_step

    candidates = scored.sort_values("step_name").reset_index(drop=True)
    probabilities = candidates["fragility_for_probability"].to_numpy(dtype=float) / total
    selected_index = int(rng.choice(len(candidates), p=probabilities))
    selected = candidates.iloc[selected_index]
    selected_step = str(selected["step_name"])

    fragility_scores.attrs["selection_fallback"] = None
    fragility_scores.attrs["selected_step"] = selected_step
    fragility_scores.attrs["decision_type"] = "stochastic_fragility"
    fragility_scores.attrs["normalized_fragility_total"] = total
    fragility_scores.attrs["selected_fragility"] = float(selected["fragility_for_probability"])
    fragility_scores.attrs["selected_probability"] = float(probabilities[selected_index])
    return selected_step


def choose_step_uniform_positive_fragility(
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
    fragility_scores: pd.DataFrame,
) -> str:
    """Sample uniformly among available steps with finite positive fragility."""

    revealed = _ensure_stable_row_id(revealed_df)
    eligible = _eligible_score_rows_for_steps(fragility_scores, available_steps)
    positive = eligible.loc[np.isfinite(eligible["loo_fragility"]) & eligible["loo_fragility"].gt(0)]
    if positive.empty:
        selected_step = choose_step_uniform_step_balanced(revealed, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = "under_sampled_no_positive_fragility_mass"
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "fallback"
        return selected_step

    candidates = positive.sort_values("step_name").reset_index(drop=True)
    selected_index = int(rng.integers(0, len(candidates)))
    selected = candidates.iloc[selected_index]
    selected_step = str(selected["step_name"])

    fragility_scores.attrs["selection_fallback"] = None
    fragility_scores.attrs["selected_step"] = selected_step
    fragility_scores.attrs["decision_type"] = "uniform_positive_fragility"
    fragility_scores.attrs["selected_fragility"] = float(selected["loo_fragility"])
    fragility_scores.attrs["selected_probability"] = float(1.0 / len(candidates))
    return selected_step


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

    selected_step = choose_step_greedy_fragility(
        revealed,
        set(unrevealed["step_name"].dropna().astype(str)),
        rng,
        fragility_scores,
    )
    row_id = _choose_row_from_step(unrevealed, selected_step, rng)
    return row_id, fragility_scores


def choose_step_epsilon_greedy_fragility(
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
    epsilon: float,
    fragility_scores: pd.DataFrame,
) -> str:
    """Choose a step by epsilon-greedy LOO fragility with under-sampled exploration."""

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1].")
    if float(rng.random()) < epsilon:
        selected_step = choose_step_uniform_step_balanced(revealed_df, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = None
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "explore"
        fragility_scores.attrs["epsilon"] = float(epsilon)
        return selected_step

    selected_step = _select_fragile_step_from_available(fragility_scores, available_steps)
    if selected_step is None:
        selected_step = choose_step_uniform_step_balanced(revealed_df, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = "under_sampled_all_scores_nan_or_no_available_step"
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "fallback"
        fragility_scores.attrs["epsilon"] = float(epsilon)
        return selected_step

    fragility_scores.attrs["selection_fallback"] = None
    fragility_scores.attrs["selected_step"] = selected_step
    fragility_scores.attrs["decision_type"] = "exploit"
    fragility_scores.attrs["epsilon"] = float(epsilon)
    return selected_step


def choose_step_stochastic_epsilon_greedy_fragility(
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
    epsilon: float,
    fragility_scores: pd.DataFrame,
) -> str:
    """Explore by step balance, otherwise sample proportional to positive fragility."""

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1].")
    if float(rng.random()) < epsilon:
        selected_step = choose_step_uniform_step_balanced(revealed_df, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = None
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "explore"
        fragility_scores.attrs["epsilon"] = float(epsilon)
        return selected_step

    selected_step = choose_step_stochastic_normalized_fragility(
        revealed_df,
        available_steps,
        rng,
        fragility_scores,
    )
    if fragility_scores.attrs.get("selection_fallback") is None:
        fragility_scores.attrs["decision_type"] = "stochastic_exploit"
    fragility_scores.attrs["epsilon"] = float(epsilon)
    return selected_step


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
    selected_step = choose_step_epsilon_greedy_fragility(
        revealed,
        set(unrevealed["step_name"].dropna().astype(str)),
        rng,
        epsilon,
        scores,
    )
    row_id = _choose_row_from_step(unrevealed, selected_step, rng)
    return row_id, scores


def choose_step_exploration_bonus_fragility(
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
    c: float,
    fragility_scores: pd.DataFrame,
) -> str:
    """Choose a step by LOO fragility plus c-scaled under-sampling bonus."""

    if c < 0:
        raise ValueError("c must be non-negative.")
    revealed = _ensure_stable_row_id(revealed_df)
    eligible = _eligible_score_rows_for_steps(fragility_scores, available_steps)
    positive = eligible.loc[np.isfinite(eligible["loo_fragility"]) & eligible["loo_fragility"].gt(0)]
    if positive.empty:
        selected_step = choose_step_uniform_step_balanced(revealed, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = "under_sampled_no_positive_fragility_scale"
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "fallback"
        fragility_scores.attrs["exploration_bonus_c"] = float(c)
        fragility_scores.attrs["exploration_bonus_lambda"] = None
        return selected_step

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

    fragility_scores.attrs["selection_fallback"] = None
    fragility_scores.attrs["selected_step"] = selected_step
    fragility_scores.attrs["decision_type"] = "bonus"
    fragility_scores.attrs["exploration_bonus_c"] = float(c)
    fragility_scores.attrs["exploration_bonus_lambda"] = lambda_value
    fragility_scores.attrs["selected_fragility"] = float(selected["fragility_for_score"])
    fragility_scores.attrs["selected_bonus"] = float(selected["exploration_bonus"])
    fragility_scores.attrs["selected_acquisition_score"] = float(selected["acquisition_score"])
    return selected_step


def choose_step_stochastic_exploration_bonus_fragility(
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
    c: float,
    fragility_scores: pd.DataFrame,
) -> str:
    """Sample a step proportional to LOO fragility plus c-scaled under-sampling bonus."""

    if c < 0:
        raise ValueError("c must be non-negative.")
    revealed = _ensure_stable_row_id(revealed_df)
    eligible = _eligible_score_rows_for_steps(fragility_scores, available_steps)
    positive = eligible.loc[np.isfinite(eligible["loo_fragility"]) & eligible["loo_fragility"].gt(0)]
    if positive.empty:
        selected_step = choose_step_uniform_step_balanced(revealed, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = "under_sampled_no_positive_fragility_scale"
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "fallback"
        fragility_scores.attrs["exploration_bonus_c"] = float(c)
        fragility_scores.attrs["exploration_bonus_lambda"] = None
        return selected_step

    scale = float(np.median(positive["loo_fragility"].to_numpy(dtype=float)))
    lambda_value = float(c) * scale
    revealed_counts = revealed.groupby("step_name").size().to_dict()
    scored = eligible.copy()
    scored["n_step"] = scored["step_name"].map(lambda step: max(int(revealed_counts.get(step, 0)), 1))
    scored["fragility_for_score"] = np.where(
        np.isfinite(scored["loo_fragility"]) & scored["loo_fragility"].gt(0),
        scored["loo_fragility"].to_numpy(dtype=float),
        0.0,
    )
    scored["exploration_bonus"] = 1.0 / np.sqrt(scored["n_step"].to_numpy(dtype=float))
    scored["acquisition_score"] = (
        scored["fragility_for_score"] + lambda_value * scored["exploration_bonus"]
    )
    total = float(scored["acquisition_score"].sum())
    if total <= 0:
        selected_step = choose_step_uniform_step_balanced(revealed, available_steps, rng)
        fragility_scores.attrs["selection_fallback"] = "under_sampled_no_positive_acquisition_mass"
        fragility_scores.attrs["selected_step"] = selected_step
        fragility_scores.attrs["decision_type"] = "fallback"
        fragility_scores.attrs["exploration_bonus_c"] = float(c)
        fragility_scores.attrs["exploration_bonus_lambda"] = lambda_value
        return selected_step

    candidates = scored.sort_values("step_name").reset_index(drop=True)
    probabilities = candidates["acquisition_score"].to_numpy(dtype=float) / total
    selected_index = int(rng.choice(len(candidates), p=probabilities))
    selected = candidates.iloc[selected_index]
    selected_step = str(selected["step_name"])

    fragility_scores.attrs["selection_fallback"] = None
    fragility_scores.attrs["selected_step"] = selected_step
    fragility_scores.attrs["decision_type"] = "stochastic_bonus"
    fragility_scores.attrs["exploration_bonus_c"] = float(c)
    fragility_scores.attrs["exploration_bonus_lambda"] = lambda_value
    fragility_scores.attrs["selected_fragility"] = float(selected["fragility_for_score"])
    fragility_scores.attrs["selected_bonus"] = float(selected["exploration_bonus"])
    fragility_scores.attrs["selected_acquisition_score"] = float(selected["acquisition_score"])
    fragility_scores.attrs["selected_probability"] = float(probabilities[selected_index])
    return selected_step


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
    selected_step = choose_step_exploration_bonus_fragility(
        revealed,
        set(unrevealed["step_name"].dropna().astype(str)),
        rng,
        c,
        scores,
    )
    row_id = _choose_row_from_step(unrevealed, selected_step, rng)
    return row_id, scores


def _select_fragile_step_from_available(
    fragility_scores: pd.DataFrame,
    available_steps: set[str],
) -> str | None:
    required = {"step_name", "loo_fragility"}
    missing = required - set(fragility_scores.columns)
    if missing:
        raise ValueError(f"Fragility score table is missing columns: {sorted(missing)}")

    available_steps = set(str(step) for step in available_steps)
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
    return _eligible_score_rows_for_steps(
        fragility_scores,
        set(unrevealed_df["step_name"].dropna().astype(str)),
    )


def _eligible_score_rows_for_steps(
    fragility_scores: pd.DataFrame,
    available_steps: set[str],
) -> pd.DataFrame:
    required = {"step_name", "loo_fragility"}
    missing = required - set(fragility_scores.columns)
    if missing:
        raise ValueError(f"Fragility score table is missing columns: {sorted(missing)}")
    available_steps = set(str(step) for step in available_steps)
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
