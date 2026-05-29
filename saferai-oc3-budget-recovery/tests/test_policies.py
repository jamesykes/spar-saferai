from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saferai_budget_recovery.policies import (
    choose_next_epsilon_greedy_fragility,
    choose_next_exploration_bonus_fragility,
    choose_next_greedy_fragility,
    choose_next_uniform_row_random,
    choose_next_uniform_step_balanced,
)
from saferai_budget_recovery.reveal import FITTED_ROW_UID_COLUMN, usable_fit_rows


def _rows() -> pd.DataFrame:
    return usable_fit_rows(
        pd.DataFrame(
            [
                {
                    "step_name": "step-a",
                    "model": "model-a",
                    "run_id": "run-a",
                    "repeat_index": 1,
                    "draw_uid": "model-a__run-a__1",
                    "alpha": 2.0,
                    "beta": 5.0,
                    "fit_quality_flag": "ok",
                },
                {
                    "step_name": "step-b",
                    "model": "model-b",
                    "run_id": "run-b",
                    "repeat_index": 1,
                    "draw_uid": "model-b__run-b__1",
                    "alpha": 3.0,
                    "beta": 6.0,
                    "fit_quality_flag": "ok",
                },
                {
                    "step_name": "step-c",
                    "model": "model-c",
                    "run_id": "run-c",
                    "repeat_index": 1,
                    "draw_uid": "model-c__run-c__1",
                    "alpha": 4.0,
                    "beta": 7.0,
                    "fit_quality_flag": "ok",
                },
            ]
        )
    )


def _multi_rows() -> pd.DataFrame:
    rows = []
    for step in ["step-a", "step-b", "step-c"]:
        for repeat in range(5):
            rows.append(
                {
                    "step_name": step,
                    "model": f"model-{repeat}",
                    "run_id": f"run-{repeat}",
                    "repeat_index": repeat,
                    "draw_uid": f"model-{repeat}__run-{repeat}__{repeat}",
                    "alpha": 2.0,
                    "beta": 5.0,
                    "fit_quality_flag": "ok",
                }
            )
    return usable_fit_rows(pd.DataFrame(rows))


def _multi_rows_with_repeats(n_repeats: int) -> pd.DataFrame:
    rows = []
    for step in ["step-a", "step-b", "step-c"]:
        for repeat in range(n_repeats):
            rows.append(
                {
                    "step_name": step,
                    "model": f"model-{repeat}",
                    "run_id": f"run-{repeat}",
                    "repeat_index": repeat,
                    "draw_uid": f"model-{repeat}__run-{repeat}__{repeat}",
                    "alpha": 2.0,
                    "beta": 5.0,
                    "fit_quality_flag": "ok",
                }
            )
    return usable_fit_rows(pd.DataFrame(rows))


def test_uniform_row_random_policy_selects_unrevealed_row_id() -> None:
    unrevealed = _rows()
    selected = choose_next_uniform_row_random(unrevealed, np.random.default_rng(123))
    assert selected in set(unrevealed[FITTED_ROW_UID_COLUMN])


def test_uniform_row_random_policy_is_reproducible_with_fixed_rng_seed() -> None:
    unrevealed = _rows()
    first = choose_next_uniform_row_random(unrevealed, np.random.default_rng(123))
    second = choose_next_uniform_row_random(unrevealed, np.random.default_rng(123))
    assert first == second


def test_uniform_step_balanced_selects_under_sampled_step() -> None:
    rows = _multi_rows()
    revealed = rows.loc[rows["step_name"].eq("step-a")].head(2)
    revealed = pd.concat([revealed, rows.loc[rows["step_name"].eq("step-b")].head(1)])
    unrevealed = rows.loc[~rows[FITTED_ROW_UID_COLUMN].isin(set(revealed[FITTED_ROW_UID_COLUMN]))]
    selected = choose_next_uniform_step_balanced(revealed, unrevealed, np.random.default_rng(123))
    selected_step = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(selected), "step_name"].iloc[0]
    assert selected_step == "step-c"


def test_uniform_step_balanced_repeated_selections_stay_balanced() -> None:
    rows = _multi_rows()
    revealed = rows.groupby("step_name").head(1).copy()
    unrevealed = rows.loc[~rows[FITTED_ROW_UID_COLUMN].isin(set(revealed[FITTED_ROW_UID_COLUMN]))].copy()
    rng = np.random.default_rng(123)
    for _ in range(6):
        selected = choose_next_uniform_step_balanced(revealed, unrevealed, rng)
        selected_row = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(selected)]
        revealed = pd.concat([revealed, selected_row], ignore_index=True)
        unrevealed = unrevealed.loc[~unrevealed[FITTED_ROW_UID_COLUMN].eq(selected)].copy()
    counts = revealed.groupby("step_name").size()
    assert counts.max() - counts.min() <= 1


def test_uniform_step_balanced_never_selects_from_step_with_no_unrevealed_rows() -> None:
    rows = _multi_rows()
    revealed = rows.loc[rows["step_name"].eq("step-a")].copy()
    revealed = pd.concat([revealed, rows.loc[rows["step_name"].eq("step-b")].head(1)])
    unrevealed = rows.loc[rows["step_name"].eq("step-b") | rows["step_name"].eq("step-c")]
    unrevealed = unrevealed.loc[
        ~unrevealed[FITTED_ROW_UID_COLUMN].isin(set(revealed[FITTED_ROW_UID_COLUMN]))
    ]
    selected = choose_next_uniform_step_balanced(revealed, unrevealed, np.random.default_rng(123))
    selected_step = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(selected), "step_name"].iloc[0]
    assert selected_step in {"step-b", "step-c"}
    assert selected_step != "step-a"


def test_uniform_step_balanced_is_reproducible_with_fixed_rng_seed() -> None:
    rows = _multi_rows()
    revealed = rows.groupby("step_name").head(1)
    unrevealed = rows.loc[~rows[FITTED_ROW_UID_COLUMN].isin(set(revealed[FITTED_ROW_UID_COLUMN]))]
    first = choose_next_uniform_step_balanced(revealed, unrevealed, np.random.default_rng(123))
    second = choose_next_uniform_step_balanced(revealed, unrevealed, np.random.default_rng(123))
    assert first == second


def test_uniform_step_balanced_empty_unrevealed_raises() -> None:
    rows = _multi_rows()
    with pytest.raises(ValueError, match="empty unrevealed set"):
        choose_next_uniform_step_balanced(rows, rows.iloc[0:0], np.random.default_rng(123))


def test_greedy_fragility_falls_back_to_uniform_if_all_scores_nan() -> None:
    rows = _rows()
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b"], "loo_fragility": [np.nan, np.nan]}
    )
    selected, used_scores = choose_next_greedy_fragility(
        rows.head(1), rows, np.random.default_rng(123), fragility_scores=scores
    )
    assert selected in set(rows[FITTED_ROW_UID_COLUMN])
    assert used_scores.attrs["selection_fallback"] is not None


def test_greedy_fragility_skips_step_with_no_unrevealed_rows() -> None:
    rows = _rows()
    unrevealed = rows.loc[~rows["step_name"].eq("step-a")].copy()
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b"], "loo_fragility": [10.0, 1.0]}
    )
    selected, used_scores = choose_next_greedy_fragility(
        rows, unrevealed, np.random.default_rng(123), fragility_scores=scores
    )
    selected_step = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(selected), "step_name"].iloc[0]
    assert selected_step == "step-b"
    assert used_scores.attrs["selection_fallback"] is None


def test_greedy_fragility_tie_breaks_by_sorted_step_name() -> None:
    rows = _rows()
    scores = pd.DataFrame(
        {"step_name": ["step-b", "step-a"], "loo_fragility": [1.0, 1.0]}
    )
    selected, used_scores = choose_next_greedy_fragility(
        rows, rows, np.random.default_rng(123), fragility_scores=scores
    )
    selected_step = rows.loc[rows[FITTED_ROW_UID_COLUMN].eq(selected), "step_name"].iloc[0]
    assert selected_step == "step-a"
    assert used_scores.attrs["selected_step"] == "step-a"


def test_epsilon_greedy_with_zero_epsilon_exploits_like_greedy() -> None:
    rows = _rows()
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b", "step-c"], "loo_fragility": [1.0, 3.0, 2.0]}
    )
    selected, used_scores = choose_next_epsilon_greedy_fragility(
        rows,
        rows,
        np.random.default_rng(123),
        epsilon=0.0,
        fragility_scores=scores,
    )
    selected_step = rows.loc[rows[FITTED_ROW_UID_COLUMN].eq(selected), "step_name"].iloc[0]
    assert selected_step == "step-b"
    assert used_scores.attrs["decision_type"] == "exploit"


def test_epsilon_greedy_with_one_epsilon_explores_under_sampled_steps() -> None:
    rows = _multi_rows()
    revealed = rows.loc[rows["step_name"].eq("step-a")].head(2)
    revealed = pd.concat([revealed, rows.loc[rows["step_name"].eq("step-b")].head(1)])
    unrevealed = rows.loc[~rows[FITTED_ROW_UID_COLUMN].isin(set(revealed[FITTED_ROW_UID_COLUMN]))]
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b", "step-c"], "loo_fragility": [10.0, 10.0, 0.0]}
    )
    selected, used_scores = choose_next_epsilon_greedy_fragility(
        revealed,
        unrevealed,
        np.random.default_rng(123),
        epsilon=1.0,
        fragility_scores=scores,
    )
    selected_step = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(selected), "step_name"].iloc[0]
    assert selected_step == "step-c"
    assert used_scores.attrs["decision_type"] == "explore"


def test_epsilon_greedy_never_selects_exhausted_step() -> None:
    rows = _multi_rows()
    revealed = rows.loc[rows["step_name"].eq("step-a")].copy()
    unrevealed = rows.loc[~rows["step_name"].eq("step-a")].copy()
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b", "step-c"], "loo_fragility": [100.0, 1.0, 2.0]}
    )
    selected, _ = choose_next_epsilon_greedy_fragility(
        revealed,
        unrevealed,
        np.random.default_rng(123),
        epsilon=0.0,
        fragility_scores=scores,
    )
    selected_step = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(selected), "step_name"].iloc[0]
    assert selected_step == "step-c"


def test_exploration_bonus_selects_eligible_row_and_records_acquisition() -> None:
    rows = _rows()
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b", "step-c"], "loo_fragility": [1.0, 3.0, 2.0]}
    )
    selected, used_scores = choose_next_exploration_bonus_fragility(
        rows,
        rows,
        np.random.default_rng(123),
        c=0.5,
        fragility_scores=scores,
    )
    assert selected in set(rows[FITTED_ROW_UID_COLUMN])
    assert used_scores.attrs["decision_type"] == "bonus"
    assert used_scores.attrs["selected_acquisition_score"] > 0


def test_exploration_bonus_falls_back_when_fragilities_zero_or_nan() -> None:
    rows = _multi_rows()
    revealed = rows.loc[rows["step_name"].eq("step-a")].head(2)
    revealed = pd.concat([revealed, rows.loc[rows["step_name"].eq("step-b")].head(1)])
    unrevealed = rows.loc[~rows[FITTED_ROW_UID_COLUMN].isin(set(revealed[FITTED_ROW_UID_COLUMN]))]
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b", "step-c"], "loo_fragility": [0.0, np.nan, 0.0]}
    )
    selected, used_scores = choose_next_exploration_bonus_fragility(
        revealed,
        unrevealed,
        np.random.default_rng(123),
        c=1.0,
        fragility_scores=scores,
    )
    selected_step = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(selected), "step_name"].iloc[0]
    assert selected_step == "step-c"
    assert used_scores.attrs["decision_type"] == "fallback"


def test_exploration_bonus_larger_c_can_prefer_lower_count_node() -> None:
    rows = _multi_rows_with_repeats(6)
    revealed = rows.loc[rows["step_name"].eq("step-a")].head(5)
    revealed = pd.concat([revealed, rows.loc[rows["step_name"].eq("step-b")].head(1)])
    unrevealed = rows.loc[~rows[FITTED_ROW_UID_COLUMN].isin(set(revealed[FITTED_ROW_UID_COLUMN]))]
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b"], "loo_fragility": [1.0, 0.8]}
    )
    low_c_selected, _ = choose_next_exploration_bonus_fragility(
        revealed,
        unrevealed,
        np.random.default_rng(123),
        c=0.0,
        fragility_scores=scores,
    )
    high_c_selected, _ = choose_next_exploration_bonus_fragility(
        revealed,
        unrevealed,
        np.random.default_rng(123),
        c=2.0,
        fragility_scores=scores,
    )
    low_c_step = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(low_c_selected), "step_name"].iloc[0]
    high_c_step = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(high_c_selected), "step_name"].iloc[0]
    assert low_c_step == "step-a"
    assert high_c_step == "step-b"


def test_parameterized_policy_decisions_are_reproducible() -> None:
    rows = _rows()
    scores = pd.DataFrame(
        {"step_name": ["step-a", "step-b", "step-c"], "loo_fragility": [1.0, 3.0, 2.0]}
    )
    first, _ = choose_next_epsilon_greedy_fragility(
        rows,
        rows,
        np.random.default_rng(123),
        epsilon=0.2,
        fragility_scores=scores,
    )
    second, _ = choose_next_epsilon_greedy_fragility(
        rows,
        rows,
        np.random.default_rng(123),
        epsilon=0.2,
        fragility_scores=scores,
    )
    assert first == second
