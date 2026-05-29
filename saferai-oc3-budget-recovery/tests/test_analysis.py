from __future__ import annotations

import numpy as np
import pandas as pd

from saferai_budget_recovery.analysis import (
    compute_auc_by_seed_policy,
    compute_policy_differences,
    summarize_policy_results,
    summarize_concentration_by_budget,
    summarize_win_rate_by_budget,
    summarize_win_rate_by_seed,
)


def _results_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_name": "uniform_step_balanced",
                "reveal_seed": 1,
                "budget": 45,
                "squared_wasserstein2_to_full_reference": 0.0,
                "p_success_mean": 0.2,
                "step_count_l1_from_perfect_balance": 0.0,
            },
            {
                "policy_name": "greedy_loo_fragility",
                "reveal_seed": 1,
                "budget": 45,
                "squared_wasserstein2_to_full_reference": 0.0,
                "p_success_mean": 0.2,
                "step_count_l1_from_perfect_balance": 0.0,
            },
            {
                "policy_name": "uniform_step_balanced",
                "reveal_seed": 1,
                "budget": 90,
                "squared_wasserstein2_to_full_reference": 2.0,
                "p_success_mean": 0.3,
                "step_count_l1_from_perfect_balance": 0.0,
            },
            {
                "policy_name": "greedy_loo_fragility",
                "reveal_seed": 1,
                "budget": 90,
                "squared_wasserstein2_to_full_reference": 1.0,
                "p_success_mean": 0.4,
                "step_count_l1_from_perfect_balance": 10.0,
            },
            {
                "policy_name": "uniform_step_balanced",
                "reveal_seed": 2,
                "budget": 90,
                "squared_wasserstein2_to_full_reference": 4.0,
                "p_success_mean": 0.5,
                "step_count_l1_from_perfect_balance": 0.0,
            },
            {
                "policy_name": "greedy_loo_fragility",
                "reveal_seed": 2,
                "budget": 90,
                "squared_wasserstein2_to_full_reference": 5.0,
                "p_success_mean": 0.6,
                "step_count_l1_from_perfect_balance": 20.0,
            },
        ]
    )


def test_summarize_policy_results_grouped_means_and_counts() -> None:
    summary = summarize_policy_results(_results_df())
    row = summary.loc[
        summary["policy_name"].eq("uniform_step_balanced") & summary["budget"].eq(90)
    ].iloc[0]
    assert row["n_runs"] == 2
    assert row["distance_mean"] == 3.0
    assert row["step_count_l1_mean"] == 0.0


def test_compute_policy_differences_aligns_by_seed_and_budget() -> None:
    diff = compute_policy_differences(_results_df())
    row = diff.loc[diff["reveal_seed"].eq(1) & diff["budget"].eq(90)].iloc[0]
    assert row["uniform_distance"] == 2.0
    assert row["greedy_distance"] == 1.0
    assert row["difference_greedy_minus_uniform"] == -1.0


def test_greedy_better_true_when_greedy_distance_lower() -> None:
    diff = compute_policy_differences(_results_df())
    row = diff.loc[diff["reveal_seed"].eq(1) & diff["budget"].eq(90)].iloc[0]
    assert bool(row["greedy_better"]) is True


def test_relative_difference_handles_zero_baseline() -> None:
    diff = compute_policy_differences(_results_df())
    row = diff.loc[diff["reveal_seed"].eq(1) & diff["budget"].eq(45)].iloc[0]
    assert np.isnan(row["relative_difference_greedy_minus_uniform"])


def test_auc_calculation_returns_finite_nonnegative_values() -> None:
    auc = compute_auc_by_seed_policy(_results_df())
    assert np.isfinite(auc["auc_distance"]).all()
    assert (auc["auc_distance"] >= 0).all()


def test_summarize_win_rate_by_budget_counts_greedy_wins() -> None:
    diff = compute_policy_differences(_results_df())
    win_rate = summarize_win_rate_by_budget(diff)
    row = win_rate.loc[win_rate["budget"].eq(90)].iloc[0]
    assert row["n_seeds"] == 2
    assert row["greedy_wins"] == 1
    assert row["greedy_win_fraction"] == 0.5


def test_summarize_win_rate_by_seed_includes_auc_difference() -> None:
    results = _results_df()
    diff = compute_policy_differences(results)
    auc = compute_auc_by_seed_policy(results)
    by_seed = summarize_win_rate_by_seed(diff, auc)
    row = by_seed.loc[by_seed["reveal_seed"].eq(1)].iloc[0]
    assert row["n_budgets"] == 2
    assert "greedy_minus_uniform_auc" in by_seed.columns


def test_summarize_concentration_by_budget_counts_initial_and_above_initial_steps() -> None:
    results = pd.DataFrame(
        [
            {
                "policy_name": "greedy_loo_fragility",
                "budget": 90,
                "step_count_l1_from_perfect_balance": 80.0,
                "max_min_revealed_rows_per_step_ratio": 10.0,
                "max_revealed_rows_per_step": 50,
                "min_revealed_rows_per_step": 5,
                "revealed_rows_by_step": '{"a": 50, "b": 5, "c": 35}',
            },
            {
                "policy_name": "greedy_loo_fragility",
                "budget": 90,
                "step_count_l1_from_perfect_balance": 70.0,
                "max_min_revealed_rows_per_step_ratio": 8.0,
                "max_revealed_rows_per_step": 40,
                "min_revealed_rows_per_step": 5,
                "revealed_rows_by_step": '{"a": 40, "b": 45, "c": 5}',
            },
        ]
    )
    summary = summarize_concentration_by_budget(results)
    row = summary.iloc[0]
    assert row["mean_step_count_l1_imbalance"] == 75.0
    assert row["average_steps_above_initial_5"] == 2.0
    assert row["average_steps_still_at_initial_5"] == 1.0
