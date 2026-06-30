from __future__ import annotations

import json

import numpy as np
import pandas as pd

from saferai_budget_recovery.reporting import (
    build_concentration_by_budget,
    build_error_by_budget,
    build_main_policy_comparison,
    dataframe_to_markdown,
    write_markdown_table,
)


def _run_outputs() -> dict:
    run_a_results = pd.DataFrame(
        [
            {
                "policy_name": "uniform_step_balanced",
                "reveal_seed": 1,
                "budget": 45,
                "squared_wasserstein2_to_full_reference": 1.0,
                "step_count_l1_from_perfect_balance": 0.0,
                "max_min_revealed_rows_per_step_ratio": 1.0,
                "min_revealed_rows_per_step": 5,
                "max_revealed_rows_per_step": 5,
                "revealed_rows_by_step": json.dumps({"a": 5, "b": 5}),
            },
            {
                "policy_name": "uniform_step_balanced",
                "reveal_seed": 2,
                "budget": 45,
                "squared_wasserstein2_to_full_reference": 3.0,
                "step_count_l1_from_perfect_balance": 0.0,
                "max_min_revealed_rows_per_step_ratio": 1.0,
                "min_revealed_rows_per_step": 5,
                "max_revealed_rows_per_step": 5,
                "revealed_rows_by_step": json.dumps({"a": 5, "b": 5}),
            },
            {
                "policy_name": "greedy_loo_fragility",
                "reveal_seed": 1,
                "budget": 45,
                "squared_wasserstein2_to_full_reference": 2.0,
                "step_count_l1_from_perfect_balance": 10.0,
                "max_min_revealed_rows_per_step_ratio": 2.0,
                "min_revealed_rows_per_step": 4,
                "max_revealed_rows_per_step": 8,
                "revealed_rows_by_step": json.dumps({"a": 8, "b": 4}),
            },
            {
                "policy_name": "greedy_loo_fragility",
                "reveal_seed": 2,
                "budget": 45,
                "squared_wasserstein2_to_full_reference": 4.0,
                "step_count_l1_from_perfect_balance": 20.0,
                "max_min_revealed_rows_per_step_ratio": 3.0,
                "min_revealed_rows_per_step": 3,
                "max_revealed_rows_per_step": 9,
                "revealed_rows_by_step": json.dumps({"a": 9, "b": 3}),
            },
        ]
    )
    run_b_results = pd.DataFrame(
        [
            {
                "policy_name": "exploration_bonus_c1.0",
                "reveal_seed": 1,
                "budget": 45,
                "squared_wasserstein2_to_full_reference": 0.5,
                "step_count_l1_from_perfect_balance": 6.0,
                "max_min_revealed_rows_per_step_ratio": 1.5,
                "min_revealed_rows_per_step": 4,
                "max_revealed_rows_per_step": 6,
                "revealed_rows_by_step": json.dumps({"a": 6, "b": 4}),
            }
        ]
    )
    return {
        "run_a": {
            "results": run_a_results,
            "auc": pd.DataFrame(
                [
                    {"policy_name": "uniform_step_balanced", "reveal_seed": 1, "auc_distance": 10.0},
                    {"policy_name": "uniform_step_balanced", "reveal_seed": 2, "auc_distance": 12.0},
                    {"policy_name": "greedy_loo_fragility", "reveal_seed": 1, "auc_distance": 20.0},
                    {"policy_name": "greedy_loo_fragility", "reveal_seed": 2, "auc_distance": 22.0},
                ]
            ),
            "differences": pd.DataFrame(
                [
                    {"policy_name": "greedy_loo_fragility", "policy_better": False},
                    {"policy_name": "greedy_loo_fragility", "policy_better": True},
                ]
            ),
            "concentration": pd.DataFrame(
                [
                    {
                        "policy_name": "uniform_step_balanced",
                        "budget": 45,
                        "mean_step_count_l1_imbalance": 0.0,
                        "mean_max_min_revealed_row_ratio": 1.0,
                    },
                    {
                        "policy_name": "greedy_loo_fragility",
                        "budget": 45,
                        "mean_step_count_l1_imbalance": 15.0,
                        "mean_max_min_revealed_row_ratio": 2.5,
                    },
                ]
            ),
            "report": {"settings": {"fragility_kwargs": {"max_loo_terms_per_step": 20}}},
        },
        "run_b": {
            "results": run_b_results,
            "auc": pd.DataFrame(
                [
                    {"policy_name": "exploration_bonus_c1.0", "reveal_seed": 1, "auc_distance": 5.0},
                ]
            ),
            "differences": pd.DataFrame(
                [
                    {"policy_name": "exploration_bonus_c1.0", "policy_better": True},
                ]
            ),
            "concentration": pd.DataFrame(
                [
                    {
                        "policy_name": "exploration_bonus_c1.0",
                        "budget": 45,
                        "mean_step_count_l1_imbalance": 6.0,
                        "mean_max_min_revealed_row_ratio": 1.5,
                    },
                ]
            ),
            "report": {"settings": {"fragility_kwargs": {"max_loo_terms_per_step": 20}}},
        },
    }


def test_markdown_table_writing_works(tmp_path) -> None:
    df = pd.DataFrame([{"a": 1, "b": 0.000001}])
    path = tmp_path / "table.md"
    write_markdown_table(df, path)
    assert "| a | b |" in path.read_text(encoding="utf-8")
    assert "1.000000e-06" in dataframe_to_markdown(df)


def test_main_policy_comparison_preserves_source_run() -> None:
    sources = [
        {"policy": "uniform_step_balanced", "source_run": "run_a"},
        {"policy": "exploration_bonus_c1.0", "source_run": "run_b"},
    ]
    table = build_main_policy_comparison(_run_outputs(), sources, max_budget=45)
    assert set(table["source_run"]) == {"run_a", "run_b"}
    assert set(table["policy"]) == {"uniform_step_balanced", "exploration_bonus_c1.0"}


def test_error_by_budget_aggregation_computes_mean_and_quantiles() -> None:
    sources = [{"policy": "uniform_step_balanced", "source_run": "run_a"}]
    table = build_error_by_budget(_run_outputs(), sources)
    row = table.iloc[0]
    assert row["mean_squared_w2_error"] == 2.0
    assert row["median_squared_w2_error"] == 2.0
    assert row["p25_squared_w2_error"] == 1.5
    assert row["p75_squared_w2_error"] == 2.5


def test_concentration_table_handles_missing_optional_columns_gracefully() -> None:
    outputs = {
        "run": {
            "results": pd.DataFrame(
                [
                    {
                        "policy_name": "p",
                        "reveal_seed": 1,
                        "budget": 45,
                        "step_count_l1_from_perfect_balance": 1.0,
                        "max_min_revealed_rows_per_step_ratio": 2.0,
                    }
                ]
            )
        }
    }
    table = build_concentration_by_budget(outputs, [{"policy": "p", "source_run": "run"}])
    row = table.iloc[0]
    assert row["mean_l1_imbalance"] == 1.0
    assert np.isnan(row["mean_minimum_step_count"])
    assert np.isnan(row["mean_steps_still_at_initial_seed_count"])


def test_combining_policies_from_different_runs_keeps_source_labels() -> None:
    sources = [
        {"policy": "greedy_loo_fragility", "source_run": "run_a"},
        {"policy": "exploration_bonus_c1.0", "source_run": "run_b"},
    ]
    table = build_error_by_budget(_run_outputs(), sources)
    assert set(table["source_run"]) == {"run_a", "run_b"}


def test_summary_functions_do_not_silently_drop_policies() -> None:
    sources = [
        {"policy": "uniform_step_balanced", "source_run": "run_a"},
        {"policy": "greedy_loo_fragility", "source_run": "run_a"},
        {"policy": "exploration_bonus_c1.0", "source_run": "run_b"},
    ]
    table = build_main_policy_comparison(_run_outputs(), sources, max_budget=45)
    assert list(table["policy"]) == [
        "uniform_step_balanced",
        "greedy_loo_fragility",
        "exploration_bonus_c1.0",
    ]
