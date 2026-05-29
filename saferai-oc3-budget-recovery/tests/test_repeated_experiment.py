from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from saferai_budget_recovery import config
from saferai_budget_recovery.analysis import (
    compute_auc_by_seed_policy,
    compute_policy_differences,
    summarize_policy_results,
)
from saferai_budget_recovery.experiment import run_policy_recovery


def _complete_fit_df(rows_per_group: int = 2) -> pd.DataFrame:
    rows = []
    for step_i, step in enumerate(config.EXPECTED_MITRE_STEP_LABELS):
        for model_i, model in enumerate(config.EXPECTED_LLM_MODELS):
            for repeat in range(rows_per_group):
                rows.append(
                    {
                        "step_name": step,
                        "model": model,
                        "run_id": f"run-{repeat}",
                        "repeat_index": repeat,
                        "draw_uid": f"{model}__run-{repeat}__{repeat}",
                        "alpha": 2.0 + 0.1 * step_i,
                        "beta": 5.0 + 0.1 * model_i,
                        "fit_quality_flag": "ok",
                    }
                )
    return pd.DataFrame(rows)


def test_repeated_experiment_aggregation_on_multiple_synthetic_seeds() -> None:
    fit_df = _complete_fit_df()
    all_results = []
    for seed in [11, 22]:
        for policy in ["uniform_step_balanced", "greedy_loo_fragility"]:
            results, _ = run_policy_recovery(
                fit_df,
                policy_name=policy,
                budgets=[45, 50],
                reveal_seed=seed,
                reference_seed=100 + seed,
                sample_seed_base=200,
                n_reference_samples=50,
                n_budget_samples=30,
                n_grid=11,
                fragility_kwargs={"n_samples": 15, "n_grid": 7},
                fragility_recompute_every=5,
            )
            all_results.append(results)
    combined = pd.concat(all_results, ignore_index=True)
    summary = summarize_policy_results(combined)
    diff = compute_policy_differences(combined)
    auc = compute_auc_by_seed_policy(combined)

    assert {"policy_name", "budget", "distance_mean", "n_runs"}.issubset(summary.columns)
    assert {"reveal_seed", "budget", "greedy_better"}.issubset(diff.columns)
    assert {"policy_name", "reveal_seed", "auc_distance"}.issubset(auc.columns)
    assert np.isfinite(combined["squared_wasserstein2_to_full_reference"]).all()
    assert (combined["squared_wasserstein2_to_full_reference"] >= 0).all()


def test_repeated_experiment_script_exposes_fast_dev_10_seeds_mode() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "07_run_repeated_policy_experiment.py"
    spec = importlib.util.spec_from_file_location("run_repeated_policy_experiment", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = module.CONFIGURATIONS["FAST_DEV_10_SEEDS"]
    assert settings["policies"] == ["uniform_step_balanced", "greedy_loo_fragility"]
    assert settings["fragility_kwargs"]["max_loo_terms_per_step"] == 20
    assert 1200 in settings["budgets"]


def test_repeated_experiment_script_exposes_v8_all_policies_mode() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "07_run_repeated_policy_experiment.py"
    spec = importlib.util.spec_from_file_location("run_repeated_policy_experiment_v8", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = module.CONFIGURATIONS["V8_ALL_POLICIES_DEV"]
    policy_names = {spec["name"] for spec in settings["policy_specs"]}
    assert "epsilon_greedy_eps0.2" in policy_names
    assert "exploration_bonus_c0.5" in policy_names
    assert settings["fragility_kwargs"]["max_loo_terms_per_step"] == 20
    assert module.MODE in module.CONFIGURATIONS


def test_repeated_experiment_script_exposes_exploration_bonus_sensitivity_mode() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "07_run_repeated_policy_experiment.py"
    spec = importlib.util.spec_from_file_location("run_repeated_policy_experiment_bonus", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = module.CONFIGURATIONS["EXPLORATION_BONUS_SENSITIVITY_DEV"]
    policy_names = {spec["name"] for spec in settings["policy_specs"]}
    assert policy_names == {
        "uniform_step_balanced",
        "exploration_bonus_c0.25",
        "exploration_bonus_c0.5",
        "exploration_bonus_c1.0",
    }
    assert settings["n_reference_samples"] == 40_000
    assert settings["n_budget_samples"] == 12_000
    assert settings["fragility_kwargs"]["max_loo_terms_per_step"] == 20
    assert module.MODE == "EXPLORATION_BONUS_SENSITIVITY_DEV"


def test_exploration_bonus_c_value_summary() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "07_run_repeated_policy_experiment.py"
    spec = importlib.util.spec_from_file_location("run_repeated_policy_experiment_summary", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    results = pd.DataFrame(
        [
            {
                "policy_name": "exploration_bonus_c0.25",
                "reveal_seed": 1,
                "budget": 1200,
                "squared_wasserstein2_to_full_reference": 2.0,
                "min_revealed_rows_per_step": 10,
                "max_revealed_rows_per_step": 30,
            },
            {
                "policy_name": "exploration_bonus_c0.5",
                "reveal_seed": 1,
                "budget": 1200,
                "squared_wasserstein2_to_full_reference": 1.0,
                "min_revealed_rows_per_step": 12,
                "max_revealed_rows_per_step": 25,
            },
            {
                "policy_name": "uniform_step_balanced",
                "reveal_seed": 1,
                "budget": 1200,
                "squared_wasserstein2_to_full_reference": 1.5,
                "min_revealed_rows_per_step": 20,
                "max_revealed_rows_per_step": 21,
            },
        ]
    )
    differences = pd.DataFrame(
        [
            {"policy_name": "exploration_bonus_c0.25", "policy_better": False},
            {"policy_name": "exploration_bonus_c0.5", "policy_better": True},
        ]
    )
    auc = pd.DataFrame(
        [
            {"policy_name": "exploration_bonus_c0.25", "auc_distance": 20.0},
            {"policy_name": "exploration_bonus_c0.5", "auc_distance": 10.0},
        ]
    )
    concentration = pd.DataFrame(
        [
            {
                "policy_name": "exploration_bonus_c0.25",
                "budget": 1200,
                "mean_step_count_l1_imbalance": 30.0,
                "median_step_count_l1_imbalance": 30.0,
                "max_step_count_l1_imbalance": 30.0,
                "mean_max_min_revealed_row_ratio": 3.0,
            },
            {
                "policy_name": "exploration_bonus_c0.5",
                "budget": 1200,
                "mean_step_count_l1_imbalance": 20.0,
                "median_step_count_l1_imbalance": 20.0,
                "max_step_count_l1_imbalance": 20.0,
                "mean_max_min_revealed_row_ratio": 2.0,
            },
        ]
    )
    summary = module._exploration_bonus_c_value_summary(
        results_df=results,
        differences_df=differences,
        auc_df=auc,
        concentration_df=concentration,
        max_budget=1200,
    )

    assert list(summary["c"]) == [0.25, 0.5]
    row = summary.loc[summary["c"].eq(0.5)].iloc[0]
    assert row["average_auc"] == 10.0
    assert row["win_fraction_vs_uniform"] == 1.0
    assert row["mean_l1_imbalance_at_1200"] == 20.0
    notes = module._exploration_bonus_sensitivity_notes(summary)
    assert notes["lowest_average_auc_c"] == 0.5
