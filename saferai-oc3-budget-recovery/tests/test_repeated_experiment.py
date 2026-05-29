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
