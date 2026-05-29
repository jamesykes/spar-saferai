from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saferai_budget_recovery import config
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


def test_tiny_synthetic_experiment_returns_requested_budgets_for_both_policies() -> None:
    fit_df = _complete_fit_df()
    for policy in ["uniform_step_balanced", "greedy_loo_fragility"]:
        results, diagnostics = run_policy_recovery(
            fit_df,
            policy_name=policy,
            budgets=[45, 50],
            reveal_seed=11,
            reference_seed=22,
            sample_seed_base=33,
            n_reference_samples=60,
            n_budget_samples=40,
            n_grid=21,
            fragility_kwargs={"n_samples": 20, "n_grid": 11},
            fragility_recompute_every=5,
        )
        assert list(results["budget"]) == [45, 50]
        assert diagnostics["initial_seed_size"] == 45
        assert diagnostics["total_usable_fitted_rows"] == 90


def test_tiny_synthetic_experiment_budget_accounting_and_distances() -> None:
    results, _ = run_policy_recovery(
        _complete_fit_df(),
        policy_name="uniform_step_balanced",
        budgets=[45, 50],
        reveal_seed=11,
        reference_seed=22,
        sample_seed_base=33,
        n_reference_samples=60,
        n_budget_samples=40,
        n_grid=21,
    )
    assert list(results["additional_rows_beyond_initial_seed"]) == [0, 5]
    assert np.isfinite(results["squared_wasserstein2_to_full_reference"]).all()
    assert (results["squared_wasserstein2_to_full_reference"] >= 0).all()
    for col in [
        "max_min_revealed_rows_per_step_ratio",
        "step_count_std",
        "step_count_l1_from_perfect_balance",
        "revealed_rows_by_step",
    ]:
        assert col in results.columns


def test_tiny_synthetic_experiment_is_reproducible_with_same_seeds() -> None:
    kwargs = dict(
        fit_df=_complete_fit_df(),
        policy_name="uniform_step_balanced",
        budgets=[45, 50],
        reveal_seed=11,
        reference_seed=22,
        sample_seed_base=33,
        n_reference_samples=60,
        n_budget_samples=40,
        n_grid=21,
    )
    first, _ = run_policy_recovery(**kwargs)
    second, _ = run_policy_recovery(**kwargs)
    assert np.allclose(
        first["squared_wasserstein2_to_full_reference"],
        second["squared_wasserstein2_to_full_reference"],
    )


def test_uniform_row_random_policy_name_is_available_as_diagnostic() -> None:
    results, _ = run_policy_recovery(
        _complete_fit_df(),
        policy_name="uniform_row_random",
        budgets=[45, 50],
        reveal_seed=11,
        reference_seed=22,
        sample_seed_base=33,
        n_reference_samples=60,
        n_budget_samples=40,
        n_grid=21,
    )
    assert list(results["budget"]) == [45, 50]


def test_old_uniform_policy_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown policy_name='uniform'"):
        run_policy_recovery(
            _complete_fit_df(),
            policy_name="uniform",
            budgets=[45, 50],
            reveal_seed=11,
            reference_seed=22,
            sample_seed_base=33,
            n_reference_samples=60,
            n_budget_samples=40,
            n_grid=21,
        )
