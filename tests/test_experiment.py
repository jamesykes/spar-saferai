from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saferai_budget_recovery import config
from saferai_budget_recovery.distances import empirical_quantiles, quantile_grid
from saferai_budget_recovery.experiment import run_policy_recovery
from saferai_budget_recovery.sampling import sample_full_reference_p_success


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
        assert diagnostics["uses_hidden_reveal_orders"] is True


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


def test_hidden_reveal_order_diagnostics_are_present() -> None:
    _, diagnostics = run_policy_recovery(
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
    assert diagnostics["uses_hidden_reveal_orders"] is True
    assert diagnostics["hidden_reveal_order_metadata"]["uses_model_aware_cycling"] is True
    assert diagnostics["hidden_reveal_order_coverage"]["covers_all_usable_rows_exactly_once"] is True
    assert len(diagnostics["hidden_reveal_order_post_seed_lengths_by_step"]) == 9


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


def test_reference_samples_and_quantiles_can_be_reused() -> None:
    fit_df = _complete_fit_df()
    grid = quantile_grid(21)
    reference_samples = sample_full_reference_p_success(fit_df, n_samples=60, seed=22)[
        "p_success"
    ].to_numpy()
    reference_quantiles = empirical_quantiles(reference_samples, grid)
    results, diagnostics = run_policy_recovery(
        fit_df,
        policy_name="uniform_step_balanced",
        budgets=[45, 50],
        reveal_seed=11,
        reference_seed=22,
        sample_seed_base=33,
        n_reference_samples=60,
        n_budget_samples=40,
        n_grid=21,
        reference_samples=reference_samples,
        reference_quantiles=reference_quantiles,
    )
    assert diagnostics["reference_reused"] is True
    assert diagnostics["reference_sampling_seconds"] < 0.1
    assert np.isfinite(results["squared_wasserstein2_to_full_reference"]).all()


def test_runtime_metadata_fields_are_present() -> None:
    _, diagnostics = run_policy_recovery(
        _complete_fit_df(),
        policy_name="greedy_loo_fragility",
        budgets=[45, 50],
        reveal_seed=11,
        reference_seed=22,
        sample_seed_base=33,
        n_reference_samples=60,
        n_budget_samples=40,
        n_grid=21,
        fragility_kwargs={"n_samples": 15, "n_grid": 7, "max_loo_terms_per_step": 2},
        fragility_recompute_every=5,
    )
    for key in [
        "reference_sampling_seconds",
        "budget_sampling_seconds",
        "fragility_runtime_seconds",
        "fragility_recomputation_count",
        "avg_fragility_recompute_seconds",
        "total_loo_terms_available",
        "total_loo_terms_used",
    ]:
        assert key in diagnostics


def test_approximate_fragility_settings_are_recorded_in_diagnostics() -> None:
    _, diagnostics = run_policy_recovery(
        _complete_fit_df(),
        policy_name="greedy_loo_fragility",
        budgets=[45, 50],
        reveal_seed=11,
        reference_seed=22,
        sample_seed_base=33,
        n_reference_samples=60,
        n_budget_samples=40,
        n_grid=21,
        fragility_kwargs={"n_samples": 15, "n_grid": 7, "max_loo_terms_per_step": 2},
        fragility_recompute_every=5,
    )
    assert diagnostics["fragility_kwargs"]["max_loo_terms_per_step"] == 2
    assert diagnostics["any_loo_subsampled"] is True
    assert diagnostics["total_loo_terms_used"] <= diagnostics["total_loo_terms_available"]


def test_parameterized_policy_label_is_preserved_in_results_and_diagnostics() -> None:
    results, diagnostics = run_policy_recovery(
        _complete_fit_df(),
        policy_name="epsilon_greedy_loo_fragility",
        policy_label="epsilon_greedy_eps0.2",
        policy_kwargs={"epsilon": 0.2},
        budgets=[45, 50],
        reveal_seed=11,
        reference_seed=22,
        sample_seed_base=33,
        n_reference_samples=60,
        n_budget_samples=40,
        n_grid=21,
        fragility_kwargs={"n_samples": 15, "n_grid": 7, "max_loo_terms_per_step": 2},
        fragility_recompute_every=5,
    )
    assert set(results["policy_name"]) == {"epsilon_greedy_eps0.2"}
    assert diagnostics["policy_name"] == "epsilon_greedy_eps0.2"
    assert diagnostics["base_policy_name"] == "epsilon_greedy_loo_fragility"
    assert diagnostics["policy_kwargs"]["epsilon"] == 0.2
    assert "decision_type_counts" in diagnostics


def test_exploration_bonus_policy_runs_in_tiny_synthetic_experiment() -> None:
    results, diagnostics = run_policy_recovery(
        _complete_fit_df(),
        policy_name="exploration_bonus_loo_fragility",
        policy_label="exploration_bonus_c0.5",
        policy_kwargs={"c": 0.5},
        budgets=[45, 50],
        reveal_seed=11,
        reference_seed=22,
        sample_seed_base=33,
        n_reference_samples=60,
        n_budget_samples=40,
        n_grid=21,
        fragility_kwargs={"n_samples": 15, "n_grid": 7, "max_loo_terms_per_step": 2},
        fragility_recompute_every=5,
    )
    assert list(results["budget"]) == [45, 50]
    assert set(results["policy_name"]) == {"exploration_bonus_c0.5"}
    assert diagnostics["policy_kwargs"]["c"] == 0.5
    assert np.isfinite(results["squared_wasserstein2_to_full_reference"]).all()


def test_all_primary_parameterized_policy_specs_run_with_hidden_orders() -> None:
    specs = [
        ("uniform_step_balanced", "uniform_step_balanced", {}),
        ("greedy_loo_fragility", "greedy_loo_fragility", {}),
        ("epsilon_greedy_loo_fragility", "epsilon_greedy_eps0.2", {"epsilon": 0.2}),
        ("exploration_bonus_loo_fragility", "exploration_bonus_c1.0", {"c": 1.0}),
    ]
    for policy_name, label, kwargs in specs:
        results, diagnostics = run_policy_recovery(
            _complete_fit_df(),
            policy_name=policy_name,
            policy_label=label,
            policy_kwargs=kwargs,
            budgets=[45, 50],
            reveal_seed=11,
            reference_seed=22,
            sample_seed_base=33,
            n_reference_samples=60,
            n_budget_samples=40,
            n_grid=21,
            fragility_kwargs={"n_samples": 15, "n_grid": 7, "max_loo_terms_per_step": 2},
            fragility_recompute_every=5,
        )
        assert set(results["policy_name"]) == {label}
        assert diagnostics["uses_hidden_reveal_orders"] is True
        assert np.isfinite(results["squared_wasserstein2_to_full_reference"]).all()
