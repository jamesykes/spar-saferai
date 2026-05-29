"""Run a repeated development experiment for the two primary policies."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saferai_budget_recovery import config
from saferai_budget_recovery.analysis import (
    DISTANCE_COL,
    compute_auc_by_seed_policy,
    compute_policy_differences,
    compute_policy_differences_against_baseline,
    summarize_policy_results,
    summarize_concentration_by_budget,
    summarize_policy_win_rate_by_budget,
    summarize_policy_win_rate_by_seed,
    summarize_win_rate_by_budget,
    summarize_win_rate_by_seed,
)
from saferai_budget_recovery.distances import empirical_quantiles, quantile_grid
from saferai_budget_recovery.experiment import run_policy_recovery
from saferai_budget_recovery.reveal import usable_fit_rows
from saferai_budget_recovery.sampling import sample_full_reference_p_success


OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "repeated_policy_experiment"

CONFIGURATIONS = {
    "FAST_DEV": {
        "budgets": [45, 90, 180, 360, 720],
        "reveal_seeds": [101, 202, 303, 404, 505],
        "policies": ["uniform_step_balanced", "greedy_loo_fragility"],
        "n_reference_samples": 30_000,
        "n_budget_samples": 10_000,
        "n_grid": 401,
        "fragility_kwargs": {
            "n_samples": 500,
            "n_grid": 151,
            "max_loo_terms_per_step": 20,
        },
        "fragility_recompute_every": 90,
        "reference_seed": 907000,
        "sample_seed_base": 1008000,
    },
    "FAST_DEV_10_SEEDS": {
        "budgets": [45, 90, 180, 360, 720, 1200],
        "reveal_seeds": [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010],
        "policies": ["uniform_step_balanced", "greedy_loo_fragility"],
        "n_reference_samples": 40_000,
        "n_budget_samples": 12_000,
        "n_grid": 401,
        "fragility_kwargs": {
            "n_samples": 600,
            "n_grid": 151,
            "max_loo_terms_per_step": 20,
        },
        "fragility_recompute_every": 90,
        "reference_seed": 907000,
        "sample_seed_base": 1008000,
    },
    "V8_ALL_POLICIES_DEV": {
        "budgets": [45, 90, 180, 360, 720, 1200],
        "reveal_seeds": [101, 202, 303, 404, 505],
        "policy_specs": [
            {"name": "uniform_step_balanced", "policy_name": "uniform_step_balanced", "policy_kwargs": {}},
            {"name": "greedy_loo_fragility", "policy_name": "greedy_loo_fragility", "policy_kwargs": {}},
            {
                "name": "epsilon_greedy_eps0.2",
                "policy_name": "epsilon_greedy_loo_fragility",
                "policy_kwargs": {"epsilon": 0.2},
            },
            {
                "name": "exploration_bonus_c0.5",
                "policy_name": "exploration_bonus_loo_fragility",
                "policy_kwargs": {"c": 0.5},
            },
        ],
        "n_reference_samples": 40_000,
        "n_budget_samples": 12_000,
        "n_grid": 401,
        "fragility_kwargs": {
            "n_samples": 600,
            "n_grid": 151,
            "max_loo_terms_per_step": 20,
        },
        "fragility_recompute_every": 90,
        "reference_seed": 907000,
        "sample_seed_base": 1008000,
        "requested_settings": {
            "budgets": [45, 90, 180, 360, 720, 1200],
            "reveal_seeds": [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010],
            "policy_specs": [
                {"name": "uniform_step_balanced", "policy_name": "uniform_step_balanced", "policy_kwargs": {}},
                {"name": "greedy_loo_fragility", "policy_name": "greedy_loo_fragility", "policy_kwargs": {}},
                {
                    "name": "epsilon_greedy_eps0.2",
                    "policy_name": "epsilon_greedy_loo_fragility",
                    "policy_kwargs": {"epsilon": 0.2},
                },
                {
                    "name": "exploration_bonus_c0.25",
                    "policy_name": "exploration_bonus_loo_fragility",
                    "policy_kwargs": {"c": 0.25},
                },
                {
                    "name": "exploration_bonus_c0.5",
                    "policy_name": "exploration_bonus_loo_fragility",
                    "policy_kwargs": {"c": 0.5},
                },
                {
                    "name": "exploration_bonus_c1.0",
                    "policy_name": "exploration_bonus_loo_fragility",
                    "policy_kwargs": {"c": 1.0},
                },
            ],
            "n_reference_samples": 40_000,
            "n_budget_samples": 12_000,
            "n_grid": 401,
            "fragility_kwargs": {
                "n_samples": 600,
                "n_grid": 151,
                "max_loo_terms_per_step": 20,
            },
            "fragility_recompute_every": 90,
            "reference_seed": 907000,
            "sample_seed_base": 1008000,
        },
        "runtime_reduction_notes": [
            "Reduced reveal seeds from 10 to 5 after the unreduced all-policy run projected excessive runtime.",
            "Removed exploration_bonus_c0.25 and exploration_bonus_c1.0 after the 5-seed all-c-values run was still too slow; kept exploration_bonus_c0.5.",
        ],
    },
    "EXPLORATION_BONUS_SENSITIVITY_DEV": {
        "budgets": [45, 90, 180, 360, 720, 1200],
        "reveal_seeds": [101, 202, 303, 404, 505],
        "policy_specs": [
            {"name": "uniform_step_balanced", "policy_name": "uniform_step_balanced", "policy_kwargs": {}},
            {
                "name": "exploration_bonus_c0.25",
                "policy_name": "exploration_bonus_loo_fragility",
                "policy_kwargs": {"c": 0.25},
            },
            {
                "name": "exploration_bonus_c0.5",
                "policy_name": "exploration_bonus_loo_fragility",
                "policy_kwargs": {"c": 0.5},
            },
            {
                "name": "exploration_bonus_c1.0",
                "policy_name": "exploration_bonus_loo_fragility",
                "policy_kwargs": {"c": 1.0},
            },
        ],
        "n_reference_samples": 40_000,
        "n_budget_samples": 12_000,
        "n_grid": 401,
        "fragility_kwargs": {
            "n_samples": 600,
            "n_grid": 151,
            "max_loo_terms_per_step": 20,
        },
        "fragility_recompute_every": 90,
        "reference_seed": 907000,
        "sample_seed_base": 1008000,
    },
    "MORE_EXACT_DEV": {
        "budgets": [45, 90, 180, 360, 720],
        "reveal_seeds": [101, 202, 303],
        "policies": ["uniform_step_balanced", "greedy_loo_fragility"],
        "n_reference_samples": 50_000,
        "n_budget_samples": 20_000,
        "n_grid": 501,
        "fragility_kwargs": {
            "n_samples": 1000,
            "n_grid": 201,
            "max_loo_terms_per_step": None,
        },
        "fragility_recompute_every": 90,
        "reference_seed": 907000,
        "sample_seed_base": 1008000,
    },
}
MODE = os.environ.get("SAFERAI_EXPERIMENT_MODE", "EXPLORATION_BONUS_SENSITIVITY_DEV")


def main() -> None:
    if not config.SOTA_BETA_FITS_PATH.exists():
        raise FileNotFoundError(
            f"SOTA Beta fits not found: {config.SOTA_BETA_FITS_PATH}. "
            "Run scripts/02_fit_beta_distributions.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MODE not in CONFIGURATIONS:
        raise ValueError(f"Unknown MODE={MODE!r}; expected one of {sorted(CONFIGURATIONS)}.")
    raw_settings = CONFIGURATIONS[MODE]
    requested_settings = json.loads(json.dumps(raw_settings.get("requested_settings", raw_settings)))
    settings = {
        key: value
        for key, value in raw_settings.items()
        if key not in {"requested_settings", "runtime_reduction_notes"}
    }
    runtime_reduction_notes = list(raw_settings.get("runtime_reduction_notes", []))
    policy_specs = _policy_specs(settings)
    prefix = MODE.lower()

    start = time.perf_counter()
    fit_df = pd.read_csv(config.SOTA_BETA_FITS_PATH)
    usable_df = usable_fit_rows(fit_df)
    total_usable = len(usable_df)
    grid = quantile_grid(settings["n_grid"])

    print("Repeated policy experiment")
    print(f"Mode: {MODE}")
    print(f"Settings: {settings}")
    print(f"Policy specs: {policy_specs}")
    ref_start = time.perf_counter()
    reference_df = sample_full_reference_p_success(
        usable_df,
        n_samples=settings["n_reference_samples"],
        seed=settings["reference_seed"],
    )
    reference_samples = reference_df["p_success"].to_numpy(dtype=float)
    reference_quantiles = empirical_quantiles(reference_samples, grid)
    reference_sampling_seconds = time.perf_counter() - ref_start
    print(f"Full reference sampled once in {reference_sampling_seconds:.2f}s")

    all_results: list[pd.DataFrame] = []
    run_diagnostics: list[dict[str, Any]] = []
    fragility_runtime_rows: list[dict[str, Any]] = []

    for reveal_seed in settings["reveal_seeds"]:
        for spec in policy_specs:
            run_start = time.perf_counter()
            print(f"Running reveal_seed={reveal_seed}, policy={spec['name']}", flush=True)
            result_df, diagnostics = run_policy_recovery(
                fit_df,
                policy_name=spec["policy_name"],
                policy_label=spec["name"],
                policy_kwargs=spec.get("policy_kwargs", {}),
                budgets=settings["budgets"],
                reveal_seed=reveal_seed,
                reference_seed=settings["reference_seed"],
                sample_seed_base=settings["sample_seed_base"],
                n_reference_samples=settings["n_reference_samples"],
                n_budget_samples=settings["n_budget_samples"],
                n_grid=settings["n_grid"],
                fragility_kwargs=settings["fragility_kwargs"],
                fragility_recompute_every=settings["fragility_recompute_every"],
                reference_samples=reference_samples,
                reference_quantiles=reference_quantiles,
            )
            elapsed = time.perf_counter() - run_start
            print(f"  finished in {elapsed:.2f}s", flush=True)
            all_results.append(result_df)
            run_diagnostics.append(diagnostics)
            fragility_runtime_rows.extend(diagnostics.get("fragility_runtime_diagnostics", []))

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = summarize_policy_results(results_df)
    legacy_greedy_differences_df = (
        compute_policy_differences(results_df)
        if {"uniform_step_balanced", "greedy_loo_fragility"}.issubset(set(results_df["policy_name"]))
        else pd.DataFrame()
    )
    differences_df = compute_policy_differences_against_baseline(results_df)
    auc_df = compute_auc_by_seed_policy(results_df)
    win_by_budget_df = summarize_policy_win_rate_by_budget(differences_df)
    win_by_seed_df = summarize_policy_win_rate_by_seed(differences_df, auc_df)
    concentration_df = summarize_concentration_by_budget(results_df)
    selected_step_counts_df = _selected_step_counts_df(run_diagnostics)
    fragility_runtime_df = pd.DataFrame(fragility_runtime_rows)
    c_value_summary_df = _exploration_bonus_c_value_summary(
        results_df=results_df,
        differences_df=differences_df,
        auc_df=auc_df,
        concentration_df=concentration_df,
        max_budget=max(settings["budgets"]),
    )

    paths = {
        "results": OUTPUT_DIR / f"{prefix}_repeated_policy_results.csv",
        "summary": OUTPUT_DIR / f"{prefix}_policy_summary_by_budget.csv",
        "differences": OUTPUT_DIR / f"{prefix}_policy_differences_vs_uniform.csv",
        "auc": OUTPUT_DIR / f"{prefix}_policy_auc_by_seed.csv",
        "win_by_budget": OUTPUT_DIR / f"{prefix}_win_rate_by_budget.csv",
        "win_by_seed": OUTPUT_DIR / f"{prefix}_win_rate_by_seed.csv",
        "concentration": OUTPUT_DIR / f"{prefix}_concentration_by_budget.csv",
        "selected_step_counts": OUTPUT_DIR / f"{prefix}_selected_step_counts.csv",
        "c_value_summary": OUTPUT_DIR / f"{prefix}_c_value_summary.csv",
        "report": OUTPUT_DIR / f"{prefix}_repeated_policy_experiment_report.json",
        "fragility_runtime": OUTPUT_DIR / f"{prefix}_fragility_runtime_diagnostics.csv",
    }
    results_df.to_csv(paths["results"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    differences_df.to_csv(paths["differences"], index=False)
    auc_df.to_csv(paths["auc"], index=False)
    win_by_budget_df.to_csv(paths["win_by_budget"], index=False)
    win_by_seed_df.to_csv(paths["win_by_seed"], index=False)
    concentration_df.to_csv(paths["concentration"], index=False)
    selected_step_counts_df.to_csv(paths["selected_step_counts"], index=False)
    c_value_summary_df.to_csv(paths["c_value_summary"], index=False)
    fragility_runtime_df.to_csv(paths["fragility_runtime"], index=False)

    runtime = time.perf_counter() - start
    report = _make_report(
        mode=MODE,
        settings=settings,
        requested_settings=requested_settings,
        runtime_reduction_notes=runtime_reduction_notes,
        policy_specs=policy_specs,
        results_df=results_df,
        summary_df=summary_df,
        differences_df=differences_df,
        legacy_greedy_differences_df=legacy_greedy_differences_df,
        auc_df=auc_df,
        win_by_budget_df=win_by_budget_df,
        win_by_seed_df=win_by_seed_df,
        concentration_df=concentration_df,
        selected_step_counts_df=selected_step_counts_df,
        c_value_summary_df=c_value_summary_df,
        run_diagnostics=run_diagnostics,
        fragility_runtime_df=fragility_runtime_df,
        runtime=runtime,
        reference_sampling_seconds=reference_sampling_seconds,
        total_usable=total_usable,
    )
    _write_json(paths["report"], report)

    distance_table = summary_df.pivot(index="budget", columns="policy_name", values="distance_mean")
    auc_summary = auc_df.groupby("policy_name")["auc_distance"].agg(["mean", "median"])
    print("Summary distance table (mean across reveal seeds):")
    print(distance_table.to_string())
    print("Win fraction vs uniform by policy and budget:")
    print(
        win_by_budget_df[
            ["policy_name", "budget", "policy_win_fraction", "mean_policy_minus_baseline_distance"]
        ].to_string(index=False)
    )
    print("AUC summary:")
    print(auc_summary.to_string())
    if not c_value_summary_df.empty:
        print("Exploration-bonus c-value summary:")
        print(
            c_value_summary_df[
                [
                    "c",
                    "average_auc",
                    "win_fraction_vs_uniform",
                    "mean_l1_imbalance_at_1200",
                    "mean_max_min_ratio_at_1200",
                ]
            ].to_string(index=False)
        )
    largest_budget = max(settings["budgets"])
    largest_concentration = concentration_df.loc[concentration_df["budget"].eq(largest_budget)]
    if not largest_concentration.empty:
        print(f"Concentration summary at largest budget ({largest_budget}):")
        print(
            largest_concentration[
                [
                    "policy_name",
                    "mean_step_count_l1_imbalance",
                    "max_step_count_l1_imbalance",
                    "mean_max_min_revealed_row_ratio",
                    "maximum_observed_revealed_row_count_any_step",
                    "minimum_observed_revealed_row_count_any_step",
                ]
            ].to_string(index=False)
        )
    print(f"Runtime seconds: {runtime:.2f}")
    for label, path in paths.items():
        print(f"{label}: {path}")


def _make_report(
    mode: str,
    settings: dict[str, Any],
    requested_settings: dict[str, Any],
    runtime_reduction_notes: list[str],
    policy_specs: list[dict[str, Any]],
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    differences_df: pd.DataFrame,
    legacy_greedy_differences_df: pd.DataFrame,
    auc_df: pd.DataFrame,
    win_by_budget_df: pd.DataFrame,
    win_by_seed_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    selected_step_counts_df: pd.DataFrame,
    c_value_summary_df: pd.DataFrame,
    run_diagnostics: list[dict[str, Any]],
    fragility_runtime_df: pd.DataFrame,
    runtime: float,
    reference_sampling_seconds: float,
    total_usable: int,
) -> dict[str, Any]:
    policy_summary: dict[str, Any] = {}
    for policy, group in results_df.groupby("policy_name"):
        auc_group = auc_df.loc[auc_df["policy_name"].eq(policy)]
        policy_summary[str(policy)] = {
            "average_distance_across_all_budgets_and_seeds": float(
                group["squared_wasserstein2_to_full_reference"].mean()
            ),
            "average_auc": float(auc_group["auc_distance"].mean()),
            "median_auc": float(auc_group["auc_distance"].median()),
        }

    max_terms = settings["fragility_kwargs"].get("max_loo_terms_per_step")
    approximation_note = None
    if max_terms is not None:
        approximation_note = (
            "Fragility-guided policies used approximate LOO-term subsampling with "
            f"max_loo_terms_per_step={max_terms}. Previous approximation audit suggested cap 20 "
            "preserved top fragile-node rankings on reduced audit settings, but this remains an "
            "approximation to the exact v8 LOO definition. Exact LOO remains available by setting "
            "this to null."
        )
    uniform_auc_mean = float(
        auc_df.loc[auc_df["policy_name"].eq("uniform_step_balanced"), "auc_distance"].mean()
    )
    auc_by_policy = auc_df.groupby("policy_name")["auc_distance"].mean().to_dict()
    best_auc_policy = min(auc_by_policy, key=auc_by_policy.get)
    overall_win_rates = {
        str(policy): float(group["policy_better"].mean())
        for policy, group in differences_df.groupby("policy_name")
    }

    return {
        "note": (
            "This is a v8 all-policies development run, not the final report run. Fragility-guided "
            "policies use approximate LOO-term subsampling with max_loo_terms_per_step=20. Previous "
            "approximation audit suggested cap 20 preserved top fragile-node rankings on reduced "
            "audit settings, but this remains an approximation to the exact v8 LOO definition."
        ),
        "mode": mode,
        "settings": settings,
        "requested_settings": requested_settings,
        "settings_reduced_from_requested": settings != requested_settings,
        "runtime_reduction_notes": runtime_reduction_notes,
        "policy_specs": policy_specs,
        "approximation_note": approximation_note,
        "runtime_seconds": runtime,
        "reference_design": {
            "reference_sampled_once_globally": True,
            "reference_seed": settings["reference_seed"],
            "reference_sampling_seconds": reference_sampling_seconds,
        },
        "total_usable_fitted_rows": int(total_usable),
        "initial_seed_size": int(results_df["budget"].min()),
        "policies_compared": [spec["name"] for spec in policy_specs],
        "number_of_reveal_seeds": int(len(settings["reveal_seeds"])),
        "budgets": settings["budgets"],
        "runtime_diagnostics": _runtime_diagnostics(run_diagnostics),
        "summary_by_policy": policy_summary,
        "overall_win_rates_vs_uniform": overall_win_rates,
        "greedy_better_count": (
            int(legacy_greedy_differences_df["greedy_better"].sum())
            if not legacy_greedy_differences_df.empty
            else None
        ),
        "total_seed_budget_comparisons": int(len(differences_df)),
        "greedy_better_fraction": (
            float(legacy_greedy_differences_df["greedy_better"].mean())
            if not legacy_greedy_differences_df.empty
            else None
        ),
        "win_rate_by_budget": win_by_budget_df.to_dict(orient="records"),
        "win_rate_by_seed": win_by_seed_df.to_dict(orient="records"),
        "concentration_by_budget": concentration_df.to_dict(orient="records"),
        "selected_step_counts": selected_step_counts_df.to_dict(orient="records"),
        "exploration_bonus_c_value_summary": c_value_summary_df.to_dict(orient="records"),
        "exploration_bonus_sensitivity": _exploration_bonus_sensitivity_notes(c_value_summary_df),
        "fragility_approximation_diagnostics": _fragility_approximation(fragility_runtime_df),
        "interpretation_notes": {
            "best_average_auc_policy": str(best_auc_policy),
            "best_average_auc": float(auc_by_policy[best_auc_policy]),
            "uniform_average_auc": uniform_auc_mean,
            "any_fragility_policy_has_lower_average_auc_than_uniform": bool(
                any(policy != "uniform_step_balanced" and auc < uniform_auc_mean for policy, auc in auc_by_policy.items())
            ),
            "any_policy_wins_majority_vs_uniform": bool(
                any(rate > 0.5 for rate in overall_win_rates.values())
            ),
            "do_not_overclaim": (
                "These summaries are development diagnostics with approximate LOO fragility, "
                "not a final statistical claim."
            ),
        },
        "run_diagnostics": run_diagnostics,
    }


def _runtime_diagnostics(run_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(run_diagnostics)
    return {
        "runtime_seconds_by_policy": _group_sum(df, "policy_name", "runtime_seconds"),
        "runtime_seconds_by_reveal_seed": _group_sum(df, "reveal_seed", "runtime_seconds"),
        "runtime_seconds_by_policy_reveal_seed": {
            f"{row['policy_name']}__{row['reveal_seed']}": float(row["runtime_seconds"])
            for row in df.to_dict(orient="records")
        },
        "fragility_runtime_seconds_total": float(df["fragility_runtime_seconds"].sum()),
        "budget_sampling_seconds_total": float(df["budget_sampling_seconds"].sum()),
        "reference_sampling_seconds_inside_runs_total": float(df["reference_sampling_seconds"].sum()),
        "fragility_recomputation_count": int(df["fragility_recomputation_count"].sum()),
        "avg_fragility_recompute_seconds": (
            float(df["fragility_runtime_seconds"].sum() / df["fragility_recomputation_count"].sum())
            if int(df["fragility_recomputation_count"].sum())
            else 0.0
        ),
        "total_loo_terms_available": int(df["total_loo_terms_available"].sum()),
        "total_loo_terms_used": int(df["total_loo_terms_used"].sum()),
        "loo_terms_used_fraction": (
            float(df["total_loo_terms_used"].sum() / df["total_loo_terms_available"].sum())
            if int(df["total_loo_terms_available"].sum())
            else None
        ),
    }


def _policy_specs(settings: dict[str, Any]) -> list[dict[str, Any]]:
    if "policy_specs" in settings:
        specs = settings["policy_specs"]
    else:
        specs = [
            {"name": policy, "policy_name": policy, "policy_kwargs": {}}
            for policy in settings.get("policies", [])
        ]
    normalized: list[dict[str, Any]] = []
    for spec in specs:
        missing = {"name", "policy_name"} - set(spec)
        if missing:
            raise ValueError(f"Policy spec is missing required keys: {sorted(missing)}")
        normalized.append(
            {
                "name": str(spec["name"]),
                "policy_name": str(spec["policy_name"]),
                "policy_kwargs": dict(spec.get("policy_kwargs", {})),
            }
        )
    return normalized


def _greedy_concentration(results_df: pd.DataFrame, run_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    greedy_rows = results_df.loc[results_df["policy_name"].eq("greedy_loo_fragility")]
    return {
        "average_step_count_l1_by_budget": {
            str(int(budget)): float(group["step_count_l1_from_perfect_balance"].mean())
            for budget, group in greedy_rows.groupby("budget")
        },
        "max_step_count_l1_by_budget": {
            str(int(budget)): float(group["step_count_l1_from_perfect_balance"].max())
            for budget, group in greedy_rows.groupby("budget")
        },
        "max_observed_revealed_row_count_per_step_by_budget": {
            str(int(budget)): int(group["max_revealed_rows_per_step"].max())
            for budget, group in greedy_rows.groupby("budget")
        },
        "selected_step_counts_aggregated": _merge_counter_dicts(
            diag.get("selected_step_counts", {})
            for diag in run_diagnostics
            if diag["policy_name"] == "greedy_loo_fragility"
        ),
        "steps_never_selected_after_initial_seed": _steps_never_selected_after_initial_seed(run_diagnostics),
    }


def _selected_step_counts_df(run_diagnostics: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for diag in run_diagnostics:
        if diag["policy_name"] == "uniform_step_balanced":
            continue
        base = {
            "policy_name": str(diag["policy_name"]),
            "base_policy_name": str(diag.get("base_policy_name", diag["policy_name"])),
            "policy_kwargs": json.dumps(diag.get("policy_kwargs", {}), sort_keys=True),
            "reveal_seed": int(diag["reveal_seed"]),
        }
        total_counts = {step: 0 for step in config.EXPECTED_MITRE_STEP_LABELS}
        total_counts.update({str(k): int(v) for k, v in diag.get("selected_step_counts", {}).items()})
        for step, count in total_counts.items():
            rows.append(
                {
                    **base,
                    "decision_type": "all",
                    "step_name": step,
                    "selected_count_after_initial_seed": int(count),
                }
            )
        by_type = diag.get("selected_step_counts_by_decision_type", {})
        for decision_type, counter in by_type.items():
            counts = {step: 0 for step in config.EXPECTED_MITRE_STEP_LABELS}
            counts.update({str(k): int(v) for k, v in counter.items()})
            for step, count in counts.items():
                rows.append(
                    {
                        **base,
                        "decision_type": str(decision_type),
                        "step_name": step,
                        "selected_count_after_initial_seed": int(count),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "policy_name",
                "base_policy_name",
                "policy_kwargs",
                "reveal_seed",
                "decision_type",
                "step_name",
                "selected_count_after_initial_seed",
            ]
        )
    return pd.DataFrame(rows).sort_values(["reveal_seed", "step_name"]).reset_index(drop=True)


def _exploration_bonus_c_value_summary(
    results_df: pd.DataFrame,
    differences_df: pd.DataFrame,
    auc_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    max_budget: int,
) -> pd.DataFrame:
    """Summarize exploration-bonus sensitivity over c values."""

    exploration_policies = sorted(
        policy for policy in results_df["policy_name"].unique() if str(policy).startswith("exploration_bonus_c")
    )
    rows: list[dict[str, Any]] = []
    for policy in exploration_policies:
        c_value = _parse_exploration_bonus_c(policy)
        policy_results = results_df.loc[results_df["policy_name"].eq(policy)]
        policy_auc = auc_df.loc[auc_df["policy_name"].eq(policy), "auc_distance"].to_numpy(dtype=float)
        policy_diff = differences_df.loc[differences_df["policy_name"].eq(policy)]
        policy_concentration = concentration_df.loc[
            concentration_df["policy_name"].eq(policy) & concentration_df["budget"].eq(max_budget)
        ]
        policy_results_at_max_budget = policy_results.loc[policy_results["budget"].eq(max_budget)]

        if policy_concentration.empty:
            concentration_row: dict[str, Any] = {}
        else:
            concentration_row = policy_concentration.iloc[0].to_dict()

        rows.append(
            {
                "policy_name": str(policy),
                "c": float(c_value),
                "average_auc": float(np.mean(policy_auc)) if len(policy_auc) else np.nan,
                "median_auc": float(np.median(policy_auc)) if len(policy_auc) else np.nan,
                "average_distance_across_all_budgets_and_seeds": float(policy_results[DISTANCE_COL].mean()),
                "win_fraction_vs_uniform": (
                    float(policy_diff["policy_better"].mean()) if not policy_diff.empty else np.nan
                ),
                f"mean_l1_imbalance_at_{max_budget}": _float_or_nan(
                    concentration_row.get("mean_step_count_l1_imbalance")
                ),
                f"median_l1_imbalance_at_{max_budget}": _float_or_nan(
                    concentration_row.get("median_step_count_l1_imbalance")
                ),
                f"max_l1_imbalance_at_{max_budget}": _float_or_nan(
                    concentration_row.get("max_step_count_l1_imbalance")
                ),
                f"mean_max_min_ratio_at_{max_budget}": _float_or_nan(
                    concentration_row.get("mean_max_min_revealed_row_ratio")
                ),
                f"min_step_count_at_{max_budget}_mean": float(
                    policy_results_at_max_budget["min_revealed_rows_per_step"].mean()
                )
                if not policy_results_at_max_budget.empty
                else np.nan,
                f"max_step_count_at_{max_budget}_mean": float(
                    policy_results_at_max_budget["max_revealed_rows_per_step"].mean()
                )
                if not policy_results_at_max_budget.empty
                else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("c").reset_index(drop=True)


def _exploration_bonus_sensitivity_notes(c_value_summary_df: pd.DataFrame) -> dict[str, Any]:
    """Return compact interpretation diagnostics for the c sensitivity table."""

    if c_value_summary_df.empty:
        return {}

    max_budget = _max_budget_from_c_summary(c_value_summary_df)
    l1_col = f"mean_l1_imbalance_at_{max_budget}"
    ordered = c_value_summary_df.sort_values("c").reset_index(drop=True)
    auc_values = ordered["average_auc"].to_numpy(dtype=float)
    l1_values = ordered[l1_col].to_numpy(dtype=float) if l1_col in ordered else np.array([])
    win_values = ordered["win_fraction_vs_uniform"].to_numpy(dtype=float)

    best_auc_row = ordered.loc[ordered["average_auc"].idxmin()]
    best_win_row = ordered.loc[ordered["win_fraction_vs_uniform"].idxmax()]
    lowest_concentration_row = ordered.loc[ordered[l1_col].idxmin()] if l1_col in ordered else None
    c05 = ordered.loc[np.isclose(ordered["c"].to_numpy(dtype=float), 0.5)]

    return {
        "lowest_average_auc_c": float(best_auc_row["c"]),
        "lowest_average_auc_policy": str(best_auc_row["policy_name"]),
        "best_win_rate_c": float(best_win_row["c"]),
        "best_win_rate_policy": str(best_win_row["policy_name"]),
        "lowest_concentration_c_at_largest_budget": (
            float(lowest_concentration_row["c"]) if lowest_concentration_row is not None else None
        ),
        "lowest_concentration_policy_at_largest_budget": (
            str(lowest_concentration_row["policy_name"]) if lowest_concentration_row is not None else None
        ),
        "increasing_c_reduces_mean_l1_concentration_monotonically": _is_nonincreasing(l1_values),
        "increasing_c_reduces_average_auc_monotonically": _is_nonincreasing(auc_values),
        "increasing_c_improves_win_fraction_monotonically": _is_nondecreasing(win_values),
        "c_0_5_summary": c05.iloc[0].to_dict() if not c05.empty else None,
        "c_0_5_is_reasonable_default_diagnostic": _c05_default_diagnostic(ordered, l1_col),
        "note": (
            "These are development sensitivity diagnostics for approximate exploration-bonus LOO "
            "fragility with max_loo_terms_per_step=20, not a final tuned-policy claim."
        ),
    }


def _parse_exploration_bonus_c(policy_name: str) -> float:
    prefix = "exploration_bonus_c"
    if not policy_name.startswith(prefix):
        raise ValueError(f"Policy name does not look like an exploration-bonus c policy: {policy_name!r}")
    return float(policy_name[len(prefix):])


def _max_budget_from_c_summary(c_value_summary_df: pd.DataFrame) -> int:
    for column in c_value_summary_df.columns:
        prefix = "mean_l1_imbalance_at_"
        if column.startswith(prefix):
            return int(column[len(prefix):])
    raise ValueError("c-value summary does not include a mean_l1_imbalance_at_<budget> column.")


def _c05_default_diagnostic(ordered_c_summary: pd.DataFrame, l1_col: str) -> str:
    c05 = ordered_c_summary.loc[np.isclose(ordered_c_summary["c"].to_numpy(dtype=float), 0.5)]
    if c05.empty:
        return "c=0.5 was not included in this run."
    row = c05.iloc[0]
    auc_rank = int(ordered_c_summary["average_auc"].rank(method="min").loc[row.name])
    concentration_rank = int(ordered_c_summary[l1_col].rank(method="min").loc[row.name])
    win_rank = int(
        ordered_c_summary["win_fraction_vs_uniform"].rank(method="min", ascending=False).loc[row.name]
    )
    return (
        f"c=0.5 ranks {auc_rank} by average AUC, {win_rank} by win fraction, "
        f"and {concentration_rank} by mean L1 concentration at the largest budget among "
        f"{len(ordered_c_summary)} c values."
    )


def _is_nonincreasing(values: np.ndarray) -> bool | None:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return None
    return bool(np.all(np.diff(finite) <= 0))


def _is_nondecreasing(values: np.ndarray) -> bool | None:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return None
    return bool(np.all(np.diff(finite) >= 0))


def _float_or_nan(value: Any) -> float:
    if value is None:
        return np.nan
    return float(value)


def _steps_never_selected_after_initial_seed(run_diagnostics: list[dict[str, Any]]) -> list[str]:
    counts = _merge_counter_dicts(
        diag.get("selected_step_counts", {})
        for diag in run_diagnostics
        if diag["policy_name"] == "greedy_loo_fragility"
    )
    return [
        step
        for step in config.EXPECTED_MITRE_STEP_LABELS
        if int(counts.get(step, 0)) == 0
    ]


def _fragility_approximation(fragility_runtime_df: pd.DataFrame) -> dict[str, Any]:
    if fragility_runtime_df.empty:
        return {}
    available = int(fragility_runtime_df["total_loo_terms_available"].sum())
    used = int(fragility_runtime_df["total_loo_terms_used"].sum())
    return {
        "fragility_recomputations": int(len(fragility_runtime_df)),
        "total_exact_loo_terms_available": available,
        "total_loo_terms_used": used,
        "fraction_available_terms_used": float(used / available) if available else None,
        "any_subsampling_used": bool(fragility_runtime_df["loo_subsampled"].any()),
        "max_loo_terms_per_step": _first_non_null(fragility_runtime_df["max_loo_terms_per_step"]),
    }


def _group_sum(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in df.groupby(group_col)[value_col].sum().sort_index().items()
    }


def _first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    value = non_null.iloc[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _merge_counter_dicts(dicts: Any) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counter in dicts:
        for key, value in counter.items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return dict(sorted(merged.items()))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


if __name__ == "__main__":
    main()
