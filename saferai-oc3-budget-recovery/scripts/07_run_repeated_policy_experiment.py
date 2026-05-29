"""Run a repeated development experiment for the two primary policies."""

from __future__ import annotations

import json
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
    compute_auc_by_seed_policy,
    compute_policy_differences,
    summarize_policy_results,
)
from saferai_budget_recovery.experiment import run_policy_recovery
from saferai_budget_recovery.reveal import usable_fit_rows


OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "repeated_policy_experiment"
RESULTS_CSV_PATH = OUTPUT_DIR / "repeated_policy_results.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "policy_summary_by_budget.csv"
DIFFERENCES_CSV_PATH = OUTPUT_DIR / "policy_differences_by_seed_budget.csv"
AUC_CSV_PATH = OUTPUT_DIR / "policy_auc_by_seed.csv"
REPORT_JSON_PATH = OUTPUT_DIR / "repeated_policy_experiment_report.json"

REQUESTED_SETTINGS = {
    "budgets": [45, 90, 180, 360, 720],
    "reveal_seeds": [101, 202, 303, 404, 505],
    "policies": ["uniform_step_balanced", "greedy_loo_fragility"],
    "n_reference_samples": 50_000,
    "n_budget_samples": 20_000,
    "n_grid": 501,
    "fragility_kwargs": {"n_samples": 1000, "n_grid": 201},
    "fragility_recompute_every": 45,
}

ACTUAL_SETTINGS = {
    "budgets": [45, 90, 180, 360, 720],
    "reveal_seeds": [101, 202],
    "policies": ["uniform_step_balanced", "greedy_loo_fragility"],
    "n_reference_samples": 30_000,
    "n_budget_samples": 10_000,
    "n_grid": 501,
    "fragility_kwargs": {"n_samples": 600, "n_grid": 151},
    "fragility_recompute_every": 90,
    "reference_seed_base": 907000,
    "sample_seed_base": 1008000,
}


def main() -> None:
    if not config.SOTA_BETA_FITS_PATH.exists():
        raise FileNotFoundError(
            f"SOTA Beta fits not found: {config.SOTA_BETA_FITS_PATH}. "
            "Run scripts/02_fit_beta_distributions.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    fit_df = pd.read_csv(config.SOTA_BETA_FITS_PATH)
    total_usable = len(usable_fit_rows(fit_df))
    all_results: list[pd.DataFrame] = []
    run_diagnostics: list[dict[str, Any]] = []

    print("Repeated policy experiment")
    print(f"Actual settings: {ACTUAL_SETTINGS}")
    for reveal_seed in ACTUAL_SETTINGS["reveal_seeds"]:
        reference_seed = ACTUAL_SETTINGS["reference_seed_base"] + reveal_seed
        for policy in ACTUAL_SETTINGS["policies"]:
            run_start = time.perf_counter()
            print(f"Running reveal_seed={reveal_seed}, policy={policy}", flush=True)
            result_df, diagnostics = run_policy_recovery(
                fit_df,
                policy_name=policy,
                budgets=ACTUAL_SETTINGS["budgets"],
                reveal_seed=reveal_seed,
                reference_seed=reference_seed,
                sample_seed_base=ACTUAL_SETTINGS["sample_seed_base"],
                n_reference_samples=ACTUAL_SETTINGS["n_reference_samples"],
                n_budget_samples=ACTUAL_SETTINGS["n_budget_samples"],
                n_grid=ACTUAL_SETTINGS["n_grid"],
                fragility_kwargs=ACTUAL_SETTINGS["fragility_kwargs"],
                fragility_recompute_every=ACTUAL_SETTINGS["fragility_recompute_every"],
            )
            elapsed = time.perf_counter() - run_start
            print(f"  finished in {elapsed:.2f}s", flush=True)
            all_results.append(result_df)
            run_diagnostics.append(diagnostics)

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = summarize_policy_results(results_df)
    differences_df = compute_policy_differences(results_df)
    auc_df = compute_auc_by_seed_policy(results_df)

    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    differences_df.to_csv(DIFFERENCES_CSV_PATH, index=False)
    auc_df.to_csv(AUC_CSV_PATH, index=False)

    runtime = time.perf_counter() - start
    report = _make_report(
        results_df=results_df,
        summary_df=summary_df,
        differences_df=differences_df,
        auc_df=auc_df,
        run_diagnostics=run_diagnostics,
        runtime=runtime,
        total_usable=total_usable,
    )
    _write_json(REPORT_JSON_PATH, report)

    distance_table = summary_df.pivot(
        index="budget", columns="policy_name", values="distance_mean"
    )
    auc_summary = auc_df.groupby("policy_name")["auc_distance"].agg(["mean", "median"])
    greedy_fraction = float(differences_df["greedy_better"].mean()) if len(differences_df) else 0.0
    print("Summary distance table (mean across reveal seeds):")
    print(distance_table.to_string())
    print("Fraction of seed-budget comparisons where greedy beats uniform:")
    print(greedy_fraction)
    print("AUC summary:")
    print(auc_summary.to_string())
    print(f"Runtime seconds: {runtime:.2f}")
    print(f"Results CSV: {RESULTS_CSV_PATH}")
    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    print(f"Differences CSV: {DIFFERENCES_CSV_PATH}")
    print(f"AUC CSV: {AUC_CSV_PATH}")
    print(f"Report JSON: {REPORT_JSON_PATH}")


def _make_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    differences_df: pd.DataFrame,
    auc_df: pd.DataFrame,
    run_diagnostics: list[dict[str, Any]],
    runtime: float,
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

    greedy_rows = results_df.loc[results_df["policy_name"].eq("greedy_loo_fragility")]
    greedy_concentration = {
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
    }

    return {
        "note": (
            "This is a repeated development experiment using moderate Monte Carlo settings, "
            "not necessarily the final report run."
        ),
        "requested_settings": REQUESTED_SETTINGS,
        "actual_settings": ACTUAL_SETTINGS,
        "settings_reduction_note": (
            "The development run uses two reveal seeds, smaller Monte Carlo samples, and "
            "less frequent fragility recomputation than the requested moderate defaults to "
            "keep runtime practical in this pass. The exact settings are recorded above."
        ),
        "runtime_seconds": runtime,
        "total_usable_fitted_rows": int(total_usable),
        "initial_seed_size": int(results_df["budget"].min()),
        "policies_compared": ACTUAL_SETTINGS["policies"],
        "number_of_reveal_seeds": int(len(ACTUAL_SETTINGS["reveal_seeds"])),
        "budgets": ACTUAL_SETTINGS["budgets"],
        "summary_by_policy": policy_summary,
        "greedy_better_count": int(differences_df["greedy_better"].sum()),
        "total_seed_budget_comparisons": int(len(differences_df)),
        "greedy_better_fraction": float(differences_df["greedy_better"].mean()),
        "greedy_concentration_diagnostics": greedy_concentration,
        "run_diagnostics": run_diagnostics,
    }


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

