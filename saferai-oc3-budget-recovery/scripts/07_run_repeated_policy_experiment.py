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
    compute_auc_by_seed_policy,
    compute_policy_differences,
    summarize_policy_results,
    summarize_concentration_by_budget,
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
MODE = os.environ.get("SAFERAI_EXPERIMENT_MODE", "FAST_DEV_10_SEEDS")


def main() -> None:
    if not config.SOTA_BETA_FITS_PATH.exists():
        raise FileNotFoundError(
            f"SOTA Beta fits not found: {config.SOTA_BETA_FITS_PATH}. "
            "Run scripts/02_fit_beta_distributions.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MODE not in CONFIGURATIONS:
        raise ValueError(f"Unknown MODE={MODE!r}; expected one of {sorted(CONFIGURATIONS)}.")
    settings = CONFIGURATIONS[MODE]
    requested_settings = settings.copy()
    prefix = MODE.lower()

    start = time.perf_counter()
    fit_df = pd.read_csv(config.SOTA_BETA_FITS_PATH)
    usable_df = usable_fit_rows(fit_df)
    total_usable = len(usable_df)
    grid = quantile_grid(settings["n_grid"])

    print("Repeated policy experiment")
    print(f"Mode: {MODE}")
    print(f"Settings: {settings}")
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
        for policy in settings["policies"]:
            run_start = time.perf_counter()
            print(f"Running reveal_seed={reveal_seed}, policy={policy}", flush=True)
            result_df, diagnostics = run_policy_recovery(
                fit_df,
                policy_name=policy,
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
    differences_df = compute_policy_differences(results_df)
    auc_df = compute_auc_by_seed_policy(results_df)
    win_by_budget_df = summarize_win_rate_by_budget(differences_df)
    win_by_seed_df = summarize_win_rate_by_seed(differences_df, auc_df)
    concentration_df = summarize_concentration_by_budget(results_df)
    selected_step_counts_df = _greedy_selected_step_counts_df(run_diagnostics)
    fragility_runtime_df = pd.DataFrame(fragility_runtime_rows)

    paths = {
        "results": OUTPUT_DIR / f"{prefix}_repeated_policy_results.csv",
        "summary": OUTPUT_DIR / f"{prefix}_policy_summary_by_budget.csv",
        "differences": OUTPUT_DIR / f"{prefix}_policy_differences_by_seed_budget.csv",
        "auc": OUTPUT_DIR / f"{prefix}_policy_auc_by_seed.csv",
        "win_by_budget": OUTPUT_DIR / f"{prefix}_win_rate_by_budget.csv",
        "win_by_seed": OUTPUT_DIR / f"{prefix}_win_rate_by_seed.csv",
        "concentration": OUTPUT_DIR / f"{prefix}_concentration_by_budget.csv",
        "greedy_selected_step_counts": OUTPUT_DIR / f"{prefix}_greedy_selected_step_counts.csv",
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
    selected_step_counts_df.to_csv(paths["greedy_selected_step_counts"], index=False)
    fragility_runtime_df.to_csv(paths["fragility_runtime"], index=False)

    runtime = time.perf_counter() - start
    report = _make_report(
        mode=MODE,
        settings=settings,
        requested_settings=requested_settings,
        results_df=results_df,
        summary_df=summary_df,
        differences_df=differences_df,
        auc_df=auc_df,
        win_by_budget_df=win_by_budget_df,
        win_by_seed_df=win_by_seed_df,
        concentration_df=concentration_df,
        selected_step_counts_df=selected_step_counts_df,
        run_diagnostics=run_diagnostics,
        fragility_runtime_df=fragility_runtime_df,
        runtime=runtime,
        reference_sampling_seconds=reference_sampling_seconds,
        total_usable=total_usable,
    )
    _write_json(paths["report"], report)

    distance_table = summary_df.pivot(index="budget", columns="policy_name", values="distance_mean")
    auc_summary = auc_df.groupby("policy_name")["auc_distance"].agg(["mean", "median"])
    greedy_fraction = float(differences_df["greedy_better"].mean()) if len(differences_df) else 0.0
    print("Summary distance table (mean across reveal seeds):")
    print(distance_table.to_string())
    print("Fraction of seed-budget comparisons where greedy beats uniform:")
    print(greedy_fraction)
    print("Greedy win fraction by budget:")
    print(win_by_budget_df[["budget", "greedy_win_fraction", "mean_greedy_minus_uniform_distance"]].to_string(index=False))
    print("AUC summary:")
    print(auc_summary.to_string())
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
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    differences_df: pd.DataFrame,
    auc_df: pd.DataFrame,
    win_by_budget_df: pd.DataFrame,
    win_by_seed_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    selected_step_counts_df: pd.DataFrame,
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
            "Greedy LOO fragility used approximate LOO-term subsampling with "
            f"max_loo_terms_per_step={max_terms}. Previous approximation audit suggested cap 20 "
            "preserved top fragile-node rankings on reduced audit settings, but this remains an "
            "approximation to the exact v8 LOO definition. Exact LOO remains available by setting "
            "this to null."
        )
    greedy_auc_mean = float(
        auc_df.loc[auc_df["policy_name"].eq("greedy_loo_fragility"), "auc_distance"].mean()
    )
    uniform_auc_mean = float(
        auc_df.loc[auc_df["policy_name"].eq("uniform_step_balanced"), "auc_distance"].mean()
    )
    greedy_win_fraction = float(differences_df["greedy_better"].mean())
    greedy_concentration = _greedy_concentration(results_df, run_diagnostics)

    return {
        "note": (
            "This is a stronger development run, not the final report run. Greedy LOO fragility "
            "uses approximate LOO-term subsampling with max_loo_terms_per_step=20. Previous "
            "approximation audit suggested cap 20 preserved top fragile-node rankings on reduced "
            "audit settings, but this remains an approximation to the exact v8 LOO definition."
        ),
        "mode": mode,
        "settings": settings,
        "requested_settings": requested_settings,
        "settings_reduced_from_requested": settings != requested_settings,
        "approximation_note": approximation_note,
        "runtime_seconds": runtime,
        "reference_design": {
            "reference_sampled_once_globally": True,
            "reference_seed": settings["reference_seed"],
            "reference_sampling_seconds": reference_sampling_seconds,
        },
        "total_usable_fitted_rows": int(total_usable),
        "initial_seed_size": int(results_df["budget"].min()),
        "policies_compared": settings["policies"],
        "number_of_reveal_seeds": int(len(settings["reveal_seeds"])),
        "budgets": settings["budgets"],
        "runtime_diagnostics": _runtime_diagnostics(run_diagnostics),
        "summary_by_policy": policy_summary,
        "greedy_better_count": int(differences_df["greedy_better"].sum()),
        "total_seed_budget_comparisons": int(len(differences_df)),
        "greedy_better_fraction": greedy_win_fraction,
        "win_rate_by_budget": win_by_budget_df.to_dict(orient="records"),
        "win_rate_by_seed": win_by_seed_df.to_dict(orient="records"),
        "concentration_by_budget": concentration_df.to_dict(orient="records"),
        "greedy_concentration_diagnostics": greedy_concentration,
        "greedy_selected_step_counts": selected_step_counts_df.to_dict(orient="records"),
        "fragility_approximation_diagnostics": _fragility_approximation(fragility_runtime_df),
        "interpretation_notes": {
            "greedy_has_lower_average_auc_than_uniform": bool(greedy_auc_mean < uniform_auc_mean),
            "greedy_wins_majority_of_seed_budget_comparisons": bool(greedy_win_fraction > 0.5),
            "greedy_advantage_if_any_comes_with_high_concentration": bool(
                max(greedy_concentration["max_step_count_l1_by_budget"].values() or [0]) > 0
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


def _greedy_selected_step_counts_df(run_diagnostics: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for diag in run_diagnostics:
        if diag["policy_name"] != "greedy_loo_fragility":
            continue
        counts = {step: 0 for step in config.EXPECTED_MITRE_STEP_LABELS}
        counts.update({str(k): int(v) for k, v in diag.get("selected_step_counts", {}).items()})
        for step, count in counts.items():
            rows.append(
                {
                    "policy_name": "greedy_loo_fragility",
                    "reveal_seed": int(diag["reveal_seed"]),
                    "step_name": step,
                    "selected_count_after_initial_seed": int(count),
                }
            )
    return pd.DataFrame(rows).sort_values(["reveal_seed", "step_name"]).reset_index(drop=True)


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
