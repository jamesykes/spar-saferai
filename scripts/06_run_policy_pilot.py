"""Run a minimal pilot comparing uniform and greedy LOO-fragility allocation."""

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
from saferai_budget_recovery.experiment import run_policy_recovery
from saferai_budget_recovery.reveal import usable_fit_rows


OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "policy_pilot"
RESULTS_CSV_PATH = OUTPUT_DIR / "policy_pilot_results.csv"
REPORT_JSON_PATH = OUTPUT_DIR / "policy_pilot_report.json"

POLICIES = ["uniform_step_balanced", "greedy_loo_fragility"]
REVEAL_SEEDS = [101]
BUDGETS = [45, 90, 180, 360]
N_REFERENCE_SAMPLES = 30_000
N_BUDGET_SAMPLES = 10_000
N_GRID = 501
FRAGILITY_KWARGS = {
    "n_samples": 800,
    "n_grid": 201,
}
FRAGILITY_RECOMPUTE_EVERY = 45
REFERENCE_SEED_BASE = 707000
SAMPLE_SEED_BASE = 808000


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

    for reveal_seed in REVEAL_SEEDS:
        reference_seed = REFERENCE_SEED_BASE + reveal_seed
        for policy in POLICIES:
            result_df, diagnostics = run_policy_recovery(
                fit_df,
                policy_name=policy,
                budgets=BUDGETS,
                reveal_seed=reveal_seed,
                reference_seed=reference_seed,
                sample_seed_base=SAMPLE_SEED_BASE,
                n_reference_samples=N_REFERENCE_SAMPLES,
                n_budget_samples=N_BUDGET_SAMPLES,
                n_grid=N_GRID,
                fragility_kwargs=FRAGILITY_KWARGS,
                fragility_recompute_every=FRAGILITY_RECOMPUTE_EVERY,
            )
            all_results.append(result_df)
            run_diagnostics.append(diagnostics)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    runtime = time.perf_counter() - start
    report = _make_report(results_df, run_diagnostics, total_usable, runtime)
    _write_json(REPORT_JSON_PATH, report)

    print("Policy pilot summary")
    print(
        "Settings: "
        f"budgets={BUDGETS}, reveal_seeds={REVEAL_SEEDS}, "
        f"n_reference_samples={N_REFERENCE_SAMPLES}, n_budget_samples={N_BUDGET_SAMPLES}, "
        f"fragility_kwargs={FRAGILITY_KWARGS}, "
        f"fragility_recompute_every={FRAGILITY_RECOMPUTE_EVERY}"
    )
    table = results_df.pivot_table(
        index="budget",
        columns="policy_name",
        values="squared_wasserstein2_to_full_reference",
        aggfunc="mean",
    )
    print("Distance table:")
    print(table.to_string())
    imbalance = results_df.pivot_table(
        index="budget",
        columns="policy_name",
        values="step_count_l1_from_perfect_balance",
        aggfunc="mean",
    )
    print("Step imbalance table (L1 from perfect balance):")
    print(imbalance.to_string())
    if {"uniform_step_balanced", "greedy_loo_fragility"}.issubset(table.columns):
        print("Greedy beats step-balanced uniform by budget:")
        for budget, row in table.iterrows():
            beats = bool(row["greedy_loo_fragility"] < row["uniform_step_balanced"])
            print(f"  {budget}: {beats}")
    print(f"Runtime seconds: {runtime:.2f}")
    print(f"Results CSV: {RESULTS_CSV_PATH}")
    print(f"Report JSON: {REPORT_JSON_PATH}")


def _make_report(
    results_df: pd.DataFrame,
    run_diagnostics: list[dict[str, Any]],
    total_usable: int,
    runtime: float,
) -> dict[str, Any]:
    policy_summaries: dict[str, Any] = {}
    for policy, group in results_df.groupby("policy_name"):
        distance_by_budget = {
            str(int(row["budget"])): float(row["squared_wasserstein2_to_full_reference"])
            for row in group.sort_values("budget").to_dict(orient="records")
        }
        policy_summaries[str(policy)] = {
            "distance_by_budget": distance_by_budget,
            "simple_average_distance_across_budgets": float(
                group["squared_wasserstein2_to_full_reference"].mean()
            ),
            "final_tested_budget_distance": float(
                group.sort_values("budget").iloc[-1]["squared_wasserstein2_to_full_reference"]
            ),
        }

    greedy_diagnostics = [
        diag for diag in run_diagnostics if diag["policy_name"] == "greedy_loo_fragility"
    ]
    return {
        "note": (
            "This is a small mechanical pilot, not the final repeated allocation experiment. "
            "This pilot uses uniform_step_balanced as the v8-aligned uniform baseline. "
            "uniform_row_random is not the final baseline. The greedy policy uses batched "
            "LOO-fragility recomputation to keep runtime practical."
        ),
        "settings": {
            "policies": POLICIES,
            "reveal_seeds": REVEAL_SEEDS,
            "budgets": BUDGETS,
            "n_reference_samples": N_REFERENCE_SAMPLES,
            "n_budget_samples": N_BUDGET_SAMPLES,
            "n_grid": N_GRID,
            "fragility_kwargs": FRAGILITY_KWARGS,
            "fragility_recompute_every": FRAGILITY_RECOMPUTE_EVERY,
            "reference_seed_base": REFERENCE_SEED_BASE,
            "sample_seed_base": SAMPLE_SEED_BASE,
        },
        "runtime_seconds": runtime,
        "total_usable_fitted_rows": int(total_usable),
        "initial_seed_size": int(results_df["budget"].min()),
        "policies_compared": POLICIES,
        "policy_summaries": policy_summaries,
        "greedy_policy_diagnostics": {
            "selected_step_counts": _merge_counter_dicts(
                diag.get("selected_step_counts", {}) for diag in greedy_diagnostics
            ),
            "fallback_count": int(sum(diag.get("fallback_count", 0) for diag in greedy_diagnostics)),
            "decision_count": int(sum(diag.get("decision_count", 0) for diag in greedy_diagnostics)),
        },
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
