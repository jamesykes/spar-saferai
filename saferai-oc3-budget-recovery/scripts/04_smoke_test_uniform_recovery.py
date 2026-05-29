"""Smoke test uniform budget recovery against the full-data reference distribution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saferai_budget_recovery import config
from saferai_budget_recovery.distances import (
    empirical_quantiles,
    quantile_grid,
    squared_wasserstein2_from_quantiles,
)
from saferai_budget_recovery.reveal import (
    make_initial_seed_reveal,
    make_uniform_reveal_order,
    revealed_at_budget,
    split_revealed_unrevealed,
    usable_fit_rows,
)
from saferai_budget_recovery.sampling import sample_full_reference_p_success, sample_p_success_from_revealed


OUTPUT_DIR = config.FORWARD_MODEL_SMOKE_TEST_DIR
SUMMARY_CSV_PATH = OUTPUT_DIR / "uniform_recovery_smoke_summary.csv"
REPORT_JSON_PATH = OUTPUT_DIR / "uniform_recovery_smoke_report.json"

INITIAL_SEED = 12345
UNIFORM_ORDER_SEED = 12345
REFERENCE_SAMPLE_SEED = 202601
BUDGET_SAMPLE_SEED_BASE = 303000
N_REFERENCE_SAMPLES = 100_000
N_BUDGET_SAMPLES = 50_000
N_QUANTILE_GRID = 1001
REQUESTED_BUDGETS = [45, 90, 180, 360, 720, 1200, 1798]


def main() -> None:
    if not config.SOTA_BETA_FITS_PATH.exists():
        raise FileNotFoundError(
            f"SOTA Beta fits not found: {config.SOTA_BETA_FITS_PATH}. "
            "Run scripts/02_fit_beta_distributions.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fit_df = pd.read_csv(config.SOTA_BETA_FITS_PATH)
    usable_df = usable_fit_rows(fit_df)
    initial_df = make_initial_seed_reveal(usable_df, seed=INITIAL_SEED, strict=True)
    initial_revealed_df, unrevealed_df = split_revealed_unrevealed(usable_df, initial_df)
    uniform_order_df = make_uniform_reveal_order(unrevealed_df, seed=UNIFORM_ORDER_SEED)

    total_usable = len(usable_df)
    budgets = [budget for budget in REQUESTED_BUDGETS if budget <= total_usable]
    if total_usable not in budgets:
        budgets.append(total_usable)
    budgets = sorted(set(budgets))

    grid = quantile_grid(N_QUANTILE_GRID)
    reference_df = sample_full_reference_p_success(
        usable_df, n_samples=N_REFERENCE_SAMPLES, seed=REFERENCE_SAMPLE_SEED
    )
    reference_samples = reference_df["p_success"].to_numpy(dtype=float)
    reference_quantiles = empirical_quantiles(reference_samples, grid)
    reference_summary = _p_success_summary(reference_samples)

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        revealed_df = revealed_at_budget(initial_revealed_df, uniform_order_df, total_budget=budget)
        if budget == total_usable:
            budget_samples = reference_samples
            budget_quantiles = reference_quantiles
            sample_seed = REFERENCE_SAMPLE_SEED
        else:
            sample_seed = BUDGET_SAMPLE_SEED_BASE + budget
            budget_df = sample_p_success_from_revealed(
                revealed_df, n_samples=N_BUDGET_SAMPLES, seed=sample_seed
            )
            budget_samples = budget_df["p_success"].to_numpy(dtype=float)
            budget_quantiles = empirical_quantiles(budget_samples, grid)

        distance = squared_wasserstein2_from_quantiles(budget_quantiles, reference_quantiles, grid)
        counts_by_step = revealed_df.groupby("step_name").size()
        min_rows = int(counts_by_step.min())
        max_rows = int(counts_by_step.max())
        summary = _p_success_summary(budget_samples)
        rows.append(
            {
                "budget": int(budget),
                "additional_rows_beyond_initial_seed": int(budget - len(initial_revealed_df)),
                "squared_wasserstein2_to_full_reference": float(distance),
                "p_success_mean": summary["mean"],
                "p_success_p05": summary["p05"],
                "p_success_p50": summary["p50"],
                "p_success_p95": summary["p95"],
                "min_revealed_rows_per_step": min_rows,
                "max_revealed_rows_per_step": max_rows,
                "max_min_revealed_rows_per_step_ratio": float(max_rows / min_rows),
                "sample_seed": int(sample_seed),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    report = {
        "note": (
            "This is a uniform-reveal smoke test, not the final repeated retrospective allocation "
            "experiment. The full-budget row reuses the full-reference samples for an exact zero "
            "distance sanity check."
        ),
        "initial_seed_size": int(len(initial_revealed_df)),
        "total_usable_fitted_rows": int(total_usable),
        "budgets_used": [int(budget) for budget in budgets],
        "seeds_used": {
            "initial_seed_allocation": INITIAL_SEED,
            "uniform_reveal_order": UNIFORM_ORDER_SEED,
            "reference_sampling": REFERENCE_SAMPLE_SEED,
            "budget_sampling_base": BUDGET_SAMPLE_SEED_BASE,
        },
        "sample_sizes": {
            "n_reference_samples": N_REFERENCE_SAMPLES,
            "n_budget_samples": N_BUDGET_SAMPLES,
            "n_quantile_grid": N_QUANTILE_GRID,
        },
        "reference_summary": reference_summary,
        "budget_results": rows,
        "final_budget_distance_is_close_to_zero": bool(
            summary_df.loc[summary_df["budget"].eq(total_usable), "squared_wasserstein2_to_full_reference"]
            .le(1e-12)
            .all()
        ),
    }
    _write_json(REPORT_JSON_PATH, report)

    print("Uniform recovery smoke test summary")
    print(f"Initial seed size: {len(initial_revealed_df)}")
    print(f"Total usable fitted rows: {total_usable}")
    for row in rows:
        print(
            f"Budget {row['budget']}: W2^2={row['squared_wasserstein2_to_full_reference']}"
        )
    final_distance = float(summary_df.iloc[-1]["squared_wasserstein2_to_full_reference"])
    print(f"Final budget distance close to zero: {final_distance <= 1e-12} ({final_distance})")
    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    print(f"Report JSON: {REPORT_JSON_PATH}")


def _p_success_summary(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "p05": float(np.percentile(samples, 5)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
    }


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
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":
    main()

