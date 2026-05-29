"""Smoke test leave-one-out output fragility at a few uniform-reveal budgets."""

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
from saferai_budget_recovery.fragility import compute_loo_fragility_scores
from saferai_budget_recovery.reveal import (
    make_initial_seed_reveal,
    make_uniform_reveal_order,
    revealed_at_budget,
    split_revealed_unrevealed,
    usable_fit_rows,
)


OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "fragility_smoke_tests"
LONG_CSV_PATH = OUTPUT_DIR / "loo_fragility_by_budget.csv"
REPORT_JSON_PATH = OUTPUT_DIR / "loo_fragility_smoke_report.json"

INITIAL_SEED = 12345
UNIFORM_ORDER_SEED = 12345
BUDGETS = [45, 90, 180, 360]
N_SAMPLES = 5000
N_GRID = 501


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

    budgets = [budget for budget in BUDGETS if budget <= len(usable_df)]
    all_scores: list[pd.DataFrame] = []
    budget_reports: dict[str, Any] = {}

    print("LOO fragility smoke test summary")
    for budget in budgets:
        start = time.perf_counter()
        revealed_df = revealed_at_budget(initial_revealed_df, uniform_order_df, total_budget=budget)
        sample_seed = 505000 + budget
        scores = compute_loo_fragility_scores(
            revealed_df,
            n_samples=N_SAMPLES,
            n_grid=N_GRID,
            seed=sample_seed,
            common_random_numbers=True,
        )
        elapsed = time.perf_counter() - start
        scores.insert(0, "average_draws_per_step", budget / 9)
        scores.insert(0, "budget", budget)
        scores["runtime_seconds"] = elapsed
        all_scores.append(scores)

        finite_scores = scores.loc[np.isfinite(scores["loo_fragility"])]
        top_three = (
            finite_scores.sort_values("loo_fragility", ascending=False)
            .head(3)[["step_name", "loo_fragility", "n_revealed"]]
            .to_dict(orient="records")
        )
        budget_reports[str(budget)] = {
            "top_three_steps_by_loo_fragility": top_three,
            "min_fragility": _float_or_none(finite_scores["loo_fragility"].min()),
            "max_fragility": _float_or_none(finite_scores["loo_fragility"].max()),
            "n_finite_fragility_scores": int(np.isfinite(scores["loo_fragility"]).sum()),
            "n_nan_fragility_scores": int(scores["loo_fragility"].isna().sum()),
            "runtime_seconds": elapsed,
        }
        all_finite = bool(np.isfinite(scores["loo_fragility"]).all())
        top_text = "; ".join(
            f"{row['step_name']}={row['loo_fragility']:.6g}" for row in top_three
        )
        print(f"Budget {budget}: top three {top_text}; all finite: {all_finite}; runtime {elapsed:.2f}s")

    long_df = pd.concat(all_scores, ignore_index=True)
    long_df.to_csv(LONG_CSV_PATH, index=False)
    report = {
        "note": (
            "This is a smoke test of the LOO output-fragility primitive, not the final "
            "allocation experiment or a value-of-information calculation."
        ),
        "budgets_used": budgets,
        "initial_seed_size": int(len(initial_revealed_df)),
        "total_usable_fitted_rows": int(len(usable_df)),
        "n_samples": N_SAMPLES,
        "n_grid": N_GRID,
        "seeds_used": {
            "initial_seed_allocation": INITIAL_SEED,
            "uniform_reveal_order": UNIFORM_ORDER_SEED,
            "fragility_seed_for_budget": {str(budget): int(505000 + budget) for budget in budgets},
        },
        "budget_reports": budget_reports,
    }
    _write_json(REPORT_JSON_PATH, report)
    print(f"Long-form CSV: {LONG_CSV_PATH}")
    print(f"Report JSON: {REPORT_JSON_PATH}")


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


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

