"""Audit LOO-term subsampling for fragility-score rankings."""

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
from saferai_budget_recovery.fragility_audit import (
    compare_fragility_rankings,
    make_per_step_error_table,
    summarize_fragility_approximation_audit,
)
from saferai_budget_recovery.policies import choose_next_uniform_step_balanced
from saferai_budget_recovery.reveal import (
    FITTED_ROW_UID_COLUMN,
    make_initial_seed_reveal,
    split_revealed_unrevealed,
    usable_fit_rows,
)


OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "fragility_approximation_audit"

REQUESTED_SETTINGS = {
    "budgets": [45, 90, 180, 360, 720],
    "reveal_seeds": [101, 202, 303],
    "n_samples": 1000,
    "n_grid": 201,
    "max_loo_terms_per_step_values": [10, 20],
}

# The exact LOO audit is intentionally expensive. Based on the preceding
# repeated-experiment timings, this development audit uses the allowed reduced
# settings while retaining low, medium, and high budgets.
SETTINGS = {
    "budgets": [45, 180, 720],
    "reveal_seeds": [101, 202],
    "n_samples": 700,
    "n_grid": 201,
    "max_loo_terms_per_step_values": [10, 20],
}
RUNTIME_REDUCTION_NOTES = [
    "Reduced reveal_seeds from [101, 202, 303] to [101, 202].",
    "Reduced budgets from [45, 90, 180, 360, 720] to [45, 180, 720].",
    "Reduced n_samples from 1000 to 700.",
]


def main() -> None:
    start = time.perf_counter()
    if not config.SOTA_BETA_FITS_PATH.exists():
        raise FileNotFoundError(
            f"SOTA Beta fits not found: {config.SOTA_BETA_FITS_PATH}. "
            "Run scripts/02_fit_beta_distributions.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fit_df = pd.read_csv(config.SOTA_BETA_FITS_PATH)
    usable_df = usable_fit_rows(fit_df)

    comparisons: list[dict[str, Any]] = []
    per_step_tables: list[pd.DataFrame] = []

    print("LOO approximation audit")
    print(f"Requested settings: {REQUESTED_SETTINGS}")
    print(f"Actual settings: {SETTINGS}")
    print("Runtime reductions:")
    for note in RUNTIME_REDUCTION_NOTES:
        print(f"  - {note}")

    for reveal_seed in SETTINGS["reveal_seeds"]:
        states = _make_step_balanced_revealed_states(
            usable_df,
            reveal_seed=int(reveal_seed),
            budgets=list(SETTINGS["budgets"]),
        )
        for budget, revealed_df in states.items():
            fragility_seed = _fragility_seed(reveal_seed=int(reveal_seed), budget=int(budget))
            print(f"Computing exact LOO: seed={reveal_seed}, budget={budget}")
            exact_df = compute_loo_fragility_scores(
                revealed_df,
                n_samples=int(SETTINGS["n_samples"]),
                n_grid=int(SETTINGS["n_grid"]),
                seed=fragility_seed,
                max_loo_terms_per_step=None,
            )
            for cap in SETTINGS["max_loo_terms_per_step_values"]:
                print(f"  Comparing approximate cap={cap}")
                approx_df = compute_loo_fragility_scores(
                    revealed_df,
                    n_samples=int(SETTINGS["n_samples"]),
                    n_grid=int(SETTINGS["n_grid"]),
                    seed=fragility_seed,
                    max_loo_terms_per_step=int(cap),
                    loo_subsample_seed=fragility_seed,
                )
                comparison = compare_fragility_rankings(exact_df, approx_df)
                comparison.update(
                    {
                        "reveal_seed": int(reveal_seed),
                        "budget": int(budget),
                        "max_loo_terms_per_step": int(cap),
                        "n_samples": int(SETTINGS["n_samples"]),
                        "n_grid": int(SETTINGS["n_grid"]),
                        "exact_total_loo_terms_available": int(
                            exact_df["n_loo_terms_available"].sum()
                        ),
                        "exact_total_loo_terms_used": int(exact_df["n_loo_terms_used"].sum()),
                        "approx_total_loo_terms_available": int(
                            approx_df["n_loo_terms_available"].sum()
                        ),
                        "approx_total_loo_terms_used": int(approx_df["n_loo_terms_used"].sum()),
                        "approx_any_subsampled": bool(approx_df["loo_subsampled"].any()),
                    }
                )
                comparisons.append(comparison)
                per_step_tables.append(
                    make_per_step_error_table(
                        exact_df=exact_df,
                        approx_df=approx_df,
                        reveal_seed=int(reveal_seed),
                        budget=int(budget),
                        max_loo_terms_per_step=int(cap),
                    )
                )

    comparison_df = pd.DataFrame(comparisons)
    per_step_df = pd.concat(per_step_tables, ignore_index=True)
    summary_by_cap = summarize_fragility_approximation_audit(comparison_df)
    runtime_seconds = time.perf_counter() - start

    comparisons_path = OUTPUT_DIR / "loo_approximation_comparisons.csv"
    per_step_path = OUTPUT_DIR / "loo_approximation_per_step_errors.csv"
    report_path = OUTPUT_DIR / "loo_approximation_audit_report.json"
    comparison_df.to_csv(comparisons_path, index=False)
    per_step_df.to_csv(per_step_path, index=False)

    report = {
        "settings_used": SETTINGS,
        "requested_settings": REQUESTED_SETTINGS,
        "runtime_reduction_notes": RUNTIME_REDUCTION_NOTES,
        "runtime_seconds": runtime_seconds,
        "number_of_comparisons": int(len(comparison_df)),
        "caps_evaluated": SETTINGS["max_loo_terms_per_step_values"],
        "budgets": SETTINGS["budgets"],
        "reveal_seeds": SETTINGS["reveal_seeds"],
        "n_samples": SETTINGS["n_samples"],
        "n_grid": SETTINGS["n_grid"],
        "summary_by_cap": summary_by_cap,
        "heuristic_threshold_note": (
            "A cap is flagged as appears_acceptable if top1 match rate >= 0.7 "
            "and mean top3 Jaccard >= 0.6. This is a rough diagnostic, not a "
            "scientific acceptance criterion."
        ),
        "audit_note": (
            "This audit compares exact v8 LOO fragility with capped LOO-term "
            "subsampling on fixed step-balanced-uniform revealed states."
        ),
    }
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True))

    print("\nSummary by cap:")
    for cap, summary in summary_by_cap.items():
        print(
            f"cap={cap}: top1_match_rate={summary['top1_match_rate']:.3f}, "
            f"mean_top3_jaccard={summary['mean_top3_jaccard']:.3f}, "
            f"mean_spearman={summary['mean_spearman_correlation']:.3f}, "
            f"flag={summary['heuristic_warning_flag']}"
        )
    cap20_flag = summary_by_cap.get("20", {}).get("heuristic_warning_flag", "not_evaluated")
    print(f"cap 20 diagnostic: {cap20_flag}")
    print(f"Runtime seconds: {runtime_seconds:.2f}")
    print(f"comparisons: {comparisons_path}")
    print(f"per-step errors: {per_step_path}")
    print(f"report: {report_path}")


def _make_step_balanced_revealed_states(
    fit_df: pd.DataFrame,
    reveal_seed: int,
    budgets: list[int],
) -> dict[int, pd.DataFrame]:
    initial = make_initial_seed_reveal(fit_df, seed=reveal_seed, strict=True)
    revealed, unrevealed = split_revealed_unrevealed(fit_df, initial)
    rng = np.random.default_rng(reveal_seed)
    states: dict[int, pd.DataFrame] = {}
    for budget in sorted(budgets):
        if budget < len(revealed):
            raise ValueError(
                f"budget={budget} is smaller than initial seed allocation size {len(revealed)}."
            )
        while len(revealed) < budget:
            row_id = choose_next_uniform_step_balanced(revealed, unrevealed, rng)
            selected = unrevealed.loc[unrevealed[FITTED_ROW_UID_COLUMN].eq(row_id)].copy()
            if len(selected) != 1:
                raise ValueError(f"Expected one selected row for {row_id!r}; found {len(selected)}.")
            revealed = pd.concat([revealed, selected], ignore_index=True)
            unrevealed = unrevealed.loc[~unrevealed[FITTED_ROW_UID_COLUMN].eq(row_id)].reset_index(
                drop=True
            )
        states[int(budget)] = revealed.copy().reset_index(drop=True)
    return states


def _fragility_seed(reveal_seed: int, budget: int) -> int:
    return int(808_000 + reveal_seed * 1_000 + budget)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


if __name__ == "__main__":
    main()
