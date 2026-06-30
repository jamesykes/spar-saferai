"""Smoke-test v8 hidden reveal orders on the fitted SOTA data."""

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
from saferai_budget_recovery.reveal import (
    FITTED_ROW_UID_COLUMN,
    initial_revealed_from_hidden_orders,
    make_hidden_reveal_orders,
    revealed_df_from_row_ids,
    usable_fit_rows,
)


OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "reveal_audit"
SEEDS = [101, 202]


def main() -> None:
    if not config.SOTA_BETA_FITS_PATH.exists():
        raise FileNotFoundError(
            f"SOTA Beta fits not found: {config.SOTA_BETA_FITS_PATH}. "
            "Run scripts/02_fit_beta_distributions.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fit_df = pd.read_csv(config.SOTA_BETA_FITS_PATH)
    usable_df = usable_fit_rows(fit_df)

    seed_summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    orders_by_seed: dict[int, dict[str, list[str]]] = {}

    for seed in SEEDS:
        hidden = make_hidden_reveal_orders(usable_df, reveal_seed=seed, strict=True)
        initial_df = initial_revealed_from_hidden_orders(usable_df, hidden)
        covered = hidden.initial_row_ids + [
            row_id for order in hidden.orders_by_step.values() for row_id in order
        ]
        duplicate_count = len(covered) - len(set(covered))
        missing_count = len(set(usable_df[FITTED_ROW_UID_COLUMN]) - set(covered))
        extra_count = len(set(covered) - set(usable_df[FITTED_ROW_UID_COLUMN]))
        initial_group_count = int(initial_df.groupby(["step_name", "model"]).size().shape[0])

        first_cycle_ok_count = 0
        for step in config.EXPECTED_MITRE_STEP_LABELS:
            order = hidden.orders_by_step[step]
            first_cycle_ids = order[: len(config.EXPECTED_LLM_MODELS)]
            first_cycle_df = revealed_df_from_row_ids(usable_df, first_cycle_ids)
            first_cycle_models = first_cycle_df["model"].astype(str).tolist()
            first_cycle_has_all_models = set(first_cycle_models) == set(config.EXPECTED_LLM_MODELS)
            first_cycle_ok_count += int(first_cycle_has_all_models)

            hidden_df = revealed_df_from_row_ids(usable_df, order)
            hidden_model_counts = hidden_df.groupby("model").size().to_dict()
            initial_models = (
                initial_df.loc[initial_df["step_name"].eq(step), "model"].astype(str).tolist()
            )
            step_rows.append(
                {
                    "reveal_seed": seed,
                    "step_name": step,
                    "initial_count": int(len(initial_models)),
                    "initial_models": ";".join(sorted(initial_models)),
                    "post_seed_hidden_count": int(len(order)),
                    "first_cycle_models": ";".join(first_cycle_models),
                    "first_cycle_unique_model_count": int(len(set(first_cycle_models))),
                    "first_cycle_has_all_expected_models": bool(first_cycle_has_all_models),
                    "hidden_model_count_min": int(min(hidden_model_counts.values())),
                    "hidden_model_count_max": int(max(hidden_model_counts.values())),
                }
            )

        seed_summaries.append(
            {
                "reveal_seed": seed,
                "initial_seed_size": int(len(hidden.initial_row_ids)),
                "total_usable_fitted_rows": int(len(usable_df)),
                "covered_row_count": int(len(covered)),
                "unique_covered_row_count": int(len(set(covered))),
                "duplicate_row_count": int(duplicate_count),
                "missing_row_count": int(missing_count),
                "extra_row_count": int(extra_count),
                "initial_step_model_group_count": int(initial_group_count),
                "all_initial_step_model_groups_present": bool(initial_group_count == 45),
                "first_cycle_steps_with_all_expected_models": int(first_cycle_ok_count),
                "all_first_cycles_have_all_expected_models": bool(
                    first_cycle_ok_count == len(config.EXPECTED_MITRE_STEP_LABELS)
                ),
                "coverage_ok": bool(
                    len(covered) == len(usable_df)
                    and len(set(covered)) == len(usable_df)
                    and duplicate_count == 0
                    and missing_count == 0
                    and extra_count == 0
                ),
                "metadata_coverage": hidden.metadata["coverage"],
            }
        )
        orders_by_seed[seed] = hidden.orders_by_step

    orders_differ_across_seeds = (
        orders_by_seed[SEEDS[0]] != orders_by_seed[SEEDS[1]]
        if len(SEEDS) >= 2
        else None
    )

    audit_df = pd.DataFrame(step_rows)
    audit_csv_path = OUTPUT_DIR / "hidden_reveal_order_audit.csv"
    audit_json_path = OUTPUT_DIR / "hidden_reveal_order_audit.json"
    audit_df.to_csv(audit_csv_path, index=False)

    report = {
        "seeds": SEEDS,
        "total_usable_fitted_rows": int(len(usable_df)),
        "expected_initial_seed_size": 45,
        "seed_summaries": seed_summaries,
        "orders_differ_across_seeds": orders_differ_across_seeds,
        "all_seed_coverages_ok": bool(all(summary["coverage_ok"] for summary in seed_summaries)),
        "all_initial_seed_sizes_ok": bool(
            all(summary["initial_seed_size"] == 45 for summary in seed_summaries)
        ),
        "all_initial_model_coverages_ok": bool(
            all(summary["all_initial_step_model_groups_present"] for summary in seed_summaries)
        ),
        "all_first_cycles_model_complete": bool(
            all(summary["all_first_cycles_have_all_expected_models"] for summary in seed_summaries)
        ),
        "note": (
            "This is a reveal-protocol smoke audit only. It does not run policy recovery "
            "experiments."
        ),
    }
    audit_json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")

    print("Hidden reveal-order smoke audit")
    print(f"total usable fitted rows: {len(usable_df)}")
    for summary in seed_summaries:
        print(
            "seed={seed}: initial={initial}, covered={covered}, duplicates={dups}, "
            "missing={missing}, first-cycle-complete-steps={cycles}/9".format(
                seed=summary["reveal_seed"],
                initial=summary["initial_seed_size"],
                covered=summary["covered_row_count"],
                dups=summary["duplicate_row_count"],
                missing=summary["missing_row_count"],
                cycles=summary["first_cycle_steps_with_all_expected_models"],
            )
        )
    print(f"orders differ across seeds: {orders_differ_across_seeds}")
    print(f"csv: {audit_csv_path}")
    print(f"json: {audit_json_path}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


if __name__ == "__main__":
    main()
