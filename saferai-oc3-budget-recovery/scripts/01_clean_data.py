"""Load, clean, and write sanity-check outputs for the OC3 DoS elicitation CSV."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saferai_budget_recovery import config
from saferai_budget_recovery.data import (
    clean_data,
    infer_or_validate_columns,
    load_raw_data,
    make_sota_subset,
    write_sanity_outputs,
)


def main() -> None:
    raw_df = load_raw_data(config.RAW_DATA_PATH)
    columns = infer_or_validate_columns(raw_df)
    clean_df, report = clean_data(raw_df, columns)
    sota_df = make_sota_subset(clean_df)
    write_sanity_outputs(raw_df, clean_df, sota_df, report, config.SANITY_OUTPUT_DIR)

    print("Cleaning summary")
    print(f"Raw rows: {report['total_raw_rows']}")
    print(f"Usable rows: {report['final_usable_row_count']}")
    print(f"Invalid rows total: {report['invalid_rows_total']}")
    print(f"Usable SOTA rows: {report['usable_sota_row_count']}")
    print(f"Usable saturated rows: {report['usable_saturated_row_count']}")
    print(f"Excluded has_error rows: {report['rows_excluded_due_to_errors']}")
    print(f"Excluded missing quartile rows: {report['rows_excluded_due_to_missing_quartiles']}")
    print(f"Excluded bad ordering rows: {report['rows_excluded_due_to_invalid_quartile_ordering']}")
    print(f"Excluded out-of-range rows: {report['rows_excluded_due_to_out_of_range_quartiles']}")
    print(
        "Mutually exclusive invalid reasons: "
        f"{report['mutually_exclusive_invalid_reason_counts']}"
    )
    print(f"Unique draw_uids overall: {report['number_of_unique_draw_uids_overall']}")
    print(f"Unique draw_uids in SOTA: {report['number_of_unique_draw_uids_in_sota']}")
    print(
        "Every SOTA MITRE step has every expected LLM model: "
        f"{report['every_sota_mitre_step_has_every_expected_llm_model']}"
    )
    print(
        "SOTA missing rows relative to nominal 1800: "
        f"{report['sota_total_missing_rows_relative_to_nominal_1800']}"
    )
    print(f"Processed valid CSV: {config.PROCESSED_DATA_PATH}")
    print(f"Processed SOTA CSV: {config.PROCESSED_DATA_PATH.parent / 'cleaned_sota_rows.csv'}")
    print(f"Sanity output directory: {config.SANITY_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
