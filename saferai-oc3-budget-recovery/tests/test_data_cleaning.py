from __future__ import annotations

import pandas as pd

from saferai_budget_recovery.data import (
    add_capability_level,
    add_draw_uid,
    clean_data,
    infer_or_validate_columns,
    make_sota_subset,
)


def _base_rows() -> list[dict]:
    return [
        {
            "model": "claude-sonnet-4-6",
            "run_id": "run-a",
            "repeat_index": 1,
            "step_name": "T1595 - Reconnaissance: Active Scanning",
            "task_name": "Paddle",
            "has_error": False,
            "percentile_25th": 0.2,
            "percentile_50th": 0.5,
            "percentile_75th": 0.8,
        }
    ]


def _clean(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    columns = infer_or_validate_columns(df)
    clean_df, _ = clean_data(df, columns)
    return clean_df


def _clean_with_report(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(rows)
    columns = infer_or_validate_columns(df)
    return clean_data(df, columns)


def test_valid_rows_are_retained() -> None:
    clean_df = _clean(_base_rows())
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["is_valid"] is True or bool(clean_df.iloc[0]["is_valid"])


def test_has_error_rows_are_excluded() -> None:
    rows = _base_rows()
    rows[0]["has_error"] = True
    assert _clean(rows).empty


def test_missing_quartiles_are_excluded() -> None:
    rows = _base_rows()
    rows[0]["percentile_50th"] = None
    assert _clean(rows).empty


def test_bad_quartile_ordering_is_excluded() -> None:
    rows = _base_rows()
    rows[0]["percentile_25th"] = 0.6
    rows[0]["percentile_50th"] = 0.5
    assert _clean(rows).empty


def test_out_of_range_quartiles_are_excluded() -> None:
    rows = _base_rows()
    rows[0]["percentile_75th"] = 1.2
    assert _clean(rows).empty


def test_draw_uid_uses_model_run_id_and_repeat_index() -> None:
    rows = _base_rows()
    rows.append({**_base_rows()[0], "model": "gpt-5-mini", "run_id": "run-b", "repeat_index": 1})
    df = pd.DataFrame(rows)
    columns = infer_or_validate_columns(df)
    with_uid = add_draw_uid(df, columns)
    assert with_uid["repeat_index"].nunique() == 1
    assert with_uid["draw_uid"].nunique() == 2
    assert set(with_uid["draw_uid"]) == {"claude-sonnet-4-6__run-a__1", "gpt-5-mini__run-b__1"}


def test_capability_mapping_assigns_sota_and_saturated() -> None:
    rows = _base_rows()
    rows.append({**_base_rows()[0], "task_name": "Randsubware"})
    df = pd.DataFrame(rows)
    columns = infer_or_validate_columns(df)
    mapped = add_capability_level(df, columns)
    assert list(mapped["capability_level"]) == ["SOTA", "saturated"]


def test_unknown_tasks_become_unknown() -> None:
    rows = _base_rows()
    rows[0]["task_name"] = "Unexpected Task"
    df = pd.DataFrame(rows)
    columns = infer_or_validate_columns(df)
    mapped = add_capability_level(df, columns)
    assert mapped.iloc[0]["capability_level"] == "unknown"


def test_make_sota_subset_returns_only_valid_sota_rows() -> None:
    rows = _base_rows()
    rows.append({**_base_rows()[0], "task_name": "pytorchLightning", "repeat_index": 2})
    rows.append({**_base_rows()[0], "task_name": "Paddle", "repeat_index": 3, "has_error": True})
    clean_df = _clean(rows)
    sota_df = make_sota_subset(clean_df)
    assert len(sota_df) == 1
    assert sota_df.iloc[0]["capability_level"] == "SOTA"
    assert sota_df.iloc[0]["is_valid"] is True or bool(sota_df.iloc[0]["is_valid"])


def test_overlapping_invalid_flags_count_once_in_invalid_rows_total() -> None:
    rows = _base_rows()
    rows.append(
        {
            **_base_rows()[0],
            "repeat_index": 2,
            "has_error": True,
            "percentile_50th": None,
        }
    )
    _, report = _clean_with_report(rows)
    assert report["rows_excluded_due_to_errors"] == 1
    assert report["rows_excluded_due_to_missing_quartiles"] == 1
    assert report["invalid_rows_total"] == 1
    assert report["final_usable_row_count"] == 1


def test_mutually_exclusive_invalid_reason_priority_works() -> None:
    rows = _base_rows()
    rows.extend(
        [
            {
                **_base_rows()[0],
                "repeat_index": 2,
                "has_error": True,
                "percentile_50th": None,
            },
            {**_base_rows()[0], "repeat_index": 3, "percentile_25th": None},
            {
                **_base_rows()[0],
                "repeat_index": 4,
                "percentile_25th": 0.9,
                "percentile_50th": 0.8,
                "percentile_75th": 1.2,
            },
            {**_base_rows()[0], "repeat_index": 5, "percentile_75th": 1.2},
        ]
    )
    _, report = _clean_with_report(rows)
    assert report["mutually_exclusive_invalid_reason_counts"] == {
        "has_error": 1,
        "missing_quartile_no_error": 1,
        "bad_ordering_no_error_or_missing": 1,
        "out_of_range_no_error_missing_or_bad_ordering": 1,
    }


def test_invalid_plus_usable_rows_equal_total_raw_rows() -> None:
    rows = _base_rows()
    rows.append({**_base_rows()[0], "repeat_index": 2, "percentile_75th": 1.2})
    _, report = _clean_with_report(rows)
    assert report["invalid_rows_total"] + report["final_usable_row_count"] == report["total_raw_rows"]


def test_unknown_task_names_appear_in_task_audit() -> None:
    rows = _base_rows()
    rows.append({**_base_rows()[0], "repeat_index": 2, "task_name": "Unexpected Task"})
    _, report = _clean_with_report(rows)
    step_summary = report["tasks_by_step_summary"]["T1595 - Reconnaissance: Active Scanning"]
    assert step_summary["unknown_row_count_before_cleaning"] == 1
    assert "Unexpected Task" in step_summary["sorted_task_names_observed"]


def test_sota_missing_rows_are_detected() -> None:
    _, report = _clean_with_report(_base_rows())
    step = "T1595 - Reconnaissance: Active Scanning"
    assert report["sota_steps_with_missing_rows"][step] == 199
    assert report["sota_total_missing_rows_relative_to_nominal_1800"] == 1799
