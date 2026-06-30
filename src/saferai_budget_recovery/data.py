"""Data loading, cleaning, and sanity-check output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from saferai_budget_recovery import config


ColumnMap = dict[str, str]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw elicitation CSV."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def infer_or_validate_columns(df: pd.DataFrame) -> dict[str, str]:
    """Infer required semantic columns from the raw CSV headers."""

    candidates = {
        "model": ("model", "llm_model", "forecaster_model"),
        "run_id": ("run_id", "run"),
        "repeat_index": ("repeat_index", "repeat", "repeat_id"),
        "task_name": ("task_name", "task", "benchmark_task"),
        "mitre_step_label": ("step_name", "mitre_step_label", "step_label", "technique_label"),
        "has_error": ("has_error", "error", "is_error"),
        "percentile_25th": ("percentile_25th", "p25", "q25"),
        "percentile_50th": ("percentile_50th", "p50", "q50", "median"),
        "percentile_75th": ("percentile_75th", "p75", "q75"),
    }

    mapping: ColumnMap = {}
    missing: list[str] = []
    for semantic_name, names in candidates.items():
        actual = next((name for name in names if name in df.columns), None)
        if actual is None:
            missing.append(f"{semantic_name} candidates={names}")
        else:
            mapping[semantic_name] = actual

    if missing:
        available = ", ".join(df.columns)
        missing_text = "; ".join(missing)
        raise ValueError(
            "Raw CSV is missing required columns. "
            f"Missing semantic columns: {missing_text}. Available columns: {available}"
        )

    return mapping


def add_draw_uid(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Add globally unique draw IDs from model, run_id, and repeat_index."""

    _require_columns(df, columns, ("model", "run_id", "repeat_index"))
    out = df.copy()
    required = [columns["model"], columns["run_id"], columns["repeat_index"]]
    missing_mask = out[required].isna().any(axis=1)
    for col in required:
        missing_mask = missing_mask | out[col].astype("string").str.strip().eq("")

    if missing_mask.any():
        examples = out.loc[missing_mask, required].head(5).to_dict(orient="records")
        raise ValueError(
            "Cannot construct draw_uid because model, run_id, or repeat_index is missing. "
            f"Example bad rows: {examples}"
        )

    out["draw_uid"] = (
        out[columns["model"]].astype(str)
        + "__"
        + out[columns["run_id"]].astype(str)
        + "__"
        + out[columns["repeat_index"]].astype(str)
    )
    return out


def add_capability_level(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Map task names to SOTA, saturated, or unknown capability levels."""

    _require_columns(df, columns, ("task_name",))
    out = df.copy()
    out["capability_level"] = (
        out[columns["task_name"]]
        .map(config.CAPABILITY_LEVEL_BY_TASK)
        .fillna("unknown")
    )
    return out


def add_validity_flags(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Add row-level validity flags for usable quartile elicitation rows."""

    _require_columns(
        df,
        columns,
        (
            "has_error",
            "percentile_25th",
            "percentile_50th",
            "percentile_75th",
        ),
    )
    out = df.copy()
    quartile_actuals = [columns[name] for name in config.QUARTILE_COLUMNS]
    for col in quartile_actuals:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    quartiles = out[quartile_actuals]
    out["invalid_has_error"] = _to_bool_series(out[columns["has_error"]])
    out["invalid_missing_quartile"] = quartiles.isna().any(axis=1)
    out["invalid_bad_ordering"] = (
        (out[columns["percentile_25th"]] > out[columns["percentile_50th"]])
        | (out[columns["percentile_50th"]] > out[columns["percentile_75th"]])
    ).fillna(False)
    out["invalid_out_of_range"] = (
        quartiles.lt(0).any(axis=1) | quartiles.gt(1).any(axis=1)
    ).fillna(False)
    invalid_flags = [
        "invalid_has_error",
        "invalid_missing_quartile",
        "invalid_bad_ordering",
        "invalid_out_of_range",
    ]
    out["is_valid"] = ~out[invalid_flags].any(axis=1)
    out["is_sota"] = out.get("capability_level", pd.Series(index=out.index, dtype=object)).eq("SOTA")
    return out


def clean_data(df: pd.DataFrame, columns: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Clean raw rows and return usable rows plus a sanity report."""

    augmented = add_draw_uid(df, columns)
    augmented = add_capability_level(augmented, columns)
    augmented = add_validity_flags(augmented, columns)

    clean_df = augmented.loc[augmented["is_valid"]].copy()
    sota_df = make_sota_subset(clean_df)

    model_col = columns["model"]
    task_col = columns["task_name"]
    step_col = columns["mitre_step_label"]
    invalid_reason_counts = _mutually_exclusive_invalid_reason_counts(augmented)
    invalid_rows_total = int((~augmented["is_valid"]).sum())
    tasks_by_step = _tasks_by_step_audit(augmented, clean_df, columns)
    sota_draws_by_step = _sota_draws_by_step_audit(sota_df, columns)
    sota_steps_with_missing_rows = _sota_steps_with_missing_rows(sota_draws_by_step)

    report = {
        "total_raw_rows": int(len(augmented)),
        "rows_excluded_due_to_errors": int(augmented["invalid_has_error"].sum()),
        "rows_excluded_due_to_missing_quartiles": int(augmented["invalid_missing_quartile"].sum()),
        "rows_excluded_due_to_invalid_quartile_ordering": int(augmented["invalid_bad_ordering"].sum()),
        "rows_excluded_due_to_out_of_range_quartiles": int(augmented["invalid_out_of_range"].sum()),
        "invalid_rows_total": invalid_rows_total,
        "valid_rows_total": int(len(clean_df)),
        "final_usable_row_count": int(len(clean_df)),
        "mutually_exclusive_invalid_reason_counts": invalid_reason_counts,
        "exclusion_count_note": (
            "The per-flag exclusion counts are not mutually exclusive; invalid_rows_total "
            "counts unique rows excluded by any cleaning rule."
        ),
        "usable_sota_row_count": int(len(sota_df)),
        "usable_saturated_row_count": int((clean_df["capability_level"] == "saturated").sum()),
        "count_by_capability_level_before_cleaning": _value_counts(augmented["capability_level"]),
        "count_by_capability_level_after_cleaning": _value_counts(clean_df["capability_level"]),
        "count_by_llm_model_after_cleaning": _value_counts(clean_df[model_col]),
        "count_by_mitre_step_label_after_cleaning": _value_counts(clean_df[step_col]),
        "count_by_mitre_step_label_for_sota_after_cleaning": _value_counts(sota_df[step_col]),
        "number_of_unique_draw_uids_overall": int(clean_df["draw_uid"].nunique()),
        "number_of_unique_draw_uids_in_sota": int(sota_df["draw_uid"].nunique()),
        "every_sota_mitre_step_has_every_expected_llm_model": _has_all_expected_models_per_step(
            sota_df, columns
        ),
        "unexpected_model_names": _unexpected_values(augmented[model_col], config.EXPECTED_LLM_MODELS),
        "unexpected_task_names": _unexpected_values(augmented[task_col], config.EXPECTED_TASKS),
        "unexpected_mitre_step_labels": _unexpected_values(
            augmented[step_col], config.EXPECTED_MITRE_STEP_LABELS
        ),
        "tasks_by_step_summary": _tasks_by_step_summary(tasks_by_step),
        "every_step_has_expected_capability_levels_before_cleaning": _every_step_has_levels(
            tasks_by_step, ("SOTA", "saturated"), suffix="before_cleaning"
        ),
        "every_step_has_expected_capability_levels_after_cleaning": _every_step_has_levels(
            tasks_by_step, ("SOTA", "saturated"), suffix="after_cleaning"
        ),
        "capability_mapping_note": (
            "The code maps capability level by task name, but audits the task names observed "
            "for each MITRE-step label because the v8 plan describes the mapping as step-specific."
        ),
        "sota_steps_with_missing_rows": sota_steps_with_missing_rows,
        "sota_total_missing_rows_relative_to_nominal_1800": int(
            sota_draws_by_step["missing_draw_count_relative_to_nominal_200"].sum()
        ),
        "column_mapping": columns,
    }
    if report["invalid_rows_total"] + report["final_usable_row_count"] != report["total_raw_rows"]:
        raise AssertionError(
            "Invalid accounting invariant failed: invalid_rows_total + final_usable_row_count "
            "must equal total_raw_rows."
        )
    return clean_df, report


def make_sota_subset(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are valid and mapped to the SOTA capability level."""

    required = {"capability_level", "is_valid"}
    missing = required - set(clean_df.columns)
    if missing:
        raise ValueError(f"Cannot create SOTA subset; missing columns: {sorted(missing)}")
    return clean_df.loc[clean_df["is_valid"] & clean_df["capability_level"].eq("SOTA")].copy()


def write_sanity_outputs(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    sota_df: pd.DataFrame,
    report: dict,
    output_dir: str | Path,
) -> None:
    """Write processed datasets and sanity-check outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    columns = report["column_mapping"]
    augmented = add_draw_uid(raw_df, columns)
    augmented = add_capability_level(augmented, columns)
    augmented = add_validity_flags(augmented, columns)

    clean_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    sota_processed_path = config.PROCESSED_DATA_PATH.parent / "cleaned_sota_rows.csv"
    sota_df.to_csv(sota_processed_path, index=False)

    _write_json(output_path / "cleaning_report.json", report)
    _write_json(output_path / "column_mapping.json", columns)

    sota_df.head(20).to_csv(output_path / "cleaned_sota_head.csv", index=False)
    _rows_by_step_and_model(sota_df, columns).to_csv(
        output_path / "rows_by_step_and_model_sota.csv", index=False
    )
    _rows_by_step_and_capability(clean_df, columns).to_csv(
        output_path / "rows_by_step_and_capability.csv", index=False
    )
    _tasks_by_step_audit(augmented, clean_df, columns).to_csv(
        output_path / "tasks_by_step.csv", index=False
    )
    _sota_draws_by_step_audit(sota_df, columns).to_csv(
        output_path / "sota_draws_by_step.csv", index=False
    )
    augmented.loc[~augmented["is_valid"]].head(50).to_csv(
        output_path / "invalid_row_examples.csv", index=False
    )


def _require_columns(df: pd.DataFrame, columns: ColumnMap, semantic_names: tuple[str, ...]) -> None:
    missing_semantic = [name for name in semantic_names if name not in columns]
    missing_actual = [
        columns[name] for name in semantic_names if name in columns and columns[name] not in df.columns
    ]
    if missing_semantic or missing_actual:
        raise ValueError(
            "Missing required columns. "
            f"Missing semantic names: {missing_semantic}; missing CSV columns: {missing_actual}"
        )


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _unexpected_values(series: pd.Series, expected: tuple[str, ...]) -> list[str]:
    observed = {str(value) for value in series.dropna().unique()}
    return sorted(observed - set(expected))


def _mutually_exclusive_invalid_reason_counts(df: pd.DataFrame) -> dict[str, int]:
    has_error = df["invalid_has_error"]
    missing = ~has_error & df["invalid_missing_quartile"]
    bad_ordering = ~has_error & ~df["invalid_missing_quartile"] & df["invalid_bad_ordering"]
    out_of_range = (
        ~has_error
        & ~df["invalid_missing_quartile"]
        & ~df["invalid_bad_ordering"]
        & df["invalid_out_of_range"]
    )
    return {
        "has_error": int(has_error.sum()),
        "missing_quartile_no_error": int(missing.sum()),
        "bad_ordering_no_error_or_missing": int(bad_ordering.sum()),
        "out_of_range_no_error_missing_or_bad_ordering": int(out_of_range.sum()),
    }


def _tasks_by_step_audit(raw_augmented: pd.DataFrame, clean_df: pd.DataFrame, columns: ColumnMap) -> pd.DataFrame:
    step_col = columns["mitre_step_label"]
    task_col = columns["task_name"]
    steps = sorted(
        set(config.EXPECTED_MITRE_STEP_LABELS)
        | {str(value) for value in raw_augmented[step_col].dropna().unique()}
    )
    rows: list[dict[str, Any]] = []
    for step in steps:
        before = raw_augmented.loc[raw_augmented[step_col].astype(str).eq(step)]
        after = clean_df.loc[clean_df[step_col].astype(str).eq(step)]
        task_counts = {
            str(task): int(count)
            for task, count in before[task_col].value_counts(dropna=False).sort_index().items()
        }
        rows.append(
            {
                "mitre_step_label": step,
                "unique_task_name_count": int(before[task_col].nunique(dropna=True)),
                "sorted_task_names_observed": "; ".join(
                    sorted(str(task) for task in before[task_col].dropna().unique())
                ),
                "row_count_by_task": json.dumps(task_counts, sort_keys=True),
                "sota_row_count_before_cleaning": int(before["capability_level"].eq("SOTA").sum()),
                "saturated_row_count_before_cleaning": int(
                    before["capability_level"].eq("saturated").sum()
                ),
                "unknown_row_count_before_cleaning": int(before["capability_level"].eq("unknown").sum()),
                "sota_row_count_after_cleaning": int(after["capability_level"].eq("SOTA").sum()),
                "saturated_row_count_after_cleaning": int(after["capability_level"].eq("saturated").sum()),
                "unknown_row_count_after_cleaning": int(after["capability_level"].eq("unknown").sum()),
            }
        )
    return pd.DataFrame(rows)


def _tasks_by_step_summary(tasks_by_step: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in tasks_by_step.to_dict(orient="records"):
        step = str(row.pop("mitre_step_label"))
        summary[step] = row
    return summary


def _every_step_has_levels(tasks_by_step: pd.DataFrame, levels: tuple[str, ...], suffix: str) -> bool:
    for _, row in tasks_by_step.iterrows():
        for level in levels:
            if int(row[f"{level.lower()}_row_count_{suffix}"]) <= 0:
                return False
    return True


def _sota_draws_by_step_audit(sota_df: pd.DataFrame, columns: ColumnMap) -> pd.DataFrame:
    step_col = columns["mitre_step_label"]
    model_col = columns["model"]
    expected_models = set(config.EXPECTED_LLM_MODELS)
    steps = sorted(
        set(config.EXPECTED_MITRE_STEP_LABELS)
        | {str(value) for value in sota_df[step_col].dropna().unique()}
    )
    rows: list[dict[str, Any]] = []
    for step in steps:
        step_df = sota_df.loc[sota_df[step_col].astype(str).eq(step)]
        observed_models = {str(value) for value in step_df[model_col].dropna().unique()}
        usable_rows = int(len(step_df))
        rows.append(
            {
                "mitre_step_label": step,
                "usable_row_count": usable_rows,
                "unique_draw_uid_count": int(step_df["draw_uid"].nunique()),
                "unique_model_count": int(step_df[model_col].nunique(dropna=True)),
                "missing_expected_models": "; ".join(sorted(expected_models - observed_models)),
                "missing_draw_count_relative_to_nominal_200": max(0, 200 - usable_rows),
            }
        )
    return pd.DataFrame(rows)


def _sota_steps_with_missing_rows(sota_draws_by_step: pd.DataFrame) -> dict[str, int]:
    missing = sota_draws_by_step.loc[
        sota_draws_by_step["missing_draw_count_relative_to_nominal_200"] > 0
    ]
    return {
        str(row["mitre_step_label"]): int(row["missing_draw_count_relative_to_nominal_200"])
        for row in missing.to_dict(orient="records")
    }


def _has_all_expected_models_per_step(df: pd.DataFrame, columns: ColumnMap) -> bool:
    step_col = columns["mitre_step_label"]
    model_col = columns["model"]
    expected_models = set(config.EXPECTED_LLM_MODELS)
    expected_steps = set(config.EXPECTED_MITRE_STEP_LABELS)
    for step in expected_steps:
        models_for_step = set(df.loc[df[step_col].eq(step), model_col].dropna().astype(str))
        if not expected_models.issubset(models_for_step):
            return False
    return True


def _rows_by_step_and_model(df: pd.DataFrame, columns: ColumnMap) -> pd.DataFrame:
    step_col = columns["mitre_step_label"]
    model_col = columns["model"]
    grouped = (
        df.groupby([step_col, model_col], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values([step_col, model_col])
    )
    return grouped


def _rows_by_step_and_capability(df: pd.DataFrame, columns: ColumnMap) -> pd.DataFrame:
    step_col = columns["mitre_step_label"]
    grouped = (
        df.groupby([step_col, "capability_level"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values([step_col, "capability_level"])
    )
    return grouped


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
