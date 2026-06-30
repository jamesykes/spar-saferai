"""Fit Beta distributions to cleaned SOTA elicitation quartiles."""

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
from saferai_budget_recovery.beta_fit import fit_beta_distributions


SOTA_CLEANED_PATH = config.PROCESSED_DATA_PATH.parent / "cleaned_sota_rows.csv"
REPORT_PATH = config.FITTED_DISTRIBUTIONS_DIR / "sota_beta_fit_report.json"


def main() -> None:
    if not SOTA_CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned SOTA rows not found: {SOTA_CLEANED_PATH}. Run scripts/01_clean_data.py first."
        )

    config.FITTED_DISTRIBUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    sota_df = pd.read_csv(SOTA_CLEANED_PATH)
    fit_df = fit_beta_distributions(sota_df)
    fit_df.to_csv(config.SOTA_BETA_FITS_PATH, index=False)

    report = _make_report(fit_df)
    REPORT_PATH.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Beta fitting summary")
    print(f"Input SOTA rows: {report['input_sota_rows']}")
    print(f"OK fits: {report['n_ok_fits']}")
    print(f"Warn fits: {report['n_warn_fits']}")
    print(f"Fail fits: {report['n_fail_fits']}")
    print(f"Rows with clipped quartiles: {report['n_rows_with_clipped_quartiles']}")
    print(f"Median RMSE: {report['median_rmse']}")
    print(f"95th percentile RMSE: {report['p95_rmse']}")
    print(f"Max RMSE: {report['max_rmse']}")
    print(f"Fitted CSV: {config.SOTA_BETA_FITS_PATH}")
    print(f"Fit report: {REPORT_PATH}")


def _make_report(fit_df: pd.DataFrame) -> dict[str, Any]:
    rmse = fit_df["fit_rmse"].dropna()
    worst_cols = [
        "step_name",
        "model",
        "run_id",
        "repeat_index",
        "draw_uid",
        "percentile_25th",
        "percentile_50th",
        "percentile_75th",
        "alpha",
        "beta",
        "fit_rmse",
        "fit_quality_flag",
    ]
    worst_rows = (
        fit_df.sort_values("fit_rmse", ascending=False, na_position="first")
        .head(20)[worst_cols]
        .to_dict(orient="records")
    )
    return {
        "input_sota_rows": int(len(fit_df)),
        "n_ok_fits": int(fit_df["fit_quality_flag"].eq("ok").sum()),
        "n_warn_fits": int(fit_df["fit_quality_flag"].eq("warn").sum()),
        "n_fail_fits": int(fit_df["fit_quality_flag"].eq("fail").sum()),
        "max_rmse": _nan_stat(rmse, np.max),
        "median_rmse": _nan_stat(rmse, np.median),
        "p95_rmse": _nan_stat(rmse, lambda values: np.percentile(values, 95)),
        "worst_20_rmse_rows": worst_rows,
        "n_rows_with_clipped_quartiles": int(fit_df["quartiles_were_clipped"].sum()),
        "count_by_mitre_step_label": _value_counts(fit_df["step_name"]),
        "count_by_fit_quality_flag": _value_counts(fit_df["fit_quality_flag"]),
        "count_by_model": _value_counts(fit_df["model"]),
    }


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _nan_stat(series: pd.Series, fn: Any) -> float | None:
    if series.empty:
        return None
    value = float(fn(series.to_numpy(dtype=float)))
    return value if np.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
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

