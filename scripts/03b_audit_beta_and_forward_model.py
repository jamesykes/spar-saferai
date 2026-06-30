"""Numerical audit for Beta fits and the OC3 DoS forward-model smoke test."""

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
from saferai_budget_recovery.forward_model import STEP_LABEL_BY_VARIABLE, oc3_dos_p_success


OUTPUT_DIR = config.FORWARD_MODEL_SMOKE_TEST_DIR
WARN_FITS_PATH = OUTPUT_DIR / "beta_warn_fits.csv"
RESIDUAL_BY_STEP_PATH = OUTPUT_DIR / "beta_fit_residual_summary_by_step.csv"
RESIDUAL_BY_MODEL_PATH = OUTPUT_DIR / "beta_fit_residual_summary_by_model.csv"
MAPPING_PATH = OUTPUT_DIR / "forward_model_mapping.json"
PLUGIN_CHECKS_PATH = OUTPUT_DIR / "plugin_forward_model_checks.json"
AUDIT_REPORT_PATH = OUTPUT_DIR / "beta_and_forward_model_audit_report.json"
SMOKE_SUMMARY_PATH = OUTPUT_DIR / "full_reference_p_success_summary.json"


def main() -> None:
    if not config.SOTA_BETA_FITS_PATH.exists():
        raise FileNotFoundError(
            f"SOTA Beta fits not found: {config.SOTA_BETA_FITS_PATH}. "
            "Run scripts/02_fit_beta_distributions.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fit_df = pd.read_csv(config.SOTA_BETA_FITS_PATH)
    _verify_forward_model_mapping(fit_df)
    fit_df = _add_abs_residuals_and_beta_mean(fit_df)

    warn_df = _warn_fits(fit_df)
    warn_df.to_csv(WARN_FITS_PATH, index=False)

    residual_by_step = _residual_summary(fit_df, "step_name")
    residual_by_model = _residual_summary(fit_df, "model")
    residual_by_step.to_csv(RESIDUAL_BY_STEP_PATH, index=False)
    residual_by_model.to_csv(RESIDUAL_BY_MODEL_PATH, index=False)

    _write_json(MAPPING_PATH, STEP_LABEL_BY_VARIABLE)
    audit_report = _audit_report(fit_df, warn_df)
    _write_json(AUDIT_REPORT_PATH, audit_report)

    plugin_checks = _plugin_forward_model_checks(fit_df)
    _write_json(PLUGIN_CHECKS_PATH, plugin_checks)

    print("Beta and forward-model audit summary")
    print(f"Warn fits: {len(warn_df)}")
    print(f"Warn-fit steps: {_compact_keys(warn_df, 'step_name')}")
    print(f"Warn-fit models: {_compact_keys(warn_df, 'model')}")
    print(f"Max RMSE: {float(fit_df['fit_rmse'].max())}")
    print("Plug-in P(success) values:")
    for name, result in plugin_checks["plugin_checks"].items():
        print(f"  {name}: {result['p_success']}")
    mc = plugin_checks.get("monte_carlo_smoke_summary")
    if mc:
        print(f"MC mean: {mc.get('mean')}")
        print(f"MC median: {mc.get('p50')}")
    else:
        print("MC summary: not available")
    print(f"Warn fits CSV: {WARN_FITS_PATH}")
    print(f"Residual summary by step: {RESIDUAL_BY_STEP_PATH}")
    print(f"Plug-in checks JSON: {PLUGIN_CHECKS_PATH}")
    print(f"Forward-model mapping JSON: {MAPPING_PATH}")


def _verify_forward_model_mapping(fit_df: pd.DataFrame) -> None:
    observed_steps = set(fit_df["step_name"].dropna().astype(str))
    missing = [label for label in STEP_LABEL_BY_VARIABLE.values() if label not in observed_steps]
    if missing:
        raise ValueError(
            "Forward-model mapping references step labels absent from fitted SOTA rows: "
            f"{missing}"
        )


def _add_abs_residuals_and_beta_mean(fit_df: pd.DataFrame) -> pd.DataFrame:
    out = fit_df.copy()
    for q in ("q25", "q50", "q75"):
        out[f"abs_residual_{q}"] = out[f"residual_{q}"].abs()
    out["beta_mean"] = out["alpha"] / (out["alpha"] + out["beta"])
    return out


def _warn_fits(fit_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "fitted_q25",
        "fitted_q50",
        "fitted_q75",
        "residual_q25",
        "residual_q50",
        "residual_q75",
        "fit_rmse",
    ]
    renamed = {
        "percentile_25th": "q25",
        "percentile_50th": "q50",
        "percentile_75th": "q75",
    }
    return (
        fit_df.loc[fit_df["fit_quality_flag"].eq("warn"), columns]
        .rename(columns=renamed)
        .sort_values("fit_rmse", ascending=False)
    )


def _residual_summary(fit_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in fit_df.groupby(group_col, dropna=False):
        row: dict[str, Any] = {
            group_col: group_value,
            "row_count": int(len(group)),
            "median_rmse": _stat(group["fit_rmse"], np.median),
            "p95_rmse": _stat(group["fit_rmse"], lambda values: np.percentile(values, 95)),
            "max_rmse": _stat(group["fit_rmse"], np.max),
        }
        for q in ("q25", "q50", "q75"):
            col = f"abs_residual_{q}"
            row[f"median_abs_residual_{q}"] = _stat(group[col], np.median)
            row[f"p95_abs_residual_{q}"] = _stat(group[col], lambda values: np.percentile(values, 95))
            row[f"max_abs_residual_{q}"] = _stat(group[col], np.max)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def _audit_report(fit_df: pd.DataFrame, warn_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "warn_fit_count": int(len(warn_df)),
        "warn_fit_count_by_mitre_step_label": _value_counts(warn_df["step_name"]),
        "warn_fit_count_by_model": _value_counts(warn_df["model"]),
        "max_rmse_by_mitre_step_label": _group_stat(fit_df, "step_name", np.max),
        "median_rmse_by_mitre_step_label": _group_stat(fit_df, "step_name", np.median),
        "p95_rmse_by_mitre_step_label": _group_stat(
            fit_df, "step_name", lambda values: np.percentile(values, 95)
        ),
        "max_rmse_by_model": _group_stat(fit_df, "model", np.max),
        "median_rmse_by_model": _group_stat(fit_df, "model", np.median),
        "p95_rmse_by_model": _group_stat(fit_df, "model", lambda values: np.percentile(values, 95)),
    }


def _plugin_forward_model_checks(fit_df: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "mean_elicited_median_by_step": _plugin_check(
            fit_df.groupby("step_name")["percentile_50th"].mean().to_dict()
        ),
        "median_elicited_median_by_step": _plugin_check(
            fit_df.groupby("step_name")["percentile_50th"].median().to_dict()
        ),
        "mean_fitted_beta_mean_by_step": _plugin_check(
            fit_df.groupby("step_name")["beta_mean"].mean().to_dict()
        ),
        "median_fitted_beta_mean_by_step": _plugin_check(
            fit_df.groupby("step_name")["beta_mean"].median().to_dict()
        ),
    }
    return {
        "note": (
            "Plug-in values are deterministic approximations and are not expected to equal "
            "the Monte Carlo mixture mean exactly; large discrepancies would indicate a "
            "possible mapping or formula issue."
        ),
        "plugin_checks": checks,
        "monte_carlo_smoke_summary": _load_mc_summary_if_available(),
    }


def _plugin_check(node_values_by_step: dict[str, float]) -> dict[str, Any]:
    kwargs = {
        variable: float(node_values_by_step[step_label])
        for variable, step_label in STEP_LABEL_BY_VARIABLE.items()
    }
    p_success = float(oc3_dos_p_success(**kwargs))
    return {
        "node_values_by_variable": kwargs,
        "node_values_by_step": {
            step_label: float(node_values_by_step[step_label])
            for step_label in STEP_LABEL_BY_VARIABLE.values()
        },
        "p_success": p_success,
    }


def _load_mc_summary_if_available() -> dict[str, Any] | None:
    if not SMOKE_SUMMARY_PATH.exists():
        return None
    summary = json.loads(SMOKE_SUMMARY_PATH.read_text(encoding="utf-8"))
    return {
        "mean": summary.get("mean"),
        "p50": summary.get("p50"),
        "p05": summary.get("p05"),
        "p95": summary.get("p95"),
    }


def _compact_keys(df: pd.DataFrame, col: str) -> str:
    if df.empty:
        return "none"
    counts = df[col].value_counts().sort_index()
    return "; ".join(f"{key} ({value})" for key, value in counts.items())


def _group_stat(fit_df: pd.DataFrame, group_col: str, fn: Any) -> dict[str, float | None]:
    return {
        str(group_value): _stat(group["fit_rmse"], fn)
        for group_value, group in fit_df.groupby(group_col, dropna=False)
    }


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _stat(series: pd.Series, fn: Any) -> float | None:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return None
    value = float(fn(values))
    return value if np.isfinite(value) else None


def _write_json(path: Path, data: Any) -> None:
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

