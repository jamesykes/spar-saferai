#!/usr/bin/env python3
"""Fit elicited uncertainty summaries row by row.

Usage
-----
Run the full pipeline on one CSV:

```bash
.venv/bin/python scripts/fit_elicited_distributions.py \
  --input-csv BN_exports_for_James/OC3_Financial_DDoS_llm_params.csv \
  --output-dir outputs/oc3_financial_ddos_fits
```

Optional row restriction for inspection:

```bash
.venv/bin/python scripts/fit_elicited_distributions.py \
  --input-csv BN_exports_for_James/OC3_Financial_DDoS_llm_params.csv \
  --output-dir outputs/oc3_financial_ddos_fits_subset \
  --row-indices 0,1,2,3
```

Outputs
-------
The script writes:

* `full_fit_results.csv`: one row per original row per fitting method
* `best_fits.csv`: one selected fit per original row
* `fit_summary.json`: aggregated summary statistics
* optional diagnostic plots if `matplotlib` is available

Method families
---------------
The code treats fitting as a comparison of plausible rules rather than one
uniquely identified methodology:

* `beta_ls_mass_mode`: Beta fit to elicited mode and interval mass
* `beta_quantile_mode`: Beta fit to implied central quantiles and mode
* `beta_concentration`: one-dimensional Beta concentration benchmark
* `pert_quantile`: Beta-PERT fit to implied central quantiles
* `pert_mass`: Beta-PERT fit to interval mass
* `pert_three_point`: baseline three-point PERT using `(low_ci, mode, high_ci)`

The implementation is intentionally row-centric. Every attempted fit returns a
result row with copied metadata, fitted parameters, optimizer status, and
diagnostics such as mode error, mass error, quantile error, and unified score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import optimize, stats

TOL = 1e-8
EPS = 1e-12
DEFAULT_PERT_LAMBDA = 4.0

IDENTITY_COLUMNS = [
    "model_id",
    "model_name",
    "node_id",
    "node_name",
    "expert",
    "benchmark",
    "benchmark_task",
]

ELICITATION_COLUMNS = [
    "low_ci",
    "high_ci",
    "mode",
    "confidence",
    "min",
    "max",
]

ALL_RESULT_COLUMNS = IDENTITY_COLUMNS + ELICITATION_COLUMNS + [
    "row_index",
    "row_type",
    "row_valid",
    "validation_errors",
    "validation_warnings",
    "method",
    "success",
    "status",
    "message",
    "alpha",
    "beta",
    "a",
    "b",
    "pert_lambda",
    "support_upper_bound_used",
    "fitted_mode",
    "fitted_mass",
    "q_low",
    "q_high",
    "fitted_low_quantile",
    "fitted_high_quantile",
    "mode_error",
    "mass_error",
    "low_error",
    "high_error",
    "loss",
    "unified_score",
]

METHOD_ORDER = [
    "beta_ls_mass_mode",
    "beta_quantile_mode",
    "beta_concentration",
    "pert_quantile",
    "pert_mass",
    "pert_three_point",
]


def validate_row(row: pd.Series) -> dict:
    """Validate one elicitation row without mutating the source data.

    Returns a dictionary with:

    * `valid`: overall boolean
    * `errors`: blocking issues
    * `warnings`: non-blocking issues worth inspecting later
    """

    errors: list[str] = []
    warnings: list[str] = []

    for column in ELICITATION_COLUMNS:
        value = row.get(column)
        if pd.isna(value):
            errors.append(f"Missing required value in `{column}`.")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])
    mode = float(row["mode"])
    confidence = float(row["confidence"])
    minimum = float(row["min"])
    maximum = float(row["max"])

    if not (0.0 < confidence < 1.0):
        errors.append("`confidence` must be strictly between 0 and 1.")

    if low_ci > mode + TOL:
        errors.append("`low_ci` must be less than or equal to `mode`.")
    if mode > high_ci + TOL:
        errors.append("`mode` must be less than or equal to `high_ci`.")

    if minimum > low_ci + TOL:
        errors.append("`min` must be less than or equal to `low_ci`.")
    if minimum > mode + TOL:
        errors.append("`min` must be less than or equal to `mode`.")

    if np.isfinite(maximum):
        if high_ci > maximum + TOL:
            errors.append("`high_ci` must be less than or equal to `max`.")
        if mode > maximum + TOL:
            errors.append("`mode` must be less than or equal to `max`.")
        if minimum > maximum + TOL:
            errors.append("`min` must be less than or equal to `max`.")
    else:
        warnings.append("`max` is non-finite; PERT support search will use a derived upper bound.")

    if low_ci > high_ci + TOL:
        errors.append("`low_ci` must be less than or equal to `high_ci`.")

    if abs(low_ci - high_ci) <= TOL:
        warnings.append("`low_ci` and `high_ci` are effectively equal; fits may be ill-conditioned.")
    if abs(mode - low_ci) <= TOL:
        warnings.append("`mode` is effectively on the lower interval boundary.")
    if abs(mode - high_ci) <= TOL:
        warnings.append("`mode` is effectively on the upper interval boundary.")

    if is_probability_row(row):
        if mode <= 0.0 + TOL or mode >= 1.0 - TOL:
            warnings.append("Probability row has boundary mode; interior-mode Beta methods will be marked invalid.")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def is_probability_row(row: pd.Series) -> bool:
    """Return True when the elicited quantity should be treated as probability-valued."""

    required = ["min", "max", "low_ci", "mode", "high_ci"]
    if any(pd.isna(row.get(column)) for column in required):
        return False

    minimum = float(row["min"])
    maximum = float(row["max"])
    low_ci = float(row["low_ci"])
    mode = float(row["mode"])
    high_ci = float(row["high_ci"])

    in_unit_interval = all(
        -TOL <= value <= 1.0 + TOL for value in (low_ci, mode, high_ci)
    )
    return minimum >= -TOL and maximum <= 1.0 + TOL and in_unit_interval


def fit_beta_ls_mass_mode(row: pd.Series) -> dict:
    """Fit a Beta distribution by least squares on elicited mode and interval mass."""

    result = make_result_template(row, "beta_ls_mass_mode")
    validation = validate_row(row)
    result.update(validation_payload(validation))

    mode = float(row["mode"])
    if not validation["valid"]:
        return fail_result(result, "invalid_row", "; ".join(validation["errors"]))
    if not (0.0 + TOL < mode < 1.0 - TOL):
        return fail_result(
            result,
            "invalid_beta_mode",
            "Interior-mode Beta fit requires `mode` strictly between 0 and 1.",
        )

    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])
    confidence = float(row["confidence"])

    alpha0, beta0 = initial_beta_guess(mode, concentration=10.0)
    x0 = np.array([math.log(alpha0 - 1.0), math.log(beta0 - 1.0)], dtype=float)

    def objective(theta: np.ndarray) -> float:
        alpha = 1.0 + np.exp(theta[0])
        beta_param = 1.0 + np.exp(theta[1])
        fitted_mode = beta_mode(alpha, beta_param)
        fitted_mass = safe_beta_interval_mass(alpha, beta_param, low_ci, high_ci)
        if not np.isfinite(fitted_mode) or not np.isfinite(fitted_mass):
            return 1e12
        mode_error = fitted_mode - mode
        mass_error = fitted_mass - confidence
        return mode_error**2 + mass_error**2

    opt_result = safe_minimize(objective, x0)
    alpha = np.nan
    beta_param = np.nan
    if opt_result is not None:
        alpha = 1.0 + np.exp(opt_result.x[0])
        beta_param = 1.0 + np.exp(opt_result.x[1])

    result.update(
        {
            "alpha": alpha,
            "beta": beta_param,
        }
    )
    populate_beta_diagnostics(result, row, alpha, beta_param)
    result["loss"] = float(opt_result.fun) if opt_result is not None and np.isfinite(opt_result.fun) else np.nan

    if opt_result is None:
        return fail_result(result, "optimization_error", "Beta least-squares optimization raised an exception.")

    result["success"] = bool(opt_result.success and np.isfinite(result["loss"]))
    result["status"] = str(opt_result.status)
    result["message"] = str(opt_result.message)
    result["unified_score"] = compute_unified_score(result)
    return result


def fit_beta_quantile_mode(row: pd.Series) -> dict:
    """Fit a Beta distribution to implied central quantiles and elicited mode."""

    result = make_result_template(row, "beta_quantile_mode")
    validation = validate_row(row)
    result.update(validation_payload(validation))

    mode = float(row["mode"])
    if not validation["valid"]:
        return fail_result(result, "invalid_row", "; ".join(validation["errors"]))
    if not (0.0 + TOL < mode < 1.0 - TOL):
        return fail_result(
            result,
            "invalid_beta_mode",
            "Interior-mode Beta fit requires `mode` strictly between 0 and 1.",
        )

    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])
    confidence = float(row["confidence"])
    q_low, q_high = implied_quantile_levels(confidence)

    alpha0, beta0 = initial_beta_guess(mode, concentration=10.0)
    x0 = np.array([math.log(alpha0 - 1.0), math.log(beta0 - 1.0)], dtype=float)

    def objective(theta: np.ndarray) -> float:
        alpha = 1.0 + np.exp(theta[0])
        beta_param = 1.0 + np.exp(theta[1])
        fitted_mode = beta_mode(alpha, beta_param)
        fitted_low = safe_beta_ppf(q_low, alpha, beta_param)
        fitted_high = safe_beta_ppf(q_high, alpha, beta_param)
        if not all(np.isfinite(value) for value in (fitted_mode, fitted_low, fitted_high)):
            return 1e12
        mode_error = fitted_mode - mode
        low_error = fitted_low - low_ci
        high_error = fitted_high - high_ci
        return mode_error**2 + low_error**2 + high_error**2

    opt_result = safe_minimize(objective, x0)
    alpha = np.nan
    beta_param = np.nan
    if opt_result is not None:
        alpha = 1.0 + np.exp(opt_result.x[0])
        beta_param = 1.0 + np.exp(opt_result.x[1])

    result.update(
        {
            "alpha": alpha,
            "beta": beta_param,
        }
    )
    populate_beta_diagnostics(result, row, alpha, beta_param)
    result["loss"] = float(opt_result.fun) if opt_result is not None and np.isfinite(opt_result.fun) else np.nan

    if opt_result is None:
        return fail_result(result, "optimization_error", "Beta quantile-mode optimization raised an exception.")

    result["success"] = bool(opt_result.success and np.isfinite(result["loss"]))
    result["status"] = str(opt_result.status)
    result["message"] = str(opt_result.message)
    result["unified_score"] = compute_unified_score(result)
    return result


def fit_beta_concentration(row: pd.Series) -> dict:
    """Fit a one-dimensional Beta concentration parameterization with fixed elicited mode."""

    result = make_result_template(row, "beta_concentration")
    validation = validate_row(row)
    result.update(validation_payload(validation))

    mode = float(row["mode"])
    if not validation["valid"]:
        return fail_result(result, "invalid_row", "; ".join(validation["errors"]))
    if not (0.0 + TOL < mode < 1.0 - TOL):
        return fail_result(
            result,
            "invalid_beta_mode",
            "Beta concentration fit requires `mode` strictly between 0 and 1.",
        )

    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])
    confidence = float(row["confidence"])
    x0 = np.array([math.log(10.0 - 2.0)], dtype=float)

    def objective(theta: np.ndarray) -> float:
        concentration = 2.0 + np.exp(theta[0])
        alpha = 1.0 + mode * (concentration - 2.0)
        beta_param = 1.0 + (1.0 - mode) * (concentration - 2.0)
        fitted_mass = safe_beta_interval_mass(alpha, beta_param, low_ci, high_ci)
        if not np.isfinite(fitted_mass):
            return 1e12
        mass_error = fitted_mass - confidence
        return mass_error**2

    opt_result = safe_minimize(objective, x0)
    alpha = np.nan
    beta_param = np.nan
    if opt_result is not None:
        concentration = 2.0 + np.exp(opt_result.x[0])
        alpha = 1.0 + mode * (concentration - 2.0)
        beta_param = 1.0 + (1.0 - mode) * (concentration - 2.0)

    result.update(
        {
            "alpha": alpha,
            "beta": beta_param,
        }
    )
    populate_beta_diagnostics(result, row, alpha, beta_param)
    result["loss"] = float(opt_result.fun) if opt_result is not None and np.isfinite(opt_result.fun) else np.nan

    if opt_result is None:
        return fail_result(result, "optimization_error", "Beta concentration optimization raised an exception.")

    result["success"] = bool(opt_result.success and np.isfinite(result["loss"]))
    result["status"] = str(opt_result.status)
    result["message"] = str(opt_result.message)
    result["unified_score"] = compute_unified_score(result)
    return result


def fit_pert_quantile(row: pd.Series) -> dict:
    """Fit a standard Beta-PERT distribution by matching implied central quantiles."""

    result = make_result_template(row, "pert_quantile")
    validation = validate_row(row)
    result.update(validation_payload(validation))

    if not validation["valid"]:
        return fail_result(result, "invalid_row", "; ".join(validation["errors"]))

    mode = float(row["mode"])
    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])
    confidence = float(row["confidence"])
    q_low, q_high = implied_quantile_levels(confidence)
    bounds_info = pert_bounds_from_row(row)
    if bounds_info["error"] is not None:
        return fail_result(result, "invalid_pert_bounds", bounds_info["error"])

    initial_guess = bounds_info["initial_guess"]
    bounds = bounds_info["bounds"]

    def objective(params: np.ndarray) -> float:
        a, b = decode_pert_params(params, bounds_info)
        penalty = pert_penalty(a, mode, b, row)
        if penalty > 0.0:
            return 1e12 + penalty
        fitted_low = pert_ppf(q_low, a, mode, b)
        fitted_high = pert_ppf(q_high, a, mode, b)
        if not np.isfinite(fitted_low) or not np.isfinite(fitted_high):
            return 1e12
        return (fitted_low - low_ci) ** 2 + (fitted_high - high_ci) ** 2

    opt_result = safe_minimize(objective, initial_guess, bounds=bounds)
    a = np.nan
    b = np.nan
    if opt_result is not None:
        a, b = decode_pert_params(opt_result.x, bounds_info)

    result.update(
        {
            "a": a,
            "b": b,
            "pert_lambda": DEFAULT_PERT_LAMBDA,
            "support_upper_bound_used": bounds_info["search_upper_bound"],
        }
    )
    populate_pert_diagnostics(result, row, a, mode, b)
    result["loss"] = float(opt_result.fun) if opt_result is not None and np.isfinite(opt_result.fun) else np.nan

    if opt_result is None:
        return fail_result(result, "optimization_error", "PERT quantile optimization raised an exception.")

    result["success"] = bool(opt_result.success and np.isfinite(result["loss"]))
    result["status"] = str(opt_result.status)
    result["message"] = str(opt_result.message)
    result["unified_score"] = compute_unified_score(result)
    return result


def fit_pert_mass(row: pd.Series) -> dict:
    """Fit a standard Beta-PERT distribution by matching elicited interval mass."""

    result = make_result_template(row, "pert_mass")
    validation = validate_row(row)
    result.update(validation_payload(validation))

    if not validation["valid"]:
        return fail_result(result, "invalid_row", "; ".join(validation["errors"]))

    mode = float(row["mode"])
    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])
    confidence = float(row["confidence"])
    bounds_info = pert_bounds_from_row(row)
    if bounds_info["error"] is not None:
        return fail_result(result, "invalid_pert_bounds", bounds_info["error"])

    initial_guess = bounds_info["initial_guess"]
    bounds = bounds_info["bounds"]

    def objective(params: np.ndarray) -> float:
        a, b = decode_pert_params(params, bounds_info)
        penalty = pert_penalty(a, mode, b, row)
        if penalty > 0.0:
            return 1e12 + penalty
        fitted_mass = pert_cdf(high_ci, a, mode, b) - pert_cdf(low_ci, a, mode, b)
        if not np.isfinite(fitted_mass):
            return 1e12
        return (fitted_mass - confidence) ** 2

    opt_result = safe_minimize(objective, initial_guess, bounds=bounds)
    a = np.nan
    b = np.nan
    if opt_result is not None:
        a, b = decode_pert_params(opt_result.x, bounds_info)

    result.update(
        {
            "a": a,
            "b": b,
            "pert_lambda": DEFAULT_PERT_LAMBDA,
            "support_upper_bound_used": bounds_info["search_upper_bound"],
        }
    )
    populate_pert_diagnostics(result, row, a, mode, b)
    result["loss"] = float(opt_result.fun) if opt_result is not None and np.isfinite(opt_result.fun) else np.nan

    if opt_result is None:
        return fail_result(result, "optimization_error", "PERT mass optimization raised an exception.")

    result["success"] = bool(opt_result.success and np.isfinite(result["loss"]))
    result["status"] = str(opt_result.status)
    result["message"] = str(opt_result.message)
    result["unified_score"] = compute_unified_score(result)
    return result


def fit_pert_three_point(row: pd.Series) -> dict:
    """Construct the baseline three-point PERT using `low_ci`, `mode`, and `high_ci` as support."""

    result = make_result_template(row, "pert_three_point")
    validation = validate_row(row)
    result.update(validation_payload(validation))

    if not validation["valid"]:
        return fail_result(result, "invalid_row", "; ".join(validation["errors"]))

    a = float(row["low_ci"])
    mode = float(row["mode"])
    b = float(row["high_ci"])

    if b < a - TOL:
        return fail_result(result, "invalid_pert_bounds", "PERT three-point baseline requires `low_ci <= high_ci`.")

    result.update(
        {
            "a": a,
            "b": b,
            "pert_lambda": DEFAULT_PERT_LAMBDA,
            "support_upper_bound_used": np.nan,
            "success": True,
            "status": "baseline",
            "message": "Constructed directly from elicited three-point inputs.",
        }
    )
    populate_pert_diagnostics(result, row, a, mode, b)
    result["loss"] = np.nan
    result["unified_score"] = compute_unified_score(result)
    return result


def fit_row(row: pd.Series) -> list[dict]:
    """Fit all applicable candidate methods for a single row and return result dicts."""

    row_type = "probability" if is_probability_row(row) else "non_probability"
    methods = (
        [fit_beta_ls_mass_mode, fit_beta_quantile_mode, fit_beta_concentration]
        if row_type == "probability"
        else [fit_pert_quantile, fit_pert_mass, fit_pert_three_point]
    )

    results = []
    for method in methods:
        fitted = method(row)
        fitted["row_type"] = row_type
        if np.isnan(fitted["unified_score"]) and fitted["success"]:
            fitted["unified_score"] = compute_unified_score(fitted)
        results.append(enforce_result_columns(fitted))
    return results


def select_best_fit(fit_results_for_one_row: list[dict]) -> dict:
    """Select the preferred fit for a row using unified score and explicit tie-breaking."""

    if not fit_results_for_one_row:
        raise ValueError("`fit_results_for_one_row` must contain at least one result.")

    successful = [result for result in fit_results_for_one_row if bool(result.get("success"))]
    if not successful:
        summary = dict(fit_results_for_one_row[0])
        summary.update(
            {
                "method": "",
                "success": False,
                "status": "no_successful_fit",
                "message": "All candidate methods failed for this row.",
                "alpha": np.nan,
                "beta": np.nan,
                "a": np.nan,
                "b": np.nan,
                "pert_lambda": np.nan,
                "support_upper_bound_used": np.nan,
                "fitted_mode": np.nan,
                "fitted_mass": np.nan,
                "fitted_low_quantile": np.nan,
                "fitted_high_quantile": np.nan,
                "mode_error": np.nan,
                "mass_error": np.nan,
                "low_error": np.nan,
                "high_error": np.nan,
                "loss": np.nan,
                "unified_score": np.nan,
            }
        )
        return enforce_result_columns(summary)

    def sort_key(result: dict) -> tuple[float, int]:
        score = result.get("unified_score")
        if pd.isna(score):
            score = np.inf
        method_rank = METHOD_ORDER.index(result["method"]) if result["method"] in METHOD_ORDER else len(METHOD_ORDER)
        return (float(score), method_rank)

    return enforce_result_columns(min(successful, key=sort_key))


def main(input_csv_path: str | Path, output_dir: str | Path) -> None:
    """Run the full fitting pipeline and write outputs to `output_dir`."""

    run_pipeline(
        input_csv_path=input_csv_path,
        output_dir=output_dir,
        row_indices=None,
        max_rows=None,
        make_plots=True,
    )


def run_pipeline(
    input_csv_path: str | Path,
    output_dir: str | Path,
    row_indices: list[int] | None,
    max_rows: int | None,
    make_plots: bool,
) -> None:
    """Internal pipeline implementation used by both `main` and the CLI."""

    input_path = Path(input_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = subset_dataframe(df, row_indices=row_indices, max_rows=max_rows)

    full_results: list[dict] = []
    best_results: list[dict] = []

    probability_row_count = 0
    non_probability_row_count = 0
    valid_row_count = 0
    invalid_row_count = 0

    for row_index, row in df.iterrows():
        validation = validate_row(row)
        if validation["valid"]:
            valid_row_count += 1
        else:
            invalid_row_count += 1

        if is_probability_row(row):
            probability_row_count += 1
        else:
            non_probability_row_count += 1

        fit_results = fit_row(row)
        full_results.extend(fit_results)
        best_results.append(select_best_fit(fit_results))

    full_df = pd.DataFrame(full_results)
    best_df = pd.DataFrame(best_results)
    full_df = full_df.reindex(columns=ALL_RESULT_COLUMNS)
    best_df = best_df.reindex(columns=ALL_RESULT_COLUMNS)

    full_csv_path = output_path / "full_fit_results.csv"
    best_csv_path = output_path / "best_fits.csv"
    summary_json_path = output_path / "fit_summary.json"

    full_df.to_csv(full_csv_path, index=False)
    best_df.to_csv(best_csv_path, index=False)

    summary = build_summary(
        full_df=full_df,
        best_df=best_df,
        total_rows=len(df),
        valid_rows=valid_row_count,
        invalid_rows=invalid_row_count,
        probability_rows=probability_row_count,
        non_probability_rows=non_probability_row_count,
    )
    summary_json_path.write_text(json.dumps(summary, indent=2))

    if make_plots:
        maybe_make_plots(full_df, best_df, output_path)

    print_console_summary(
        total_rows=len(df),
        probability_rows=probability_row_count,
        non_probability_rows=non_probability_row_count,
        full_df=full_df,
        best_df=best_df,
    )


def make_result_template(row: pd.Series, method: str) -> dict:
    """Create the shared row-level result payload used by every fitting method."""

    payload = {column: row.get(column, np.nan) for column in IDENTITY_COLUMNS + ELICITATION_COLUMNS}
    payload.update(
        {
            "row_index": int(row.name),
            "row_type": "probability" if is_probability_row(row) else "non_probability",
            "row_valid": False,
            "validation_errors": "",
            "validation_warnings": "",
            "method": method,
            "success": False,
            "status": "",
            "message": "",
            "alpha": np.nan,
            "beta": np.nan,
            "a": np.nan,
            "b": np.nan,
            "pert_lambda": np.nan,
            "support_upper_bound_used": np.nan,
            "fitted_mode": np.nan,
            "fitted_mass": np.nan,
            "q_low": np.nan,
            "q_high": np.nan,
            "fitted_low_quantile": np.nan,
            "fitted_high_quantile": np.nan,
            "mode_error": np.nan,
            "mass_error": np.nan,
            "low_error": np.nan,
            "high_error": np.nan,
            "loss": np.nan,
            "unified_score": np.nan,
        }
    )
    return payload


def validation_payload(validation: dict) -> dict:
    """Convert validation metadata to flat CSV-friendly fields."""

    return {
        "row_valid": bool(validation["valid"]),
        "validation_errors": " | ".join(validation["errors"]),
        "validation_warnings": " | ".join(validation["warnings"]),
    }


def fail_result(result: dict, status: str, message: str) -> dict:
    """Mark a fit result as failed while preserving any already-populated diagnostics."""

    result["success"] = False
    result["status"] = status
    result["message"] = message
    result["unified_score"] = compute_unified_score(result)
    return result


def enforce_result_columns(result: dict) -> dict:
    """Ensure every result has the complete set of expected output columns."""

    completed = dict(result)
    for column in ALL_RESULT_COLUMNS:
        completed.setdefault(column, np.nan)
    return completed


def implied_quantile_levels(confidence: float) -> tuple[float, float]:
    """Map interval confidence to central quantile levels."""

    q_low = (1.0 - confidence) / 2.0
    q_high = 1.0 - q_low
    return q_low, q_high


def initial_beta_guess(mode: float, concentration: float) -> tuple[float, float]:
    """Return a simple interior Beta initialization from mode and concentration."""

    alpha0 = 1.0 + mode * (concentration - 2.0)
    beta0 = 1.0 + (1.0 - mode) * (concentration - 2.0)
    return max(alpha0, 1.0 + EPS), max(beta0, 1.0 + EPS)


def beta_mode(alpha: float, beta_param: float) -> float:
    """Return the interior mode of a Beta distribution when defined."""

    if alpha > 1.0 and beta_param > 1.0:
        return (alpha - 1.0) / (alpha + beta_param - 2.0)
    return np.nan


def safe_beta_interval_mass(alpha: float, beta_param: float, low_ci: float, high_ci: float) -> float:
    """Return Beta interval mass with defensive NaN handling."""

    try:
        low_tail = stats.beta.cdf(low_ci, alpha, beta_param)
        high_tail = stats.beta.cdf(high_ci, alpha, beta_param)
        mass = high_tail - low_tail
        return float(mass) if np.isfinite(mass) else np.nan
    except Exception:
        return np.nan


def safe_beta_ppf(q: float, alpha: float, beta_param: float) -> float:
    """Return a Beta quantile with defensive NaN handling."""

    try:
        value = stats.beta.ppf(q, alpha, beta_param)
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def populate_beta_diagnostics(result: dict, row: pd.Series, alpha: float, beta_param: float) -> None:
    """Populate common Beta diagnostics for a fitted parameter pair."""

    if not np.isfinite(alpha) or not np.isfinite(beta_param):
        return

    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])
    mode = float(row["mode"])
    confidence = float(row["confidence"])
    q_low, q_high = implied_quantile_levels(confidence)

    fitted_mode = beta_mode(alpha, beta_param)
    fitted_mass = safe_beta_interval_mass(alpha, beta_param, low_ci, high_ci)
    fitted_low = safe_beta_ppf(q_low, alpha, beta_param)
    fitted_high = safe_beta_ppf(q_high, alpha, beta_param)

    result.update(
        {
            "q_low": q_low,
            "q_high": q_high,
            "fitted_mode": fitted_mode,
            "fitted_mass": fitted_mass,
            "fitted_low_quantile": fitted_low,
            "fitted_high_quantile": fitted_high,
            "mode_error": fitted_mode - mode if np.isfinite(fitted_mode) else np.nan,
            "mass_error": fitted_mass - confidence if np.isfinite(fitted_mass) else np.nan,
            "low_error": fitted_low - low_ci if np.isfinite(fitted_low) else np.nan,
            "high_error": fitted_high - high_ci if np.isfinite(fitted_high) else np.nan,
        }
    )


def pert_shape_parameters(a: float, mode: float, b: float, pert_lambda: float = DEFAULT_PERT_LAMBDA) -> tuple[float, float]:
    """Return Beta shape parameters for the transformed Beta-PERT distribution."""

    width = b - a
    if not np.isfinite(width) or width <= TOL:
        return np.nan, np.nan
    alpha = 1.0 + pert_lambda * (mode - a) / width
    beta_param = 1.0 + pert_lambda * (b - mode) / width
    if alpha <= 0.0 or beta_param <= 0.0:
        return np.nan, np.nan
    return alpha, beta_param


def pert_cdf(x: float, a: float, mode: float, b: float, pert_lambda: float = DEFAULT_PERT_LAMBDA) -> float:
    """CDF for a standard Beta-PERT distribution on support `[a, b]`."""

    alpha, beta_param = pert_shape_parameters(a, mode, b, pert_lambda)
    if not np.isfinite(alpha) or not np.isfinite(beta_param):
        return np.nan
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    scaled = (x - a) / (b - a)
    return float(stats.beta.cdf(scaled, alpha, beta_param))


def pert_ppf(q: float, a: float, mode: float, b: float, pert_lambda: float = DEFAULT_PERT_LAMBDA) -> float:
    """PPF for a standard Beta-PERT distribution on support `[a, b]`."""

    alpha, beta_param = pert_shape_parameters(a, mode, b, pert_lambda)
    if not np.isfinite(alpha) or not np.isfinite(beta_param):
        return np.nan
    scaled = stats.beta.ppf(q, alpha, beta_param)
    return float(a + (b - a) * scaled) if np.isfinite(scaled) else np.nan


def pert_pdf(x: float, a: float, mode: float, b: float, pert_lambda: float = DEFAULT_PERT_LAMBDA) -> float:
    """PDF for a standard Beta-PERT distribution on support `[a, b]`."""

    alpha, beta_param = pert_shape_parameters(a, mode, b, pert_lambda)
    if not np.isfinite(alpha) or not np.isfinite(beta_param):
        return np.nan
    if x < a or x > b or b - a <= TOL:
        return 0.0
    scaled = (x - a) / (b - a)
    density = stats.beta.pdf(scaled, alpha, beta_param) / (b - a)
    return float(density) if np.isfinite(density) else np.nan


def populate_pert_diagnostics(result: dict, row: pd.Series, a: float, mode: float, b: float) -> None:
    """Populate common Beta-PERT diagnostics for one fitted support pair."""

    if not (np.isfinite(a) and np.isfinite(b)):
        return

    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])
    confidence = float(row["confidence"])
    q_low, q_high = implied_quantile_levels(confidence)

    fitted_mass = pert_cdf(high_ci, a, mode, b) - pert_cdf(low_ci, a, mode, b)
    fitted_low = pert_ppf(q_low, a, mode, b)
    fitted_high = pert_ppf(q_high, a, mode, b)
    fitted_mode = mode if a - TOL <= mode <= b + TOL else np.nan

    result.update(
        {
            "q_low": q_low,
            "q_high": q_high,
            "fitted_mode": fitted_mode,
            "fitted_mass": fitted_mass if np.isfinite(fitted_mass) else np.nan,
            "fitted_low_quantile": fitted_low,
            "fitted_high_quantile": fitted_high,
            "mode_error": fitted_mode - mode if np.isfinite(fitted_mode) else np.nan,
            "mass_error": fitted_mass - confidence if np.isfinite(fitted_mass) else np.nan,
            "low_error": fitted_low - low_ci if np.isfinite(fitted_low) else np.nan,
            "high_error": fitted_high - high_ci if np.isfinite(fitted_high) else np.nan,
        }
    )


def pert_bounds_from_row(row: pd.Series) -> dict:
    """Derive optimization bounds and an initial guess for PERT support fitting."""

    minimum = float(row["min"])
    maximum = float(row["max"])
    mode = float(row["mode"])
    low_ci = float(row["low_ci"])
    high_ci = float(row["high_ci"])

    if minimum > mode + TOL:
        return {"error": "`min` exceeds `mode`, so PERT support is infeasible."}

    if np.isfinite(maximum) and maximum < mode - TOL:
        return {"error": "`max` is less than `mode`, so PERT support is infeasible."}

    if np.isfinite(maximum):
        search_upper_bound = maximum
    else:
        spread = max(
            1.0,
            abs(high_ci - low_ci),
            abs(mode - low_ci),
            abs(high_ci - mode),
        )
        search_upper_bound = max(high_ci + 10.0 * spread, mode + spread)

    if search_upper_bound < mode - TOL:
        return {"error": "Derived PERT upper bound is below the elicited mode."}

    lower_a = minimum
    upper_a = mode
    lower_b = mode
    upper_b = search_upper_bound

    a0_real = float(np.clip(low_ci, lower_a, upper_a))
    b0_real = float(np.clip(high_ci, lower_b, upper_b))

    if b0_real - a0_real <= TOL:
        if upper_b - lower_a <= TOL:
            return {"error": "PERT support bounds collapse to zero width."}
        a0_real = lower_a
        b0_real = upper_b

    a_span = max(upper_a - lower_a, TOL)
    b_span = max(upper_b - lower_b, TOL)
    a0_unit = (a0_real - lower_a) / a_span
    b0_unit = (b0_real - lower_b) / b_span

    return {
        "error": None,
        "real_bounds": [(lower_a, upper_a), (lower_b, upper_b)],
        "bounds": [(0.0, 1.0), (0.0, 1.0)],
        "initial_guess": np.array([a0_unit, b0_unit], dtype=float),
        "search_upper_bound": search_upper_bound if not np.isfinite(maximum) else np.nan,
    }


def decode_pert_params(params: np.ndarray, bounds_info: dict) -> tuple[float, float]:
    """Map unit-box optimization parameters back to real PERT support values."""

    (lower_a, upper_a), (lower_b, upper_b) = bounds_info["real_bounds"]
    u = float(np.clip(params[0], 0.0, 1.0))
    v = float(np.clip(params[1], 0.0, 1.0))
    a = lower_a + u * (upper_a - lower_a)
    b = lower_b + v * (upper_b - lower_b)
    return float(a), float(b)


def pert_penalty(a: float, mode: float, b: float, row: pd.Series) -> float:
    """Soft penalty for numerically invalid PERT support combinations."""

    minimum = float(row["min"])
    maximum = float(row["max"])

    penalty = 0.0
    if a < minimum - TOL:
        penalty += (minimum - a) ** 2
    if a > mode + TOL:
        penalty += (a - mode) ** 2
    if b < mode - TOL:
        penalty += (mode - b) ** 2
    if np.isfinite(maximum) and b > maximum + TOL:
        penalty += (b - maximum) ** 2
    if b - a <= TOL:
        penalty += (TOL - (b - a) + 1.0) ** 2
    return penalty


def safe_minimize(
    objective,
    x0: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
) -> optimize.OptimizeResult | None:
    """Run a local optimization with defensive exception handling."""

    try:
        return optimize.minimize(
            objective,
            x0=np.asarray(x0, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
    except Exception:
        return None


def compute_unified_score(result: dict) -> float:
    """Compute the transparent comparison score from the defined diagnostic errors."""

    terms = []
    for key in ("mode_error", "mass_error", "low_error", "high_error"):
        value = result.get(key, np.nan)
        if pd.notna(value) and np.isfinite(value):
            terms.append(float(value) ** 2)
    return float(sum(terms)) if terms else np.nan


def subset_dataframe(df: pd.DataFrame, row_indices: list[int] | None, max_rows: int | None) -> pd.DataFrame:
    """Optionally restrict processing to selected rows for easier inspection."""

    if row_indices is not None:
        df = df.loc[row_indices]
    if max_rows is not None:
        df = df.head(max_rows)
    return df.copy()


def build_summary(
    full_df: pd.DataFrame,
    best_df: pd.DataFrame,
    total_rows: int,
    valid_rows: int,
    invalid_rows: int,
    probability_rows: int,
    non_probability_rows: int,
) -> dict:
    """Build a JSON-serializable summary of pipeline outcomes."""

    successful_full = full_df[full_df["success"] == True]  # noqa: E712
    successful_best = best_df[best_df["success"] == True]  # noqa: E712

    success_counts = (
        full_df.groupby("method")["success"]
        .sum()
        .reindex(METHOD_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    best_method_counts = (
        successful_best.groupby("method")
        .size()
        .reindex(METHOD_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    average_scores = (
        successful_full.groupby("method")["unified_score"]
        .mean()
        .reindex(METHOD_ORDER)
        .to_dict()
    )
    rows_with_all_failures = int((best_df["success"] == False).sum())  # noqa: E712

    return {
        "total_rows": int(total_rows),
        "valid_rows": int(valid_rows),
        "invalid_rows": int(invalid_rows),
        "probability_rows": int(probability_rows),
        "non_probability_rows": int(non_probability_rows),
        "successful_fits_by_method": success_counts,
        "best_fit_selections_by_method": best_method_counts,
        "average_unified_score_by_method": {
            method: (None if pd.isna(value) else float(value))
            for method, value in average_scores.items()
        },
        "rows_where_all_methods_failed": rows_with_all_failures,
    }


def maybe_make_plots(full_df: pd.DataFrame, best_df: pd.DataFrame, output_path: Path) -> None:
    """Write simple diagnostic plots when matplotlib is available."""

    try:
        mpl_config_dir = output_path / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional at runtime
        return

    successful = full_df[(full_df["success"] == True) & np.isfinite(full_df["unified_score"])]  # noqa: E712
    if not successful.empty:
        plt.figure(figsize=(10, 6))
        for method in METHOD_ORDER:
            method_scores = successful.loc[successful["method"] == method, "unified_score"]
            if not method_scores.empty:
                plt.hist(method_scores, bins=30, alpha=0.5, label=method)
        plt.xlabel("Unified score")
        plt.ylabel("Count")
        plt.title("Unified score distribution by method")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path / "unified_score_histogram.png", dpi=150)
        plt.close()

    best_counts = (
        best_df[best_df["success"] == True]
        .groupby("method")
        .size()
        .reindex(METHOD_ORDER, fill_value=0)
    )
    plt.figure(figsize=(10, 5))
    best_counts.plot(kind="bar")
    plt.ylabel("Rows selected")
    plt.title("Best-fit method counts")
    plt.tight_layout()
    plt.savefig(output_path / "best_fit_method_counts.png", dpi=150)
    plt.close()

    failure_counts = (
        full_df.groupby("method")["success"]
        .apply(lambda series: int((series == False).sum()))  # noqa: E712
        .reindex(METHOD_ORDER, fill_value=0)
    )
    plt.figure(figsize=(10, 5))
    failure_counts.plot(kind="bar")
    plt.ylabel("Failed fits")
    plt.title("Failure counts by method")
    plt.tight_layout()
    plt.savefig(output_path / "failure_counts_by_method.png", dpi=150)
    plt.close()


def print_console_summary(
    total_rows: int,
    probability_rows: int,
    non_probability_rows: int,
    full_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> None:
    """Print the concise console summary requested by the user."""

    success_counts = (
        full_df.groupby("method")["success"]
        .sum()
        .reindex(METHOD_ORDER, fill_value=0)
        .astype(int)
    )
    best_counts = (
        best_df[best_df["success"] == True]
        .groupby("method")
        .size()
        .reindex(METHOD_ORDER, fill_value=0)
        .astype(int)
    )
    no_successful_fit_count = int((best_df["success"] == False).sum())  # noqa: E712

    print(f"Rows processed: {total_rows}")
    print(f"Probability rows: {probability_rows}")
    print(f"Non-probability rows: {non_probability_rows}")
    print("Successful fits by method:")
    for method, count in success_counts.items():
        print(f"  {method}: {count}")
    print("Best-method counts:")
    for method, count in best_counts.items():
        print(f"  {method}: {count}")
    print(f"Rows with no successful fit: {no_successful_fit_count}")


def parse_row_indices(raw: str | None) -> list[int] | None:
    """Parse a comma-separated list of row indices for CLI subsetting."""

    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Path to the elicitation CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON/plot outputs.")
    parser.add_argument(
        "--row-indices",
        default=None,
        help="Optional comma-separated row indices to process for focused inspection.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows processed after any row-index filtering.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip optional plot generation.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Iterable[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    run_pipeline(
        input_csv_path=args.input_csv,
        output_dir=args.output_dir,
        row_indices=parse_row_indices(args.row_indices),
        max_rows=args.max_rows,
        make_plots=not args.skip_plots,
    )


if __name__ == "__main__":
    run_cli()
