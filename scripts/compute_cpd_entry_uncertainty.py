#!/usr/bin/env python3
"""Aggregate expert-level fitted distributions into cpd_entry-level uncertainty metrics.

Usage
-----
Run on the fitted `best_fits.csv` from the previous pipeline:

```bash
.venv/bin/python scripts/compute_cpd_entry_uncertainty.py \
  --input-csv outputs/oc3_financial_ddos_fits/best_fits.csv \
  --output-dir outputs/oc3_financial_ddos_cpd_entry_uncertainty
```

Interpretation of the main measures
----------------------------------
* `mixture_variance`: total uncertainty of the equally weighted mixture of
  expert distributions.
* `mean_within_expert_variance`: average internal uncertainty of experts.
* `between_expert_variance_of_means`: disagreement in central tendency across
  experts.
* Wasserstein measures: distribution-level disagreement across experts,
  computed from expert quantile functions on a dense probability grid.

Outputs
-------
* `cpd_entry_uncertainty.csv`: one row per cpd_entry with classical and
  Wasserstein disagreement measures.
* `pairwise_wasserstein_long.csv`: one row per expert pair within each cpd_entry.
* `cpd_entry_summary.json`: aggregate dataset-level summary statistics.
* `cpd_entry_rankings.csv`: rank cpd_entry rows by several disagreement metrics.

Additional diagnostics written for inspection:
* `excluded_fitted_rows.csv`: rows skipped before grouping, with reasons.
* `cpd_entry_warnings.csv`: cpd_entry-level warnings and notes.
* Optional quantile-curve plots for selected cpd_entry groups.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


TOL = 1e-8
GRID_EPS = 1e-6
GRID_SIZE = 2001
NEARLY_IDENTICAL_W2_THRESHOLD = 1e-6
LARGE_DISAGREEMENT_W1_THRESHOLD = 0.15
LARGE_DISAGREEMENT_W2_THRESHOLD = 0.15

CPD_ENTRY_IDENTITY_COLUMNS = [
    "model_id",
    "model_name",
    "node_id",
    "node_name",
    "benchmark",
    "benchmark_task",
]

EXPERT_IDENTITY_COLUMNS = CPD_ENTRY_IDENTITY_COLUMNS + ["expert"]

DEFAULT_OUTPUT_COLUMNS = CPD_ENTRY_IDENTITY_COLUMNS + [
    "n_experts_used",
    "experts_used",
    "best_methods_used",
    "distribution_families_used",
    "sum_within_expert_variances",
    "mean_within_expert_variance",
    "mixture_mean",
    "between_expert_variance_of_means",
    "mixture_second_moment",
    "mixture_variance",
    "mixture_variance_law_total_variance",
    "mixture_variance_consistency_gap",
    "mean_pairwise_w1",
    "mean_pairwise_w1_sq",
    "mean_pairwise_w2",
    "mean_pairwise_w2_sq",
    "max_pairwise_w1",
    "max_pairwise_w2",
    "frechet_dispersion_w1",
    "frechet_variance_like_w1",
    "frechet_dispersion_w2",
    "frechet_variance_w2",
    "min_expert_mean",
    "max_expert_mean",
    "std_expert_means",
    "min_expert_variance",
    "max_expert_variance",
    "warning_flags",
]


def load_best_fits(path: str | Path) -> pd.DataFrame:
    """Load `best_fits.csv` and normalize a few schema differences."""

    df = pd.read_csv(path)

    if "best_method" not in df.columns and "method" in df.columns:
        df["best_method"] = df["method"]

    if "success" in df.columns:
        df["success"] = df["success"].apply(coerce_truthy)
    else:
        df["success"] = False

    return df


def validate_beta_row(row: pd.Series) -> dict:
    """Validate one expert-level row for Beta-based cpd_entry aggregation."""

    errors: list[str] = []
    warnings: list[str] = []

    for column in EXPERT_IDENTITY_COLUMNS + ["alpha", "beta", "success", "min", "max"]:
        if column not in row.index:
            errors.append(f"Missing required column `{column}`.")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    if not coerce_truthy(row.get("success")):
        errors.append("`success` is not truthy.")

    alpha = row.get("alpha")
    beta_param = row.get("beta")
    if pd.isna(alpha) or pd.isna(beta_param):
        errors.append("Missing `alpha` or `beta`.")
    else:
        alpha = float(alpha)
        beta_param = float(beta_param)
        if alpha <= 0.0 or beta_param <= 0.0:
            errors.append("`alpha` and `beta` must both be strictly positive.")

    minimum = row.get("min")
    maximum = row.get("max")
    if pd.isna(minimum) or pd.isna(maximum):
        errors.append("Missing support bounds `min` or `max`.")
    else:
        minimum = float(minimum)
        maximum = float(maximum)
        if minimum < -TOL or maximum > 1.0 + TOL:
            errors.append("Support is not effectively within [0, 1].")
        if minimum > 0.0 + TOL or maximum < 1.0 - TOL:
            warnings.append("Support differs slightly from exact [0, 1].")

    for column in ("low_ci", "mode", "high_ci"):
        if column in row.index and pd.notna(row[column]):
            value = float(row[column])
            if value < -TOL or value > 1.0 + TOL:
                errors.append(f"`{column}` lies outside [0, 1].")

    row_type = row.get("row_type")
    if pd.notna(row_type) and str(row_type) != "probability":
        warnings.append(f"`row_type` is `{row_type}`, not `probability`.")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def infer_distribution_family(row: pd.Series) -> str | None:
    """Infer the fitted distribution family stored on one best-fit row."""

    alpha = row.get("alpha")
    beta_param = row.get("beta")
    a = row.get("a")
    b = row.get("b")
    mode = row.get("mode")

    if pd.notna(alpha) and pd.notna(beta_param):
        return "beta"
    if pd.notna(a) and pd.notna(b) and pd.notna(mode):
        return "pert"
    return None


def validate_fitted_row(row: pd.Series) -> dict:
    """Validate one expert-level row for family-aware cpd_entry aggregation."""

    errors: list[str] = []
    warnings: list[str] = []

    for column in EXPERT_IDENTITY_COLUMNS + ["success"]:
        if column not in row.index:
            errors.append(f"Missing required column `{column}`.")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    if not coerce_truthy(row.get("success")):
        errors.append("`success` is not truthy.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    family = infer_distribution_family(row)
    if family is None:
        errors.append("Row does not contain a usable fitted Beta or PERT parameterization.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if family == "beta":
        beta_validation = validate_beta_row(row)
        return beta_validation

    a = row.get("a")
    b = row.get("b")
    mode = row.get("mode")
    pert_lambda = row.get("pert_lambda")

    if pd.isna(a) or pd.isna(b) or pd.isna(mode):
        errors.append("PERT row requires `a`, `b`, and `mode`.")
    else:
        a = float(a)
        b = float(b)
        mode = float(mode)
        if b <= a:
            errors.append("PERT row requires `b > a`.")
        if mode < a - TOL or mode > b + TOL:
            errors.append("PERT row requires `a <= mode <= b`.")

    if pd.isna(pert_lambda):
        warnings.append("Missing `pert_lambda`; defaulting to 4.0.")
    else:
        pert_lambda = float(pert_lambda)
        if pert_lambda <= 0.0:
            errors.append("`pert_lambda` must be strictly positive.")

    minimum = row.get("min")
    maximum = row.get("max")
    if pd.notna(minimum) and pd.notna(a) and float(a) < float(minimum) - TOL:
        warnings.append("PERT support lower bound is below elicited `min`.")
    if pd.notna(maximum) and np.isfinite(maximum) and pd.notna(b) and float(b) > float(maximum) + TOL:
        warnings.append("PERT support upper bound exceeds elicited `max`.")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def beta_mean(alpha: float, beta: float) -> float:
    """Return the mean of `Beta(alpha, beta)`."""

    return float(alpha / (alpha + beta))


def beta_variance(alpha: float, beta: float) -> float:
    """Return the variance of `Beta(alpha, beta)`."""

    total = alpha + beta
    return float(alpha * beta / ((total**2) * (total + 1.0)))


def beta_quantile_grid(alpha: float, beta: float, u_grid: np.ndarray) -> np.ndarray:
    """Return a stable Beta quantile curve evaluated on `u_grid`."""

    quantiles = stats.beta.ppf(u_grid, alpha, beta)
    return np.asarray(quantiles, dtype=float)


def pert_shape_parameters(a: float, mode: float, b: float, pert_lambda: float = 4.0) -> tuple[float, float]:
    """Return Beta shape parameters for a transformed standard Beta-PERT fit."""

    width = b - a
    if not np.isfinite(width) or width <= TOL:
        return np.nan, np.nan
    alpha = 1.0 + pert_lambda * (mode - a) / width
    beta_param = 1.0 + pert_lambda * (b - mode) / width
    if alpha <= 0.0 or beta_param <= 0.0:
        return np.nan, np.nan
    return float(alpha), float(beta_param)


def pert_mean(a: float, mode: float, b: float, pert_lambda: float = 4.0) -> float:
    """Return the mean of the transformed Beta-PERT distribution."""

    alpha, beta_param = pert_shape_parameters(a, mode, b, pert_lambda)
    if not np.isfinite(alpha) or not np.isfinite(beta_param):
        return np.nan
    return float(a + (b - a) * alpha / (alpha + beta_param))


def pert_variance(a: float, mode: float, b: float, pert_lambda: float = 4.0) -> float:
    """Return the variance of the transformed Beta-PERT distribution."""

    alpha, beta_param = pert_shape_parameters(a, mode, b, pert_lambda)
    if not np.isfinite(alpha) or not np.isfinite(beta_param):
        return np.nan
    scale = b - a
    unit_variance = alpha * beta_param / (((alpha + beta_param) ** 2) * (alpha + beta_param + 1.0))
    return float((scale**2) * unit_variance)


def pert_quantile_grid(a: float, mode: float, b: float, pert_lambda: float, u_grid: np.ndarray) -> np.ndarray:
    """Return the PERT quantile curve evaluated on `u_grid`."""

    alpha, beta_param = pert_shape_parameters(a, mode, b, pert_lambda)
    quantiles = stats.beta.ppf(u_grid, alpha, beta_param)
    return np.asarray(a + (b - a) * quantiles, dtype=float)


def fitted_row_mean_variance_quantiles(row: pd.Series, u_grid: np.ndarray) -> tuple[str, float, float, np.ndarray]:
    """Compute mean, variance, and quantile grid for one fitted expert row."""

    family = infer_distribution_family(row)
    if family == "beta":
        alpha = float(row["alpha"])
        beta_param = float(row["beta"])
        return (
            family,
            beta_mean(alpha, beta_param),
            beta_variance(alpha, beta_param),
            beta_quantile_grid(alpha, beta_param, u_grid),
        )

    if family == "pert":
        a = float(row["a"])
        b = float(row["b"])
        mode = float(row["mode"])
        pert_lambda = 4.0 if pd.isna(row.get("pert_lambda")) else float(row["pert_lambda"])
        return (
            family,
            pert_mean(a, mode, b, pert_lambda),
            pert_variance(a, mode, b, pert_lambda),
            pert_quantile_grid(a, mode, b, pert_lambda, u_grid),
        )

    raise ValueError("Row does not contain a supported fitted family.")


def compute_pairwise_wasserstein_metrics(quantile_matrix: np.ndarray, u_grid: np.ndarray) -> dict:
    """Compute pairwise W1/W2 disagreement summaries from quantile curves."""

    n_experts = quantile_matrix.shape[0]
    pairwise_rows: list[dict] = []
    w1_values: list[float] = []
    w2_values: list[float] = []
    w2_sq_values: list[float] = []

    if n_experts < 2:
        return {
            "mean_pairwise_w1": np.nan,
            "mean_pairwise_w1_sq": np.nan,
            "mean_pairwise_w2": np.nan,
            "mean_pairwise_w2_sq": np.nan,
            "max_pairwise_w1": np.nan,
            "max_pairwise_w2": np.nan,
            "pairwise_rows": pairwise_rows,
        }

    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            diff = quantile_matrix[i] - quantile_matrix[j]
            w1 = float(np.trapezoid(np.abs(diff), u_grid))
            w2_sq = float(np.trapezoid(diff**2, u_grid))
            w2 = float(np.sqrt(max(w2_sq, 0.0)))

            w1_values.append(w1)
            w2_values.append(w2)
            w2_sq_values.append(w2_sq)
            pairwise_rows.append(
                {
                    "expert_i_index": i,
                    "expert_j_index": j,
                    "w1": w1,
                    "w1_sq": float(w1**2),
                    "w2": w2,
                    "w2_sq": w2_sq,
                }
            )

    return {
        "mean_pairwise_w1": float(np.mean(w1_values)),
        "mean_pairwise_w1_sq": float(np.mean(np.square(w1_values))),
        "mean_pairwise_w2": float(np.mean(w2_values)),
        "mean_pairwise_w2_sq": float(np.mean(w2_sq_values)),
        "max_pairwise_w1": float(np.max(w1_values)),
        "max_pairwise_w2": float(np.max(w2_values)),
        "pairwise_rows": pairwise_rows,
    }


def compute_w2_barycenter_quantiles(quantile_matrix: np.ndarray) -> np.ndarray:
    """Compute the 1D W2 barycenter quantile curve via pointwise mean."""

    return np.mean(quantile_matrix, axis=0)


def compute_w1_barycenter_quantiles(quantile_matrix: np.ndarray) -> np.ndarray:
    """Compute the practical 1D W1 barycenter proxy via pointwise median."""

    return np.median(quantile_matrix, axis=0)


def compute_cpd_entry_metrics(group_df: pd.DataFrame, u_grid: np.ndarray) -> dict:
    """Compute one cpd_entry row of classical and Wasserstein metrics."""

    if group_df.empty:
        raise ValueError("`group_df` must contain at least one expert row.")

    first_row = group_df.iloc[0]
    result = {column: first_row.get(column) for column in CPD_ENTRY_IDENTITY_COLUMNS}

    experts_used = group_df["expert"].astype(str).tolist()
    methods_used = (
        sorted(group_df["best_method"].dropna().astype(str).unique().tolist())
        if "best_method" in group_df.columns
        else []
    )

    result.update(
        {
            "n_experts_used": int(len(group_df)),
            "experts_used": json.dumps(experts_used),
            "best_methods_used": json.dumps(methods_used),
        }
    )

    warning_flags: list[str] = []
    if len(group_df) < 5:
        warning_flags.append("fewer_than_5_experts")

    expert_families: list[str] = []
    expert_means: list[float] = []
    expert_variances: list[float] = []
    quantile_curves: list[np.ndarray] = []

    for _, row in group_df.iterrows():
        family, mean_value, variance_value, quantile_curve = fitted_row_mean_variance_quantiles(row, u_grid)
        expert_families.append(family)
        expert_means.append(mean_value)
        expert_variances.append(variance_value)
        quantile_curves.append(quantile_curve)

    if len(set(expert_families)) > 1:
        warning_flags.append("mixed_distribution_families")

    result["distribution_families_used"] = json.dumps(sorted(set(expert_families)))

    expert_means = np.asarray(expert_means, dtype=float)
    expert_variances = np.asarray(expert_variances, dtype=float)

    sum_within_expert_variances = float(np.sum(expert_variances))
    mean_within_expert_variance = float(np.mean(expert_variances))
    mixture_mean = float(np.mean(expert_means))
    between_expert_variance_of_means = float(np.mean((expert_means - mixture_mean) ** 2))
    mixture_second_moment = float(np.mean(expert_variances + expert_means**2))
    mixture_variance = float(mixture_second_moment - mixture_mean**2)
    mixture_variance_ltv = float(mean_within_expert_variance + between_expert_variance_of_means)
    mixture_variance_consistency_gap = float(mixture_variance - mixture_variance_ltv)

    quantile_matrix = np.vstack(quantile_curves)

    pairwise_metrics = compute_pairwise_wasserstein_metrics(quantile_matrix, u_grid)

    w2_barycenter = compute_w2_barycenter_quantiles(quantile_matrix)
    w1_barycenter = compute_w1_barycenter_quantiles(quantile_matrix)

    w2_sq_to_bary = np.array(
        [
            np.trapezoid((quantile_curve - w2_barycenter) ** 2, u_grid)
            for quantile_curve in quantile_matrix
        ],
        dtype=float,
    )
    w2_to_bary = np.sqrt(np.maximum(w2_sq_to_bary, 0.0))

    w1_to_bary = np.array(
        [
            np.trapezoid(np.abs(quantile_curve - w1_barycenter), u_grid)
            for quantile_curve in quantile_matrix
        ],
        dtype=float,
    )

    frechet_dispersion_w1 = float(np.mean(w1_to_bary))
    frechet_variance_like_w1 = float(np.mean(np.square(w1_to_bary)))
    frechet_dispersion_w2 = float(np.mean(w2_to_bary))
    frechet_variance_w2 = float(np.mean(w2_sq_to_bary))

    if len(group_df) >= 2:
        max_pairwise_w2 = pairwise_metrics["max_pairwise_w2"]
        mean_pairwise_w1 = pairwise_metrics["mean_pairwise_w1"]
        mean_pairwise_w2 = pairwise_metrics["mean_pairwise_w2"]
        if np.isfinite(max_pairwise_w2) and max_pairwise_w2 <= NEARLY_IDENTICAL_W2_THRESHOLD:
            warning_flags.append("experts_nearly_identical")
        if (
            np.isfinite(mean_pairwise_w1) and mean_pairwise_w1 >= LARGE_DISAGREEMENT_W1_THRESHOLD
        ) or (
            np.isfinite(mean_pairwise_w2) and mean_pairwise_w2 >= LARGE_DISAGREEMENT_W2_THRESHOLD
        ):
            warning_flags.append("large_wasserstein_disagreement")

    result.update(
        {
            "sum_within_expert_variances": sum_within_expert_variances,
            "mean_within_expert_variance": mean_within_expert_variance,
            "mixture_mean": mixture_mean,
            "between_expert_variance_of_means": between_expert_variance_of_means,
            "mixture_second_moment": mixture_second_moment,
            "mixture_variance": mixture_variance,
            "mixture_variance_law_total_variance": mixture_variance_ltv,
            "mixture_variance_consistency_gap": mixture_variance_consistency_gap,
            "mean_pairwise_w1": pairwise_metrics["mean_pairwise_w1"],
            "mean_pairwise_w1_sq": pairwise_metrics["mean_pairwise_w1_sq"],
            "mean_pairwise_w2": pairwise_metrics["mean_pairwise_w2"],
            "mean_pairwise_w2_sq": pairwise_metrics["mean_pairwise_w2_sq"],
            "max_pairwise_w1": pairwise_metrics["max_pairwise_w1"],
            "max_pairwise_w2": pairwise_metrics["max_pairwise_w2"],
            "frechet_dispersion_w1": frechet_dispersion_w1,
            "frechet_variance_like_w1": frechet_variance_like_w1,
            "frechet_dispersion_w2": frechet_dispersion_w2,
            "frechet_variance_w2": frechet_variance_w2,
            "min_expert_mean": float(np.min(expert_means)),
            "max_expert_mean": float(np.max(expert_means)),
            "std_expert_means": float(np.std(expert_means)),
            "min_expert_variance": float(np.min(expert_variances)),
            "max_expert_variance": float(np.max(expert_variances)),
            "warning_flags": "|".join(warning_flags),
            "_pairwise_rows": pairwise_metrics["pairwise_rows"],
            "_experts_used_list": experts_used,
            "_w1_barycenter_quantiles": w1_barycenter,
            "_w2_barycenter_quantiles": w2_barycenter,
            "_quantile_matrix": quantile_matrix,
        }
    )

    return result


def plot_cpd_entry_quantiles(
    cpd_entry_metrics: dict,
    u_grid: np.ndarray,
    output_path: Path,
) -> None:
    """Plot expert quantile curves plus W1/W2 barycenter curves for one cpd_entry."""

    try:
        mpl_config_dir = output_path.parent / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib.pyplot as plt
    except Exception:
        return

    quantile_matrix = cpd_entry_metrics["_quantile_matrix"]
    experts = cpd_entry_metrics["_experts_used_list"]
    w1_barycenter = cpd_entry_metrics["_w1_barycenter_quantiles"]
    w2_barycenter = cpd_entry_metrics["_w2_barycenter_quantiles"]

    plt.figure(figsize=(10, 6))
    for expert, quantile_curve in zip(experts, quantile_matrix):
        plt.plot(u_grid, quantile_curve, alpha=0.5, linewidth=1.2, label=str(expert))
    plt.plot(u_grid, w2_barycenter, color="black", linewidth=2.5, label="W2 barycenter")
    plt.plot(u_grid, w1_barycenter, color="darkred", linewidth=2.0, linestyle="--", label="W1 barycenter")
    plt.xlabel("u")
    plt.ylabel("Quantile")
    plt.title(
        f"{cpd_entry_metrics['model_name']} | {cpd_entry_metrics['node_name']} | {cpd_entry_metrics['benchmark']}"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def coerce_truthy(value) -> bool:
    """Interpret booleans written as bools, numbers, or strings."""

    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"true", "1", "yes", "y", "t"}


def flatten_pairwise_rows(cpd_entry_metrics: dict) -> list[dict]:
    """Expand cpd_entry-level pairwise Wasserstein metrics to long-form rows."""

    pairwise_rows = []
    experts = cpd_entry_metrics["_experts_used_list"]
    families = json.loads(cpd_entry_metrics["distribution_families_used"]) if cpd_entry_metrics.get("distribution_families_used") else []
    methods = json.loads(cpd_entry_metrics["best_methods_used"])

    for pairwise_row in cpd_entry_metrics["_pairwise_rows"]:
        row = {column: cpd_entry_metrics[column] for column in CPD_ENTRY_IDENTITY_COLUMNS}
        i = pairwise_row["expert_i_index"]
        j = pairwise_row["expert_j_index"]
        row.update(
            {
                "expert_i": experts[i],
                "expert_j": experts[j],
                "w1": pairwise_row["w1"],
                "w1_sq": pairwise_row["w1_sq"],
                "w2": pairwise_row["w2"],
                "w2_sq": pairwise_row["w2_sq"],
            }
        )
        if families:
            row["distribution_families_used"] = cpd_entry_metrics["distribution_families_used"]
        if methods:
            row["best_methods_used"] = cpd_entry_metrics["best_methods_used"]
        pairwise_rows.append(row)

    return pairwise_rows


def build_rankings(cpd_entry_df: pd.DataFrame) -> pd.DataFrame:
    """Rank cpd_entry rows by several main uncertainty/disagreement metrics."""

    ranking_df = cpd_entry_df.copy()
    rank_specs = {
        "rank_by_mixture_variance": "mixture_variance",
        "rank_by_frechet_variance_w2": "frechet_variance_w2",
        "rank_by_mean_pairwise_w1": "mean_pairwise_w1",
        "rank_by_mean_pairwise_w2": "mean_pairwise_w2",
    }

    for rank_column, metric_column in rank_specs.items():
        ranking_df[rank_column] = ranking_df[metric_column].rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )

    sort_columns = list(rank_specs.keys())
    return ranking_df.sort_values(sort_columns, kind="stable")


def summarize_results(
    cpd_entry_df: pd.DataFrame,
    excluded_rows_df: pd.DataFrame,
) -> dict:
    """Build dataset-level summary statistics for the cpd_entry outputs."""

    return {
        "total_cpd_entries": int(len(cpd_entry_df)),
        "average_number_of_experts_per_cpd_entry": float(cpd_entry_df["n_experts_used"].mean())
        if not cpd_entry_df.empty
        else 0.0,
        "number_of_cpd_entries_with_fewer_than_5_experts": int((cpd_entry_df["n_experts_used"] < 5).sum())
        if not cpd_entry_df.empty
        else 0,
        "average_mixture_variance": float(cpd_entry_df["mixture_variance"].mean())
        if not cpd_entry_df.empty
        else np.nan,
        "average_frechet_variance_w2": float(cpd_entry_df["frechet_variance_w2"].mean())
        if not cpd_entry_df.empty
        else np.nan,
        "average_mean_pairwise_w1": float(cpd_entry_df["mean_pairwise_w1"].mean())
        if not cpd_entry_df.empty
        else np.nan,
        "average_mean_pairwise_w2": float(cpd_entry_df["mean_pairwise_w2"].mean())
        if not cpd_entry_df.empty
        else np.nan,
        "excluded_rows": int(len(excluded_rows_df)),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Path to `best_fits.csv`.")
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    parser.add_argument(
        "--n-grid",
        type=int,
        default=GRID_SIZE,
        help=f"Number of quantile grid points to use. Default: {GRID_SIZE}.",
    )
    parser.add_argument(
        "--plot-cpd-entry-index",
        type=int,
        default=None,
        help="Optional integer index into the cpd_entry output table to plot.",
    )
    return parser.parse_args(argv)


def setup_logger(output_dir: Path) -> logging.Logger:
    """Configure a simple file and console logger."""

    logger = logging.getLogger("compute_cpd_entry_uncertainty")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(levelname)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "processing.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main(input_csv_path: str | Path, output_dir: str | Path) -> None:
    """Run the cpd_entry aggregation pipeline."""

    input_path = Path(input_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_path)

    u_grid = np.linspace(GRID_EPS, 1.0 - GRID_EPS, GRID_SIZE)
    raw_df = load_best_fits(input_path)

    valid_rows: list[pd.Series] = []
    excluded_rows: list[dict] = []
    for _, row in raw_df.iterrows():
        validation = validate_fitted_row(row)
        if validation["valid"]:
            valid_rows.append(row)
        else:
            excluded = row.to_dict()
            excluded["validation_errors"] = " | ".join(validation["errors"])
            excluded["validation_warnings"] = " | ".join(validation["warnings"])
            excluded_rows.append(excluded)

    valid_df = pd.DataFrame(valid_rows)
    excluded_rows_df = pd.DataFrame(excluded_rows)

    if not excluded_rows_df.empty:
        excluded_rows_df.to_csv(output_path / "excluded_fitted_rows.csv", index=False)
        logger.info("Excluded %s rows before cpd_entry aggregation.", len(excluded_rows_df))
    else:
        logger.info("Excluded 0 rows before cpd_entry aggregation.")

    cpd_entry_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    cpd_entry_warning_rows: list[dict] = []

    if valid_df.empty:
        cpd_entry_df = pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)
        pairwise_df = pd.DataFrame(
            columns=CPD_ENTRY_IDENTITY_COLUMNS + ["expert_i", "expert_j", "w1", "w1_sq", "w2", "w2_sq"]
        )
    else:
        grouped = valid_df.groupby(CPD_ENTRY_IDENTITY_COLUMNS, dropna=False, sort=True)
        logger.info("Processing %s cpd_entry groups.", len(grouped))

        for cpd_entry_index, (_, group_df) in enumerate(grouped):
            try:
                metrics = compute_cpd_entry_metrics(group_df.reset_index(drop=True), u_grid)
                cpd_entry_rows.append({column: metrics.get(column, np.nan) for column in DEFAULT_OUTPUT_COLUMNS})
                pairwise_rows.extend(flatten_pairwise_rows(metrics))

                if metrics["warning_flags"]:
                    cpd_entry_warning_rows.append(
                        {
                            **{column: metrics[column] for column in CPD_ENTRY_IDENTITY_COLUMNS},
                            "warning_flags": metrics["warning_flags"],
                            "n_experts_used": metrics["n_experts_used"],
                        }
                    )
                    logger.warning(
                        "cpd_entry %s has warnings: %s",
                        cpd_entry_index,
                        metrics["warning_flags"],
                    )
            except Exception as exc:
                warning_row = {column: group_df.iloc[0].get(column) for column in CPD_ENTRY_IDENTITY_COLUMNS}
                warning_row.update(
                    {
                        "warning_flags": f"cpd_entry_processing_error:{exc}",
                        "n_experts_used": int(len(group_df)),
                    }
                )
                cpd_entry_warning_rows.append(warning_row)
                logger.warning("Failed to process cpd_entry %s: %s", cpd_entry_index, exc)

        cpd_entry_df = pd.DataFrame(cpd_entry_rows).reindex(columns=DEFAULT_OUTPUT_COLUMNS)
        pairwise_df = pd.DataFrame(pairwise_rows)

    cpd_entry_path = output_path / "cpd_entry_uncertainty.csv"
    pairwise_path = output_path / "pairwise_wasserstein_long.csv"
    summary_path = output_path / "cpd_entry_summary.json"
    rankings_path = output_path / "cpd_entry_rankings.csv"
    warnings_path = output_path / "cpd_entry_warnings.csv"

    cpd_entry_df.to_csv(cpd_entry_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)

    cpd_entry_warnings_df = pd.DataFrame(cpd_entry_warning_rows)
    if not cpd_entry_warnings_df.empty:
        cpd_entry_warnings_df.to_csv(warnings_path, index=False)

    summary = summarize_results(cpd_entry_df, excluded_rows_df)
    summary_path.write_text(json.dumps(summary, indent=2))

    rankings_df = build_rankings(cpd_entry_df) if not cpd_entry_df.empty else cpd_entry_df.copy()
    rankings_df.to_csv(rankings_path, index=False)

    logger.info("Wrote %s", cpd_entry_path)
    logger.info("Wrote %s", pairwise_path)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", rankings_path)

    if not cpd_entry_df.empty:
        logger.info("Average experts per cpd_entry: %.3f", cpd_entry_df["n_experts_used"].mean())
        logger.info("Average mixture variance: %.6f", cpd_entry_df["mixture_variance"].mean())
        logger.info("Average Frechet variance W2: %.6f", cpd_entry_df["frechet_variance_w2"].mean())
        logger.info("Average mean pairwise W1: %.6f", cpd_entry_df["mean_pairwise_w1"].mean())
        logger.info("Average mean pairwise W2: %.6f", cpd_entry_df["mean_pairwise_w2"].mean())


def run_cli(argv: Iterable[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)

    global GRID_SIZE  # Keep the requested API simple for `main`.
    GRID_SIZE = int(args.n_grid)

    main(args.input_csv, args.output_dir)

    if args.plot_cpd_entry_index is not None:
        output_path = Path(args.output_dir)
        cpd_entry_df = pd.read_csv(output_path / "cpd_entry_uncertainty.csv")
        if 0 <= args.plot_cpd_entry_index < len(cpd_entry_df):
            raw_df = load_best_fits(args.input_csv)
            valid_rows = [row for _, row in raw_df.iterrows() if validate_fitted_row(row)["valid"]]
            valid_df = pd.DataFrame(valid_rows)
            u_grid = np.linspace(GRID_EPS, 1.0 - GRID_EPS, GRID_SIZE)
            grouped = list(valid_df.groupby(CPD_ENTRY_IDENTITY_COLUMNS, dropna=False, sort=True))
            _, group_df = grouped[args.plot_cpd_entry_index]
            metrics = compute_cpd_entry_metrics(group_df.reset_index(drop=True), u_grid)
            plot_name = f"cpd_entry_quantiles_{args.plot_cpd_entry_index:03d}.png"
            plot_cpd_entry_quantiles(metrics, u_grid, output_path / plot_name)


if __name__ == "__main__":
    run_cli()
