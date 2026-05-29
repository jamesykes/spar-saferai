"""Fit row-level Beta distributions to elicited quartiles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import beta as beta_dist

from saferai_budget_recovery import config


PROBABILITIES = np.array([0.25, 0.50, 0.75], dtype=float)
FITTED_Q_COLUMNS = ("fitted_q25", "fitted_q50", "fitted_q75")
RESIDUAL_COLUMNS = ("residual_q25", "residual_q50", "residual_q75")


def fit_beta_to_quartiles(q25: float, q50: float, q75: float, eps: float = 1e-6) -> dict[str, Any]:
    """Fit a Beta distribution to 25th/50th/75th percentiles by least squares."""

    raw_quartiles = np.array([q25, q50, q75], dtype=float)
    result = _empty_fit_result()

    if not np.all(np.isfinite(raw_quartiles)):
        result.update(
            {
                "optimizer_success": False,
                "optimizer_message": "Input quartiles must be finite.",
                "fit_quality_flag": "fail",
            }
        )
        return result
    if np.any(raw_quartiles < 0) or np.any(raw_quartiles > 1):
        result.update(
            {
                "optimizer_success": False,
                "optimizer_message": "Input quartiles must lie in [0, 1].",
                "fit_quality_flag": "fail",
            }
        )
        return result
    if raw_quartiles[0] > raw_quartiles[1] or raw_quartiles[1] > raw_quartiles[2]:
        result.update(
            {
                "optimizer_success": False,
                "optimizer_message": "Input quartiles must be ordered q25 <= q50 <= q75.",
                "fit_quality_flag": "fail",
            }
        )
        return result

    clipped_quartiles = np.clip(raw_quartiles, eps, 1.0 - eps)
    quartiles_were_clipped = bool(np.any(clipped_quartiles != raw_quartiles))
    initial_log_params = np.log(_initial_alpha_beta(clipped_quartiles))

    def residuals(log_params: np.ndarray) -> np.ndarray:
        alpha, beta = np.exp(log_params)
        fitted = beta_dist.ppf(PROBABILITIES, alpha, beta)
        if not np.all(np.isfinite(fitted)):
            return np.full_like(PROBABILITIES, 1e6)
        return fitted - clipped_quartiles

    try:
        opt = least_squares(
            residuals,
            initial_log_params,
            bounds=(np.log([1e-6, 1e-6]), np.log([1e6, 1e6])),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=1000,
        )
        alpha, beta = np.exp(opt.x)
        fitted_quartiles = beta_dist.ppf(PROBABILITIES, alpha, beta)
        residual_values = fitted_quartiles - clipped_quartiles
        sse = float(np.sum(residual_values**2))
        rmse = float(np.sqrt(sse / len(PROBABILITIES)))
        parameters_finite = bool(np.isfinite(alpha) and np.isfinite(beta) and alpha > 0 and beta > 0)
        success = bool(opt.success and parameters_finite and np.all(np.isfinite(fitted_quartiles)))
        fit_quality_flag = _fit_quality_flag(success, rmse)
        result.update(
            {
                "alpha": float(alpha) if success else np.nan,
                "beta": float(beta) if success else np.nan,
                "fit_sse": sse if np.isfinite(sse) else np.nan,
                "fit_rmse": rmse if np.isfinite(rmse) else np.nan,
                "fitted_q25": float(fitted_quartiles[0]) if success else np.nan,
                "fitted_q50": float(fitted_quartiles[1]) if success else np.nan,
                "fitted_q75": float(fitted_quartiles[2]) if success else np.nan,
                "residual_q25": float(residual_values[0]) if success else np.nan,
                "residual_q50": float(residual_values[1]) if success else np.nan,
                "residual_q75": float(residual_values[2]) if success else np.nan,
                "optimizer_success": success,
                "optimizer_message": str(opt.message),
                "quartiles_were_clipped": quartiles_were_clipped,
                "fit_quality_flag": fit_quality_flag,
            }
        )
    except Exception as exc:
        result.update(
            {
                "optimizer_success": False,
                "optimizer_message": f"Optimizer raised {type(exc).__name__}: {exc}",
                "quartiles_were_clipped": quartiles_were_clipped,
                "fit_quality_flag": "fail",
            }
        )
    return result


def fit_beta_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Fit one Beta distribution per input row and preserve row count/order."""

    missing = [col for col in config.QUARTILE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot fit Beta distributions; missing quartile columns: {missing}")

    out = df.copy()
    fit_rows = [
        fit_beta_to_quartiles(
            row[config.QUARTILE_COLUMNS[0]],
            row[config.QUARTILE_COLUMNS[1]],
            row[config.QUARTILE_COLUMNS[2]],
        )
        for row in out.to_dict(orient="records")
    ]
    fit_df = pd.DataFrame(fit_rows, index=out.index)
    return pd.concat([out, fit_df], axis=1)


def _initial_alpha_beta(quartiles: np.ndarray) -> np.ndarray:
    median = float(np.clip(quartiles[1], 1e-4, 1.0 - 1e-4))
    iqr = max(float(quartiles[2] - quartiles[0]), 1e-4)
    sigma = max(iqr / 1.349, 1e-4)
    variance = sigma**2
    concentration = median * (1.0 - median) / variance - 1.0
    if not np.isfinite(concentration) or concentration <= 0:
        return np.array([2.0, 2.0], dtype=float)
    alpha = median * concentration
    beta = (1.0 - median) * concentration
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
        return np.array([2.0, 2.0], dtype=float)
    return np.array([max(alpha, 1e-3), max(beta, 1e-3)], dtype=float)


def _fit_quality_flag(success: bool, rmse: float) -> str:
    if not success or not np.isfinite(rmse):
        return "fail"
    if rmse <= 0.02:
        return "ok"
    return "warn"


def _empty_fit_result() -> dict[str, Any]:
    return {
        "alpha": np.nan,
        "beta": np.nan,
        "fit_sse": np.nan,
        "fit_rmse": np.nan,
        "fitted_q25": np.nan,
        "fitted_q50": np.nan,
        "fitted_q75": np.nan,
        "residual_q25": np.nan,
        "residual_q50": np.nan,
        "residual_q75": np.nan,
        "optimizer_success": False,
        "optimizer_message": "",
        "quartiles_were_clipped": False,
        "fit_quality_flag": "fail",
    }

