"""Exchangeable nodewise mixtures for fitted SOTA elicitation rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from saferai_budget_recovery import config


USABLE_FIT_FLAGS = {"ok", "warn"}


def build_nodewise_mixtures(fit_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Collect usable fitted Beta rows by MITRE-step label."""

    required = {"step_name", "alpha", "beta", "fit_quality_flag"}
    missing = required - set(fit_df.columns)
    if missing:
        raise ValueError(f"Cannot build nodewise mixtures; missing columns: {sorted(missing)}")

    usable = fit_df.loc[
        fit_df["fit_quality_flag"].isin(USABLE_FIT_FLAGS)
        & np.isfinite(fit_df["alpha"])
        & np.isfinite(fit_df["beta"])
        & (fit_df["alpha"] > 0)
        & (fit_df["beta"] > 0)
    ].copy()

    mixtures: dict[str, pd.DataFrame] = {}
    for step in config.EXPECTED_MITRE_STEP_LABELS:
        step_rows = usable.loc[usable["step_name"].eq(step)].copy()
        if step_rows.empty:
            raise ValueError(f"No usable fitted Beta distributions for MITRE-step label: {step}")
        mixtures[step] = step_rows.reset_index(drop=True)
    return mixtures


def sample_nodewise_mixture(
    mixture_rows: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from one exchangeable nodewise mixture."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    required = {"alpha", "beta"}
    missing = required - set(mixture_rows.columns)
    if missing:
        raise ValueError(f"Cannot sample mixture; missing columns: {sorted(missing)}")
    if mixture_rows.empty:
        raise ValueError("Cannot sample an empty nodewise mixture.")

    alpha = mixture_rows["alpha"].to_numpy(dtype=float)
    beta = mixture_rows["beta"].to_numpy(dtype=float)
    if np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(beta)) or np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("Mixture rows must have finite positive alpha and beta parameters.")

    component_indices = rng.integers(0, len(mixture_rows), size=n_samples)
    uniforms = rng.uniform(1e-12, 1.0 - 1e-12, size=n_samples)
    samples = beta_dist.ppf(uniforms, alpha[component_indices], beta[component_indices])
    if np.any(~np.isfinite(samples)):
        raise ValueError("Mixture sampling produced non-finite values.")
    return samples

