from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from saferai_budget_recovery.beta_fit import fit_beta_distributions, fit_beta_to_quartiles


def test_fit_beta_to_known_quartiles_recovers_quartiles() -> None:
    q25, q50, q75 = beta_dist.ppf([0.25, 0.50, 0.75], 3.0, 7.0)
    fit = fit_beta_to_quartiles(q25, q50, q75)
    assert fit["fit_quality_flag"] == "ok"
    assert fit["fit_rmse"] < 1e-4
    assert np.allclose(
        [fit["fitted_q25"], fit["fitted_q50"], fit["fitted_q75"]],
        [q25, q50, q75],
        atol=1e-4,
    )


def test_exact_zero_one_quartiles_trigger_clipping_and_do_not_crash() -> None:
    fit = fit_beta_to_quartiles(0.0, 0.5, 1.0)
    assert fit["quartiles_were_clipped"] is True
    assert fit["fit_quality_flag"] in {"ok", "warn"}
    assert np.isfinite(fit["alpha"])
    assert np.isfinite(fit["beta"])


def test_invalid_quartiles_fail_gracefully() -> None:
    fit = fit_beta_to_quartiles(0.8, 0.5, 0.9)
    assert fit["fit_quality_flag"] == "fail"
    assert np.isnan(fit["alpha"])
    assert "ordered" in fit["optimizer_message"]


def test_fit_beta_distributions_preserves_row_count() -> None:
    df = pd.DataFrame(
        [
            {"percentile_25th": 0.2, "percentile_50th": 0.4, "percentile_75th": 0.6},
            {"percentile_25th": 0.1, "percentile_50th": 0.2, "percentile_75th": 0.3},
        ]
    )
    fitted = fit_beta_distributions(df)
    assert len(fitted) == len(df)
    assert {"alpha", "beta", "fit_rmse", "fit_quality_flag"}.issubset(fitted.columns)

