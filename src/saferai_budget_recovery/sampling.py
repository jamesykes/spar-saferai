"""Sampling helpers for the full-data exchangeable reference model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from saferai_budget_recovery.forward_model import (
    STEP_LABEL_BY_VARIABLE,
    oc3_dos_p_success,
    oc3_dos_tactic_probabilities,
)
from saferai_budget_recovery.mixtures import build_nodewise_mixtures, sample_nodewise_mixture


def sample_full_reference_p_success(
    fit_df: pd.DataFrame,
    n_samples: int,
    seed: int = 12345,
) -> pd.DataFrame:
    """Sample the full-data exchangeable nodewise reference distribution."""

    return sample_p_success_from_revealed(fit_df, n_samples=n_samples, seed=seed)


def sample_p_success_from_revealed(
    revealed_fit_df: pd.DataFrame,
    n_samples: int,
    seed: int = 12345,
) -> pd.DataFrame:
    """Sample OC3 DoS P(success) from currently revealed fitted rows."""

    rng = np.random.default_rng(seed)
    mixtures = build_nodewise_mixtures(revealed_fit_df)
    leaf_samples = {
        variable: sample_nodewise_mixture(mixtures[step_label], n_samples, rng)
        for variable, step_label in STEP_LABEL_BY_VARIABLE.items()
    }
    tactics = oc3_dos_tactic_probabilities(**leaf_samples)
    p_success = oc3_dos_p_success(**leaf_samples)

    out = pd.DataFrame(leaf_samples)
    for name, values in tactics.items():
        out[name] = values
    out["p_success"] = p_success
    return out
