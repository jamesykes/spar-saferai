from __future__ import annotations

import numpy as np
import pandas as pd

from saferai_budget_recovery import config
from saferai_budget_recovery.mixtures import build_nodewise_mixtures, sample_nodewise_mixture
from saferai_budget_recovery.sampling import sample_full_reference_p_success


def _synthetic_fit_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step_name": step,
                "alpha": 2.0 + i,
                "beta": 5.0 + i,
                "fit_quality_flag": "ok",
                "draw_uid": f"draw-{i}",
            }
            for i, step in enumerate(config.EXPECTED_MITRE_STEP_LABELS)
        ]
    )


def test_build_mixtures_returns_all_expected_steps() -> None:
    mixtures = build_nodewise_mixtures(_synthetic_fit_df())
    assert set(mixtures) == set(config.EXPECTED_MITRE_STEP_LABELS)
    assert all(len(rows) == 1 for rows in mixtures.values())


def test_sampling_returns_values_in_unit_interval() -> None:
    mixture = _synthetic_fit_df().head(1)
    samples = sample_nodewise_mixture(mixture, n_samples=1000, rng=np.random.default_rng(123))
    assert samples.shape == (1000,)
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_sampling_is_reproducible_with_fixed_seed() -> None:
    mixture = _synthetic_fit_df().head(2)
    first = sample_nodewise_mixture(mixture, n_samples=50, rng=np.random.default_rng(123))
    second = sample_nodewise_mixture(mixture, n_samples=50, rng=np.random.default_rng(123))
    assert np.allclose(first, second)


def test_full_reference_sampling_returns_p_success_in_unit_interval() -> None:
    sample_df = sample_full_reference_p_success(_synthetic_fit_df(), n_samples=250, seed=12345)
    assert "p_success" in sample_df.columns
    assert len(sample_df) == 250
    assert np.all((sample_df["p_success"] >= 0.0) & (sample_df["p_success"] <= 1.0))

