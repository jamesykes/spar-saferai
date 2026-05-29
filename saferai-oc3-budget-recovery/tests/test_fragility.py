from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saferai_budget_recovery import config
from saferai_budget_recovery.fragility import (
    compute_current_output_quantiles,
    compute_loo_fragility_scores,
    loo_perturbed_revealed_df,
)
from saferai_budget_recovery.reveal import FITTED_ROW_UID_COLUMN, usable_fit_rows
from saferai_budget_recovery.distances import quantile_grid


def _synthetic_revealed(rows_per_step: int = 2) -> pd.DataFrame:
    rows = []
    for step_i, step in enumerate(config.EXPECTED_MITRE_STEP_LABELS):
        for row_i in range(rows_per_step):
            rows.append(
                {
                    "step_name": step,
                    "model": config.EXPECTED_LLM_MODELS[row_i % len(config.EXPECTED_LLM_MODELS)],
                    "run_id": f"run-{row_i}",
                    "repeat_index": 1,
                    "draw_uid": f"model-{row_i}__run-{row_i}__1",
                    "alpha": 2.0 + step_i * 0.1 + row_i * 0.1,
                    "beta": 5.0 + step_i * 0.1 + row_i * 0.1,
                    "fit_quality_flag": "ok",
                }
            )
    return usable_fit_rows(pd.DataFrame(rows))


def test_loo_perturbed_revealed_df_removes_exactly_one_intended_row() -> None:
    revealed = _synthetic_revealed(rows_per_step=2)
    target = revealed.iloc[0]
    perturbed = loo_perturbed_revealed_df(
        revealed,
        step_label=target["step_name"],
        row_id=target[FITTED_ROW_UID_COLUMN],
    )
    assert len(perturbed) == len(revealed) - 1
    assert target[FITTED_ROW_UID_COLUMN] not in set(perturbed[FITTED_ROW_UID_COLUMN])


def test_loo_perturbation_preserves_other_steps() -> None:
    revealed = _synthetic_revealed(rows_per_step=2)
    target = revealed.iloc[0]
    before_counts = revealed.groupby("step_name").size()
    perturbed = loo_perturbed_revealed_df(
        revealed,
        step_label=target["step_name"],
        row_id=target[FITTED_ROW_UID_COLUMN],
    )
    after_counts = perturbed.groupby("step_name").size()
    for step in config.EXPECTED_MITRE_STEP_LABELS:
        expected = before_counts[step] - 1 if step == target["step_name"] else before_counts[step]
        assert after_counts[step] == expected


def test_node_with_fewer_than_two_rows_returns_nan_fragility() -> None:
    revealed = _synthetic_revealed(rows_per_step=2)
    step = config.EXPECTED_MITRE_STEP_LABELS[0]
    one_row_for_step = revealed.loc[revealed["step_name"].eq(step)].head(1)
    other_steps = revealed.loc[~revealed["step_name"].eq(step)]
    revealed = pd.concat([one_row_for_step, other_steps], ignore_index=True)
    scores = compute_loo_fragility_scores(revealed, n_samples=50, n_grid=21, seed=123)
    row = scores.loc[scores["step_name"].eq(step)].iloc[0]
    assert np.isnan(row["loo_fragility"])
    assert row["n_loo_terms"] == 0


def test_compute_loo_fragility_scores_returns_one_row_per_expected_step() -> None:
    scores = compute_loo_fragility_scores(_synthetic_revealed(rows_per_step=2), n_samples=50, n_grid=21, seed=123)
    assert set(scores["step_name"]) == set(config.EXPECTED_MITRE_STEP_LABELS)
    assert len(scores) == len(config.EXPECTED_MITRE_STEP_LABELS)


def test_fragility_values_are_non_negative_when_finite() -> None:
    scores = compute_loo_fragility_scores(_synthetic_revealed(rows_per_step=2), n_samples=50, n_grid=21, seed=123)
    finite = scores["loo_fragility"].dropna()
    assert (finite >= 0).all()


def test_compute_loo_fragility_scores_is_reproducible_with_fixed_seed() -> None:
    revealed = _synthetic_revealed(rows_per_step=2)
    first = compute_loo_fragility_scores(revealed, n_samples=50, n_grid=21, seed=123)
    second = compute_loo_fragility_scores(revealed, n_samples=50, n_grid=21, seed=123)
    assert np.allclose(first["loo_fragility"], second["loo_fragility"])


def test_missing_required_step_fails_clearly() -> None:
    revealed = _synthetic_revealed(rows_per_step=2)
    missing_step = config.EXPECTED_MITRE_STEP_LABELS[0]
    incomplete = revealed.loc[~revealed["step_name"].eq(missing_step)]
    with pytest.raises(ValueError, match="No usable fitted Beta distributions"):
        compute_current_output_quantiles(incomplete, n_samples=10, quantile_grid=quantile_grid(11), seed=123)


def test_stable_row_identifier_does_not_rely_on_repeat_index_alone() -> None:
    revealed = _synthetic_revealed(rows_per_step=2)
    assert revealed["repeat_index"].nunique() == 1
    assert revealed[FITTED_ROW_UID_COLUMN].nunique() == len(revealed)


def test_exact_mode_uses_all_loo_terms_by_default() -> None:
    revealed = _synthetic_revealed(rows_per_step=3)
    scores = compute_loo_fragility_scores(revealed, n_samples=20, n_grid=11, seed=123)
    assert (scores["n_loo_terms_available"] == 3).all()
    assert (scores["n_loo_terms_used"] == 3).all()
    assert not scores["loo_subsampled"].any()


def test_approximate_mode_uses_at_most_max_terms_per_step() -> None:
    revealed = _synthetic_revealed(rows_per_step=4)
    scores = compute_loo_fragility_scores(
        revealed,
        n_samples=20,
        n_grid=11,
        seed=123,
        max_loo_terms_per_step=2,
        loo_subsample_seed=999,
    )
    assert (scores["n_loo_terms_available"] == 4).all()
    assert (scores["n_loo_terms_used"] == 2).all()
    assert scores["loo_subsampled"].all()
    assert (scores["max_loo_terms_per_step"] == 2).all()


def test_approximate_mode_is_reproducible_with_fixed_seed() -> None:
    revealed = _synthetic_revealed(rows_per_step=4)
    first = compute_loo_fragility_scores(
        revealed,
        n_samples=20,
        n_grid=11,
        seed=123,
        max_loo_terms_per_step=2,
        loo_subsample_seed=999,
    )
    second = compute_loo_fragility_scores(
        revealed,
        n_samples=20,
        n_grid=11,
        seed=123,
        max_loo_terms_per_step=2,
        loo_subsample_seed=999,
    )
    assert np.allclose(first["loo_fragility"], second["loo_fragility"])


def test_fragility_table_includes_subsampling_accounting_columns() -> None:
    scores = compute_loo_fragility_scores(_synthetic_revealed(rows_per_step=2), n_samples=20, n_grid=11, seed=123)
    for col in ["n_loo_terms_available", "n_loo_terms_used", "loo_subsampled", "max_loo_terms_per_step"]:
        assert col in scores.columns
