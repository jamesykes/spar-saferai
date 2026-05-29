from __future__ import annotations

import pandas as pd
import pytest

from saferai_budget_recovery import config
from saferai_budget_recovery.reveal import (
    make_initial_seed_reveal,
    make_uniform_reveal_order,
    revealed_at_budget,
    split_revealed_unrevealed,
    usable_fit_rows,
)


def _complete_fit_df(rows_per_group: int = 2) -> pd.DataFrame:
    rows = []
    for step in config.EXPECTED_MITRE_STEP_LABELS:
        for model in config.EXPECTED_LLM_MODELS:
            for repeat in range(rows_per_group):
                rows.append(
                    {
                        "step_name": step,
                        "model": model,
                        "run_id": "run-a",
                        "repeat_index": repeat,
                        "draw_uid": f"{model}__run-a__{repeat}",
                        "alpha": 2.0,
                        "beta": 5.0,
                        "fit_quality_flag": "ok",
                    }
                )
    return pd.DataFrame(rows)


def test_initial_seed_allocation_returns_45_rows_on_complete_data() -> None:
    initial = make_initial_seed_reveal(_complete_fit_df(), seed=123)
    assert len(initial) == 45


def test_initial_seed_has_one_row_per_step_model() -> None:
    initial = make_initial_seed_reveal(_complete_fit_df(), seed=123)
    counts = initial.groupby(["step_name", "model"]).size()
    assert counts.nunique() == 1
    assert counts.iloc[0] == 1
    assert len(counts) == 45


def test_split_revealed_unrevealed_preserves_total_and_has_no_overlap() -> None:
    fit_df = _complete_fit_df()
    initial = make_initial_seed_reveal(fit_df, seed=123)
    revealed, unrevealed = split_revealed_unrevealed(fit_df, initial)
    assert len(revealed) + len(unrevealed) == len(usable_fit_rows(fit_df))
    assert set(revealed["fitted_row_uid"]).isdisjoint(set(unrevealed["fitted_row_uid"]))


def test_uniform_reveal_order_is_reproducible() -> None:
    fit_df = _complete_fit_df()
    initial = make_initial_seed_reveal(fit_df, seed=123)
    _, unrevealed = split_revealed_unrevealed(fit_df, initial)
    first = make_uniform_reveal_order(unrevealed, seed=456)
    second = make_uniform_reveal_order(unrevealed, seed=456)
    assert list(first["fitted_row_uid"]) == list(second["fitted_row_uid"])
    assert list(first["uniform_reveal_rank"]) == list(range(1, len(first) + 1))


def test_revealed_at_budget_includes_initial_and_respects_budget() -> None:
    fit_df = _complete_fit_df()
    initial = make_initial_seed_reveal(fit_df, seed=123)
    _, unrevealed = split_revealed_unrevealed(fit_df, initial)
    order = make_uniform_reveal_order(unrevealed, seed=456)
    revealed = revealed_at_budget(initial, order, total_budget=60)
    assert len(revealed) == 60
    assert set(initial["fitted_row_uid"]).issubset(set(revealed["fitted_row_uid"]))


def test_revealed_at_budget_rejects_invalid_budgets() -> None:
    fit_df = _complete_fit_df()
    initial = make_initial_seed_reveal(fit_df, seed=123)
    _, unrevealed = split_revealed_unrevealed(fit_df, initial)
    order = make_uniform_reveal_order(unrevealed, seed=456)
    with pytest.raises(ValueError, match="smaller than the initial seed allocation"):
        revealed_at_budget(initial, order, total_budget=44)
    with pytest.raises(ValueError, match="exceeds the maximum available budget"):
        revealed_at_budget(initial, order, total_budget=len(fit_df) + 1)

