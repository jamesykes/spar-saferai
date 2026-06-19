from __future__ import annotations

import pandas as pd
import pytest

from saferai_budget_recovery import config
from saferai_budget_recovery.reveal import (
    FITTED_ROW_UID_COLUMN,
    initial_revealed_from_hidden_orders,
    make_hidden_reveal_orders,
    make_initial_seed_reveal,
    make_uniform_reveal_order,
    reveal_next_for_step,
    revealed_at_budget,
    revealed_df_from_row_ids,
    split_revealed_unrevealed,
    unrevealed_steps_available,
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


def test_hidden_reveal_orders_initial_seed_has_45_rows() -> None:
    fit_df = _complete_fit_df(rows_per_group=3)
    hidden = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    initial = initial_revealed_from_hidden_orders(fit_df, hidden)
    assert len(hidden.initial_row_ids) == 45
    assert len(initial) == 45


def test_hidden_reveal_orders_have_one_initial_row_per_step_model() -> None:
    fit_df = _complete_fit_df(rows_per_group=3)
    hidden = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    initial = initial_revealed_from_hidden_orders(fit_df, hidden)
    counts = initial.groupby(["step_name", "model"]).size()
    assert len(counts) == 45
    assert counts.nunique() == 1
    assert counts.iloc[0] == 1


def test_hidden_reveal_orders_cover_every_usable_row_once() -> None:
    fit_df = _complete_fit_df(rows_per_group=3)
    usable = usable_fit_rows(fit_df)
    hidden = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    covered = hidden.initial_row_ids + [
        row_id for step_order in hidden.orders_by_step.values() for row_id in step_order
    ]
    assert len(covered) == len(usable)
    assert len(set(covered)) == len(usable)
    assert set(covered) == set(usable[FITTED_ROW_UID_COLUMN])
    assert hidden.metadata["coverage"]["covers_all_usable_rows_exactly_once"] is True


def test_hidden_reveal_orders_are_reproducible_for_same_seed() -> None:
    fit_df = _complete_fit_df(rows_per_group=3)
    first = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    second = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    assert first.initial_row_ids == second.initial_row_ids
    assert first.orders_by_step == second.orders_by_step


def test_hidden_reveal_orders_differ_for_different_seeds() -> None:
    fit_df = _complete_fit_df(rows_per_group=4)
    first = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    second = make_hidden_reveal_orders(fit_df, reveal_seed=456)
    assert first.initial_row_ids != second.initial_row_ids or first.orders_by_step != second.orders_by_step


def test_hidden_reveal_orders_cycle_across_models_after_initial_seed() -> None:
    fit_df = _complete_fit_df(rows_per_group=3)
    hidden = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    step = config.EXPECTED_MITRE_STEP_LABELS[0]
    first_cycle_ids = hidden.orders_by_step[step][: len(config.EXPECTED_LLM_MODELS)]
    first_cycle = revealed_df_from_row_ids(fit_df, first_cycle_ids)
    assert set(first_cycle["model"]) == set(config.EXPECTED_LLM_MODELS)


def test_hidden_reveal_orders_missing_step_model_group_raises_in_strict_mode() -> None:
    fit_df = _complete_fit_df(rows_per_group=2)
    missing_step = config.EXPECTED_MITRE_STEP_LABELS[0]
    missing_model = config.EXPECTED_LLM_MODELS[0]
    fit_df = fit_df.loc[
        ~(fit_df["step_name"].eq(missing_step) & fit_df["model"].eq(missing_model))
    ].copy()
    with pytest.raises(ValueError, match="missing required"):
        make_hidden_reveal_orders(fit_df, reveal_seed=123, strict=True)


def test_reveal_next_for_step_returns_pre_materialized_order_without_randomness() -> None:
    fit_df = _complete_fit_df(rows_per_group=3)
    hidden = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    step = config.EXPECTED_MITRE_STEP_LABELS[0]
    revealed_ids = set(hidden.initial_row_ids)
    first = reveal_next_for_step(fit_df, hidden, revealed_ids, step)
    second = reveal_next_for_step(fit_df, hidden, revealed_ids, step)
    assert first == hidden.orders_by_step[step][0]
    assert second == first
    revealed_ids.add(first)
    assert reveal_next_for_step(fit_df, hidden, revealed_ids, step) == hidden.orders_by_step[step][1]


def test_reveal_next_for_step_raises_when_step_exhausted() -> None:
    fit_df = _complete_fit_df(rows_per_group=2)
    hidden = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    step = config.EXPECTED_MITRE_STEP_LABELS[0]
    revealed_ids = set(hidden.initial_row_ids) | set(hidden.orders_by_step[step])
    assert step not in unrevealed_steps_available(hidden, revealed_ids)
    with pytest.raises(ValueError, match="No unrevealed rows remain"):
        reveal_next_for_step(fit_df, hidden, revealed_ids, step)


def test_same_step_sequence_receives_identical_row_ids_under_same_hidden_order() -> None:
    fit_df = _complete_fit_df(rows_per_group=3)
    hidden = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    steps = [
        config.EXPECTED_MITRE_STEP_LABELS[0],
        config.EXPECTED_MITRE_STEP_LABELS[1],
        config.EXPECTED_MITRE_STEP_LABELS[0],
    ]
    assert _simulate_reveals(fit_df, hidden, steps) == _simulate_reveals(fit_df, hidden, steps)


def test_same_number_of_draws_from_step_receives_same_step_prefix() -> None:
    fit_df = _complete_fit_df(rows_per_group=4)
    hidden = make_hidden_reveal_orders(fit_df, reveal_seed=123)
    step_a = config.EXPECTED_MITRE_STEP_LABELS[0]
    step_b = config.EXPECTED_MITRE_STEP_LABELS[1]
    step_c = config.EXPECTED_MITRE_STEP_LABELS[2]
    first = _simulate_reveals(fit_df, hidden, [step_a, step_b, step_a, step_a])
    second = _simulate_reveals(fit_df, hidden, [step_b, step_a, step_c, step_a, step_a])
    assert first[step_a] == second[step_a]


def _simulate_reveals(fit_df: pd.DataFrame, hidden, steps: list[str]) -> dict[str, list[str]]:
    revealed_ids = set(hidden.initial_row_ids)
    received: dict[str, list[str]] = {}
    for step in steps:
        row_id = reveal_next_for_step(fit_df, hidden, revealed_ids, step)
        revealed_ids.add(row_id)
        received.setdefault(step, []).append(row_id)
    return received
