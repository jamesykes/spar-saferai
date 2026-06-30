"""Reveal-protocol primitives for budgeted SOTA elicitation rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np
import pandas as pd

from saferai_budget_recovery import config
from saferai_budget_recovery.mixtures import USABLE_FIT_FLAGS


FITTED_ROW_UID_COLUMN = "fitted_row_uid"


@dataclass(frozen=True)
class HiddenRevealOrders:
    """Seeded v8 hidden reveal orders shared by all policies for one seed."""

    orders_by_step: dict[str, list[str]]
    initial_row_ids: list[str]
    metadata: dict[str, Any]


def usable_fit_rows(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Return usable fitted SOTA rows with a stable row identifier."""

    required = {
        "step_name",
        "model",
        "run_id",
        "repeat_index",
        "draw_uid",
        "alpha",
        "beta",
        "fit_quality_flag",
    }
    missing = required - set(fit_df.columns)
    if missing:
        raise ValueError(f"Cannot identify usable fitted rows; missing columns: {sorted(missing)}")

    usable = fit_df.loc[
        fit_df["fit_quality_flag"].isin(USABLE_FIT_FLAGS)
        & np.isfinite(fit_df["alpha"])
        & np.isfinite(fit_df["beta"])
        & (fit_df["alpha"] > 0)
        & (fit_df["beta"] > 0)
    ].copy()
    usable[FITTED_ROW_UID_COLUMN] = (
        usable["step_name"].astype(str) + "__" + usable["draw_uid"].astype(str)
    )
    if usable[FITTED_ROW_UID_COLUMN].duplicated().any():
        duplicates = (
            usable.loc[usable[FITTED_ROW_UID_COLUMN].duplicated(), FITTED_ROW_UID_COLUMN]
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Fitted row identifiers are not unique. "
            f"Duplicate examples based on (step_name, draw_uid): {duplicates}"
        )
    return usable.reset_index(drop=True)


def make_hidden_reveal_orders(
    fit_df: pd.DataFrame,
    reveal_seed: int,
    strict: bool = True,
) -> HiddenRevealOrders:
    """Create v8 model-aware hidden reveal orders for one outer reveal seed.

    For each MITRE-step input, rows are split by LLM model, shuffled within
    model, and revealed by cycling through a seed-specific model order. The
    initial seed allocation is the first shuffled row from each model.
    """

    usable = usable_fit_rows(fit_df)
    initial_row_ids: list[str] = []
    orders_by_step: dict[str, list[str]] = {}
    model_order_by_step: dict[str, list[str]] = {}
    post_seed_model_sequence_by_step: dict[str, list[str]] = {}
    missing_groups: list[tuple[str, str]] = []

    for step_index, step in enumerate(config.EXPECTED_MITRE_STEP_LABELS):
        step_rows = usable.loc[usable["step_name"].eq(step)].copy()
        shuffled_remaining_by_model: dict[str, list[str]] = {}
        present_models: list[str] = []

        for model_index, model in enumerate(config.EXPECTED_LLM_MODELS):
            group = step_rows.loc[step_rows["model"].eq(model)].copy()
            if group.empty:
                missing_groups.append((step, model))
                continue
            ordered_group = _stable_sort_fit_rows(group)
            rng = np.random.default_rng(
                _derived_seed(reveal_seed, step_index, model_index, salt=101)
            )
            shuffled_positions = rng.permutation(len(ordered_group))
            shuffled_ids = (
                ordered_group.iloc[shuffled_positions][FITTED_ROW_UID_COLUMN]
                .astype(str)
                .tolist()
            )
            initial_row_ids.append(shuffled_ids[0])
            shuffled_remaining_by_model[model] = shuffled_ids[1:]
            present_models.append(model)

        if missing_groups and strict:
            continue

        if not present_models:
            if strict:
                missing_groups.extend((step, model) for model in config.EXPECTED_LLM_MODELS)
            orders_by_step[step] = []
            model_order_by_step[step] = []
            post_seed_model_sequence_by_step[step] = []
            continue

        model_rng = np.random.default_rng(_derived_seed(reveal_seed, step_index, 0, salt=909))
        model_order = [
            present_models[int(position)]
            for position in model_rng.permutation(len(present_models))
        ]
        model_order_by_step[step] = model_order

        post_seed_ids: list[str] = []
        post_seed_models: list[str] = []
        remaining_by_model = {model: list(ids) for model, ids in shuffled_remaining_by_model.items()}
        while any(remaining_by_model[model] for model in model_order):
            for model in model_order:
                if remaining_by_model[model]:
                    post_seed_ids.append(remaining_by_model[model].pop(0))
                    post_seed_models.append(model)

        orders_by_step[step] = post_seed_ids
        post_seed_model_sequence_by_step[step] = post_seed_models

    if missing_groups and strict:
        raise ValueError(
            "Hidden reveal orders are missing required (step, model) groups: "
            f"{missing_groups}"
        )

    all_ordered_ids = initial_row_ids + [
        row_id for step in config.EXPECTED_MITRE_STEP_LABELS for row_id in orders_by_step.get(step, [])
    ]
    usable_ids = usable[FITTED_ROW_UID_COLUMN].astype(str).tolist()
    duplicate_ids = sorted(_duplicates(all_ordered_ids))
    missing_ids = sorted(set(usable_ids) - set(all_ordered_ids))
    extra_ids = sorted(set(all_ordered_ids) - set(usable_ids))
    if duplicate_ids or missing_ids or extra_ids:
        raise ValueError(
            "Hidden reveal-order coverage invariant failed. "
            f"duplicates={duplicate_ids[:5]}, missing={missing_ids[:5]}, extra={extra_ids[:5]}"
        )

    metadata = {
        "reveal_seed": int(reveal_seed),
        "strict": bool(strict),
        "protocol": "v8_model_aware_hidden_reveal_orders",
        "uses_model_aware_cycling": True,
        "expected_steps": list(config.EXPECTED_MITRE_STEP_LABELS),
        "expected_models": list(config.EXPECTED_LLM_MODELS),
        "initial_seed_size": int(len(initial_row_ids)),
        "total_usable_fitted_rows": int(len(usable)),
        "post_seed_order_lengths_by_step": {
            step: int(len(orders_by_step.get(step, [])))
            for step in config.EXPECTED_MITRE_STEP_LABELS
        },
        "initial_count_by_step": {
            step: int(
                usable.loc[
                    usable[FITTED_ROW_UID_COLUMN].isin(initial_row_ids)
                    & usable["step_name"].eq(step)
                ].shape[0]
            )
            for step in config.EXPECTED_MITRE_STEP_LABELS
        },
        "initial_count_by_model": {
            model: int(
                usable.loc[
                    usable[FITTED_ROW_UID_COLUMN].isin(initial_row_ids)
                    & usable["model"].eq(model)
                ].shape[0]
            )
            for model in config.EXPECTED_LLM_MODELS
        },
        "model_order_by_step": model_order_by_step,
        "post_seed_model_sequence_by_step": post_seed_model_sequence_by_step,
        "coverage": {
            "covered_row_count": int(len(all_ordered_ids)),
            "unique_covered_row_count": int(len(set(all_ordered_ids))),
            "usable_row_count": int(len(usable_ids)),
            "duplicate_row_count": int(len(all_ordered_ids) - len(set(all_ordered_ids))),
            "missing_row_count": 0,
            "extra_row_count": 0,
            "covers_all_usable_rows_exactly_once": True,
        },
    }
    return HiddenRevealOrders(
        orders_by_step=orders_by_step,
        initial_row_ids=initial_row_ids,
        metadata=metadata,
    )


def initial_revealed_from_hidden_orders(
    fit_df: pd.DataFrame,
    hidden_orders: HiddenRevealOrders,
) -> pd.DataFrame:
    """Return the initial seed allocation encoded in hidden reveal orders."""

    return revealed_df_from_row_ids(fit_df, hidden_orders.initial_row_ids)


def reveal_next_for_step(
    fit_df: pd.DataFrame,
    hidden_orders: HiddenRevealOrders,
    revealed_row_ids: set[str],
    step_name: str,
) -> str:
    """Return the next unrevealed row ID for one step from its hidden order."""

    if step_name not in hidden_orders.orders_by_step:
        raise ValueError(f"Unknown MITRE-step label for hidden reveal order: {step_name}")
    # Validate the row IDs against fit_df so bad caller state fails clearly.
    _ = _row_id_set_for_fit_df(fit_df)
    for row_id in hidden_orders.orders_by_step[step_name]:
        if row_id not in revealed_row_ids:
            return row_id
    raise ValueError(f"No unrevealed rows remain for MITRE-step label: {step_name}")


def unrevealed_steps_available(
    hidden_orders: HiddenRevealOrders,
    revealed_row_ids: set[str],
) -> set[str]:
    """Return steps with at least one post-seed hidden row still unrevealed."""

    return {
        step
        for step, row_ids in hidden_orders.orders_by_step.items()
        if any(row_id not in revealed_row_ids for row_id in row_ids)
    }


def revealed_df_from_row_ids(
    fit_df: pd.DataFrame,
    row_ids: Iterable[str],
) -> pd.DataFrame:
    """Return usable fitted rows matching row IDs, preserving provided order when possible."""

    usable = usable_fit_rows(fit_df)
    if isinstance(row_ids, set):
        row_id_list = [
            str(row_id)
            for row_id in usable[FITTED_ROW_UID_COLUMN].astype(str).tolist()
            if str(row_id) in row_ids
        ]
    else:
        row_id_list = [str(row_id) for row_id in row_ids]
    if len(row_id_list) != len(set(row_id_list)):
        raise ValueError("Requested revealed row IDs contain duplicates.")
    row_by_id = {
        str(row[FITTED_ROW_UID_COLUMN]): row
        for row in usable.to_dict(orient="records")
    }
    missing = [row_id for row_id in row_id_list if row_id not in row_by_id]
    if missing:
        raise ValueError(f"Requested row IDs are absent from usable fitted rows: {missing[:10]}")
    return pd.DataFrame([row_by_id[row_id] for row_id in row_id_list]).reset_index(drop=True)


def unrevealed_df_from_hidden_orders(
    fit_df: pd.DataFrame,
    hidden_orders: HiddenRevealOrders,
    revealed_row_ids: set[str],
) -> pd.DataFrame:
    """Return all hidden-order rows not yet revealed."""

    all_ids = hidden_orders.initial_row_ids + [
        row_id
        for step in config.EXPECTED_MITRE_STEP_LABELS
        for row_id in hidden_orders.orders_by_step.get(step, [])
    ]
    unrevealed_ids = [row_id for row_id in all_ids if row_id not in revealed_row_ids]
    return revealed_df_from_row_ids(fit_df, unrevealed_ids)


def make_initial_seed_reveal(
    fit_df: pd.DataFrame,
    seed: int = 12345,
    strict: bool = True,
) -> pd.DataFrame:
    """Reveal one fitted row per expected MITRE-step and expected LLM model."""

    usable = usable_fit_rows(fit_df)
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    missing_groups: list[tuple[str, str]] = []

    for step in config.EXPECTED_MITRE_STEP_LABELS:
        for model in config.EXPECTED_LLM_MODELS:
            group = usable.loc[usable["step_name"].eq(step) & usable["model"].eq(model)]
            if group.empty:
                missing_groups.append((step, model))
                continue
            selected_indices.append(int(group.index[rng.integers(0, len(group))]))

    if missing_groups and strict:
        raise ValueError(
            "Initial seed allocation is missing required (step, model) groups: "
            f"{missing_groups}"
        )

    revealed = usable.loc[selected_indices].copy()
    revealed["initial_seed_rank"] = np.arange(1, len(revealed) + 1)
    return revealed.reset_index(drop=True)


def split_revealed_unrevealed(
    fit_df: pd.DataFrame,
    revealed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split usable fitted rows into revealed and unrevealed sets without overlap."""

    usable = usable_fit_rows(fit_df)
    revealed = _ensure_fitted_row_uid(revealed_df)
    revealed_ids = set(revealed[FITTED_ROW_UID_COLUMN])
    unknown_ids = revealed_ids - set(usable[FITTED_ROW_UID_COLUMN])
    if unknown_ids:
        raise ValueError(f"Revealed rows contain identifiers absent from fitted rows: {sorted(unknown_ids)[:10]}")

    revealed_out = usable.loc[usable[FITTED_ROW_UID_COLUMN].isin(revealed_ids)].copy()
    unrevealed_out = usable.loc[~usable[FITTED_ROW_UID_COLUMN].isin(revealed_ids)].copy()
    if len(revealed_out) != len(revealed_ids):
        raise ValueError("Revealed rows contain duplicate fitted row identifiers.")
    return revealed_out.reset_index(drop=True), unrevealed_out.reset_index(drop=True)


def make_uniform_reveal_order(
    unrevealed_df: pd.DataFrame,
    seed: int = 12345,
) -> pd.DataFrame:
    """Return unrevealed rows in a reproducible random order with 1-based ranks."""

    unrevealed = _ensure_fitted_row_uid(unrevealed_df)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unrevealed))
    ordered = unrevealed.iloc[order].copy().reset_index(drop=True)
    ordered["uniform_reveal_rank"] = np.arange(1, len(ordered) + 1)
    return ordered


def revealed_at_budget(
    initial_revealed_df: pd.DataFrame,
    uniform_order_df: pd.DataFrame,
    total_budget: int,
) -> pd.DataFrame:
    """Return initial seed rows plus uniform-revealed rows up to total_budget."""

    initial = _ensure_fitted_row_uid(initial_revealed_df)
    uniform_order = _ensure_fitted_row_uid(uniform_order_df)
    initial_size = len(initial)
    max_budget = initial_size + len(uniform_order)
    if total_budget < initial_size:
        raise ValueError(
            f"total_budget={total_budget} is smaller than the initial seed allocation size {initial_size}."
        )
    if total_budget > max_budget:
        raise ValueError(f"total_budget={total_budget} exceeds the maximum available budget {max_budget}.")

    n_additional = total_budget - initial_size
    if "uniform_reveal_rank" in uniform_order.columns:
        additional = uniform_order.sort_values("uniform_reveal_rank").head(n_additional)
    else:
        additional = uniform_order.head(n_additional)
    revealed = pd.concat([initial, additional], ignore_index=True)
    if revealed[FITTED_ROW_UID_COLUMN].duplicated().any():
        raise ValueError("Budgeted revealed set contains duplicate fitted row identifiers.")
    return revealed.reset_index(drop=True)


def _ensure_fitted_row_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if FITTED_ROW_UID_COLUMN not in out.columns:
        required = {"step_name", "draw_uid"}
        missing = required - set(out.columns)
        if missing:
            raise ValueError(f"Cannot create fitted row identifier; missing columns: {sorted(missing)}")
        out[FITTED_ROW_UID_COLUMN] = out["step_name"].astype(str) + "__" + out["draw_uid"].astype(str)
    return out


def _stable_sort_fit_rows(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        col
        for col in ("model", "run_id", "repeat_index", "draw_uid", FITTED_ROW_UID_COLUMN)
        if col in df.columns
    ]
    return df.sort_values(sort_columns).reset_index(drop=True)


def _derived_seed(reveal_seed: int, step_index: int, model_index: int, salt: int) -> int:
    # Use only deterministic integer arithmetic; do not rely on Python's salted hash().
    return int(
        (int(reveal_seed) * 1_000_003)
        + ((step_index + 1) * 10_007)
        + ((model_index + 1) * 101)
        + salt
    ) % (2**32 - 1)


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _row_id_set_for_fit_df(fit_df: pd.DataFrame) -> set[str]:
    return set(usable_fit_rows(fit_df)[FITTED_ROW_UID_COLUMN].astype(str))
