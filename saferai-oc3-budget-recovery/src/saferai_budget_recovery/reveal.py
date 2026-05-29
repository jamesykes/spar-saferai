"""Reveal-protocol primitives for budgeted SOTA elicitation rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from saferai_budget_recovery import config
from saferai_budget_recovery.mixtures import USABLE_FIT_FLAGS


FITTED_ROW_UID_COLUMN = "fitted_row_uid"


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

