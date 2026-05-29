"""Aggregation helpers for policy-comparison experiment outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


DISTANCE_COL = "squared_wasserstein2_to_full_reference"


def summarize_policy_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize policy distances and diagnostics by policy and budget."""

    _require_columns(
        results_df,
        {
            "policy_name",
            "budget",
            DISTANCE_COL,
            "p_success_mean",
            "step_count_l1_from_perfect_balance",
        },
    )
    rows: list[dict] = []
    for (policy, budget), group in results_df.groupby(["policy_name", "budget"], dropna=False):
        distances = group[DISTANCE_COL].to_numpy(dtype=float)
        imbalance = group["step_count_l1_from_perfect_balance"].to_numpy(dtype=float)
        rows.append(
            {
                "policy_name": policy,
                "budget": int(budget),
                "n_runs": int(len(group)),
                "distance_mean": float(np.mean(distances)),
                "distance_median": float(np.median(distances)),
                "distance_std": float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0,
                "distance_min": float(np.min(distances)),
                "distance_max": float(np.max(distances)),
                "distance_p25": float(np.percentile(distances, 25)),
                "distance_p75": float(np.percentile(distances, 75)),
                "p_success_mean_mean": float(group["p_success_mean"].mean()),
                "step_count_l1_mean": float(np.mean(imbalance)),
                "step_count_l1_median": float(np.median(imbalance)),
                "step_count_l1_max": float(np.max(imbalance)),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy_name", "budget"]).reset_index(drop=True)


def compute_policy_differences(
    results_df: pd.DataFrame,
    baseline_policy: str = "uniform_step_balanced",
    comparator_policy: str = "greedy_loo_fragility",
) -> pd.DataFrame:
    """Compute paired comparator-minus-baseline distances by reveal seed and budget."""

    _require_columns(results_df, {"policy_name", "reveal_seed", "budget", DISTANCE_COL})
    baseline = results_df.loc[results_df["policy_name"].eq(baseline_policy), [
        "reveal_seed",
        "budget",
        DISTANCE_COL,
    ]].rename(columns={DISTANCE_COL: "uniform_distance"})
    comparator = results_df.loc[results_df["policy_name"].eq(comparator_policy), [
        "reveal_seed",
        "budget",
        DISTANCE_COL,
    ]].rename(columns={DISTANCE_COL: "greedy_distance"})
    merged = baseline.merge(comparator, on=["reveal_seed", "budget"], how="inner")
    merged["difference_greedy_minus_uniform"] = merged["greedy_distance"] - merged["uniform_distance"]
    merged["relative_difference_greedy_minus_uniform"] = np.where(
        merged["uniform_distance"].to_numpy(dtype=float) == 0,
        np.nan,
        merged["difference_greedy_minus_uniform"] / merged["uniform_distance"],
    )
    merged["greedy_better"] = merged["greedy_distance"] < merged["uniform_distance"]
    return merged.sort_values(["reveal_seed", "budget"]).reset_index(drop=True)


def compute_auc_by_seed_policy(
    results_df: pd.DataFrame,
    distance_col: str = DISTANCE_COL,
) -> pd.DataFrame:
    """Compute simple trapezoidal AUC diagnostics over budget for each seed and policy."""

    _require_columns(results_df, {"policy_name", "reveal_seed", "budget", distance_col})
    rows: list[dict] = []
    for (policy, seed), group in results_df.groupby(["policy_name", "reveal_seed"], dropna=False):
        ordered = group.sort_values("budget")
        budgets = ordered["budget"].to_numpy(dtype=float)
        distances = ordered[distance_col].to_numpy(dtype=float)
        auc = float(np.trapezoid(distances, budgets)) if len(ordered) > 1 else 0.0
        rows.append(
            {
                "policy_name": policy,
                "reveal_seed": int(seed),
                "auc_distance": auc,
                "mean_distance_across_budgets": float(np.mean(distances)),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy_name", "reveal_seed"]).reset_index(drop=True)


def summarize_win_rate_by_budget(differences_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired greedy-vs-uniform wins by budget."""

    _require_columns(
        differences_df,
        {
            "budget",
            "greedy_better",
            "difference_greedy_minus_uniform",
            "relative_difference_greedy_minus_uniform",
        },
    )
    rows: list[dict] = []
    for budget, group in differences_df.groupby("budget", dropna=False):
        rows.append(
            {
                "budget": int(budget),
                "n_seeds": int(len(group)),
                "greedy_wins": int(group["greedy_better"].sum()),
                "greedy_win_fraction": float(group["greedy_better"].mean()),
                "mean_greedy_minus_uniform_distance": float(
                    group["difference_greedy_minus_uniform"].mean()
                ),
                "median_greedy_minus_uniform_distance": float(
                    group["difference_greedy_minus_uniform"].median()
                ),
                "mean_relative_difference_greedy_minus_uniform": _finite_mean(
                    group["relative_difference_greedy_minus_uniform"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("budget").reset_index(drop=True)


def summarize_win_rate_by_seed(
    differences_df: pd.DataFrame,
    auc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired greedy-vs-uniform wins and AUC differences by reveal seed."""

    _require_columns(differences_df, {"reveal_seed", "greedy_better"})
    _require_columns(auc_df, {"policy_name", "reveal_seed", "auc_distance"})
    auc_wide = auc_df.pivot(index="reveal_seed", columns="policy_name", values="auc_distance")
    rows: list[dict] = []
    for seed, group in differences_df.groupby("reveal_seed", dropna=False):
        uniform_auc = float(auc_wide.loc[seed, "uniform_step_balanced"])
        greedy_auc = float(auc_wide.loc[seed, "greedy_loo_fragility"])
        rows.append(
            {
                "reveal_seed": int(seed),
                "n_budgets": int(len(group)),
                "greedy_wins": int(group["greedy_better"].sum()),
                "greedy_win_fraction": float(group["greedy_better"].mean()),
                "greedy_auc": greedy_auc,
                "uniform_auc": uniform_auc,
                "greedy_minus_uniform_auc": float(greedy_auc - uniform_auc),
            }
        )
    return pd.DataFrame(rows).sort_values("reveal_seed").reset_index(drop=True)


def summarize_concentration_by_budget(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize step-allocation concentration diagnostics by policy and budget."""

    _require_columns(
        results_df,
        {
            "policy_name",
            "budget",
            "step_count_l1_from_perfect_balance",
            "max_min_revealed_rows_per_step_ratio",
            "max_revealed_rows_per_step",
            "min_revealed_rows_per_step",
            "revealed_rows_by_step",
        },
    )
    enriched = results_df.copy()
    enriched["n_steps_above_initial_5"] = enriched["revealed_rows_by_step"].map(
        lambda raw: _count_step_condition(raw, lambda value: value > 5)
    )
    enriched["n_steps_at_initial_5"] = enriched["revealed_rows_by_step"].map(
        lambda raw: _count_step_condition(raw, lambda value: value == 5)
    )
    rows: list[dict] = []
    for (policy, budget), group in enriched.groupby(["policy_name", "budget"], dropna=False):
        rows.append(
            {
                "policy_name": str(policy),
                "budget": int(budget),
                "n_runs": int(len(group)),
                "mean_step_count_l1_imbalance": float(
                    group["step_count_l1_from_perfect_balance"].mean()
                ),
                "median_step_count_l1_imbalance": float(
                    group["step_count_l1_from_perfect_balance"].median()
                ),
                "max_step_count_l1_imbalance": float(
                    group["step_count_l1_from_perfect_balance"].max()
                ),
                "mean_max_min_revealed_row_ratio": float(
                    group["max_min_revealed_rows_per_step_ratio"].mean()
                ),
                "maximum_observed_revealed_row_count_any_step": int(
                    group["max_revealed_rows_per_step"].max()
                ),
                "minimum_observed_revealed_row_count_any_step": int(
                    group["min_revealed_rows_per_step"].min()
                ),
                "average_steps_above_initial_5": float(group["n_steps_above_initial_5"].mean()),
                "average_steps_still_at_initial_5": float(group["n_steps_at_initial_5"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy_name", "budget"]).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results DataFrame is missing required columns: {sorted(missing)}")


def _finite_mean(series: pd.Series) -> float:
    finite = series[np.isfinite(series)]
    return float(finite.mean()) if len(finite) else np.nan


def _count_step_condition(raw_counts: str, predicate) -> int:
    counts = json.loads(raw_counts)
    return int(sum(1 for value in counts.values() if predicate(int(value))))
