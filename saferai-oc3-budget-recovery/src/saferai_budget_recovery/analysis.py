"""Aggregation helpers for policy-comparison experiment outputs."""

from __future__ import annotations

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


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results DataFrame is missing required columns: {sorted(missing)}")

