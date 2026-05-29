"""Helpers for auditing approximate LOO fragility rankings."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


FRAGILITY_COL = "loo_fragility"


def rank_fragility_scores(fragility_df: pd.DataFrame) -> pd.DataFrame:
    """Return fragility scores with rank 1 assigned to the largest finite score."""

    required = {"step_name", FRAGILITY_COL}
    missing = required - set(fragility_df.columns)
    if missing:
        raise ValueError(f"Fragility table is missing columns: {sorted(missing)}")

    out = fragility_df.copy()
    out["fragility_rank"] = np.nan
    finite = out.loc[np.isfinite(out[FRAGILITY_COL])].copy()
    finite = finite.sort_values([FRAGILITY_COL, "step_name"], ascending=[False, True])
    out.loc[finite.index, "fragility_rank"] = np.arange(1, len(finite) + 1, dtype=float)
    return out


def compare_fragility_rankings(
    exact_df: pd.DataFrame,
    approx_df: pd.DataFrame,
    top_k_values: tuple[int, ...] = (1, 3),
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compare exact and approximate LOO fragility scores for one revealed state."""

    if 1 not in top_k_values:
        top_k_values = (1, *top_k_values)
    if 3 not in top_k_values:
        top_k_values = (*top_k_values, 3)

    merged = _ranked_merge(exact_df, approx_df)
    finite = merged.loc[
        np.isfinite(merged["exact_loo_fragility"])
        & np.isfinite(merged["approx_loo_fragility"])
    ].copy()
    if finite.empty:
        return _empty_comparison()

    exact_top1 = _top_steps(finite, "exact_rank", 1)
    approx_top1 = _top_steps(finite, "approx_rank", 1)
    exact_top3 = _top_steps(finite, "exact_rank", 3)
    approx_top3 = _top_steps(finite, "approx_rank", 3)
    top3_overlap = set(exact_top3) & set(approx_top3)
    top3_union = set(exact_top3) | set(approx_top3)

    absolute_error = np.abs(finite["approx_loo_fragility"] - finite["exact_loo_fragility"])
    relative_error = absolute_error / (np.abs(finite["exact_loo_fragility"]) + eps)
    exact_max = float(finite["exact_loo_fragility"].max())
    approx_max = float(finite["approx_loo_fragility"].max())

    return {
        "top1_exact": exact_top1[0] if exact_top1 else None,
        "top1_approx": approx_top1[0] if approx_top1 else None,
        "top1_match": bool(exact_top1 and approx_top1 and exact_top1[0] == approx_top1[0]),
        "top3_exact": ";".join(exact_top3),
        "top3_approx": ";".join(approx_top3),
        "top3_overlap_count": int(len(top3_overlap)),
        "top3_jaccard": float(len(top3_overlap) / len(top3_union)) if top3_union else np.nan,
        "spearman_rank_correlation": _safe_corr(
            finite["exact_loo_fragility"].to_numpy(dtype=float),
            finite["approx_loo_fragility"].to_numpy(dtype=float),
            method="spearman",
        ),
        "pearson_score_correlation": _safe_corr(
            finite["exact_loo_fragility"].to_numpy(dtype=float),
            finite["approx_loo_fragility"].to_numpy(dtype=float),
            method="pearson",
        ),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_relative_absolute_error": float(np.mean(relative_error)),
        "max_relative_absolute_error": float(np.max(relative_error)),
        "exact_max_score": exact_max,
        "approx_max_score": approx_max,
        "approx_exact_max_score_ratio": float(approx_max / exact_max) if abs(exact_max) > eps else np.nan,
        "exact_top_step_in_approx_top3": bool(exact_top1 and exact_top1[0] in set(approx_top3)),
        "n_steps_compared": int(len(finite)),
    }


def make_per_step_error_table(
    exact_df: pd.DataFrame,
    approx_df: pd.DataFrame,
    reveal_seed: int,
    budget: int,
    max_loo_terms_per_step: int,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Return long-form exact-vs-approx errors by MITRE-step input."""

    merged = _ranked_merge(exact_df, approx_df)
    absolute_error = np.abs(merged["approx_loo_fragility"] - merged["exact_loo_fragility"])
    relative_error = absolute_error / (np.abs(merged["exact_loo_fragility"]) + eps)
    return pd.DataFrame(
        {
            "reveal_seed": int(reveal_seed),
            "budget": int(budget),
            "max_loo_terms_per_step": int(max_loo_terms_per_step),
            "step_name": merged["step_name"],
            "exact_loo_fragility": merged["exact_loo_fragility"],
            "approx_loo_fragility": merged["approx_loo_fragility"],
            "absolute_error": absolute_error,
            "relative_absolute_error": relative_error,
            "exact_rank": merged["exact_rank"],
            "approx_rank": merged["approx_rank"],
            "n_revealed": merged["exact_n_revealed"],
            "exact_n_loo_terms_used": merged["exact_n_loo_terms_used"],
            "approx_n_loo_terms_used": merged["approx_n_loo_terms_used"],
            "exact_n_loo_terms_available": merged["exact_n_loo_terms_available"],
            "approx_n_loo_terms_available": merged["approx_n_loo_terms_available"],
            "approx_loo_subsampled": merged["approx_loo_subsampled"],
        }
    )


def summarize_fragility_approximation_audit(comparison_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize exact-vs-approx fragility comparisons by LOO-term cap."""

    if comparison_df.empty:
        return {}
    required = {
        "max_loo_terms_per_step",
        "top1_match",
        "top3_jaccard",
        "exact_top_step_in_approx_top3",
        "spearman_rank_correlation",
        "pearson_score_correlation",
        "mean_relative_absolute_error",
        "approx_exact_max_score_ratio",
    }
    missing = required - set(comparison_df.columns)
    if missing:
        raise ValueError(f"Comparison table is missing columns: {sorted(missing)}")

    summary: dict[str, Any] = {}
    for cap, group in comparison_df.groupby("max_loo_terms_per_step", sort=True):
        top1_rate = float(group["top1_match"].mean())
        mean_top3_jaccard = float(group["top3_jaccard"].mean())
        summary[str(int(cap))] = {
            "n_comparisons": int(len(group)),
            "top1_match_rate": top1_rate,
            "mean_top3_jaccard": mean_top3_jaccard,
            "exact_top_step_in_approx_top3_fraction": float(
                group["exact_top_step_in_approx_top3"].mean()
            ),
            "mean_spearman_correlation": _finite_mean(group["spearman_rank_correlation"]),
            "median_spearman_correlation": _finite_median(group["spearman_rank_correlation"]),
            "mean_pearson_correlation": _finite_mean(group["pearson_score_correlation"]),
            "median_pearson_correlation": _finite_median(group["pearson_score_correlation"]),
            "mean_relative_absolute_error": _finite_mean(group["mean_relative_absolute_error"]),
            "median_relative_absolute_error": _finite_median(group["mean_relative_absolute_error"]),
            "mean_approx_exact_max_score_ratio": _finite_mean(
                group["approx_exact_max_score_ratio"]
            ),
            "heuristic_warning_flag": (
                "appears_acceptable"
                if top1_rate >= 0.7 and mean_top3_jaccard >= 0.6
                else "questionable"
            ),
        }
    return summary


def _ranked_merge(exact_df: pd.DataFrame, approx_df: pd.DataFrame) -> pd.DataFrame:
    exact = rank_fragility_scores(exact_df).rename(
        columns={
            FRAGILITY_COL: "exact_loo_fragility",
            "fragility_rank": "exact_rank",
            "n_revealed": "exact_n_revealed",
            "n_loo_terms_used": "exact_n_loo_terms_used",
            "n_loo_terms_available": "exact_n_loo_terms_available",
        }
    )
    approx = rank_fragility_scores(approx_df).rename(
        columns={
            FRAGILITY_COL: "approx_loo_fragility",
            "fragility_rank": "approx_rank",
            "n_revealed": "approx_n_revealed",
            "n_loo_terms_used": "approx_n_loo_terms_used",
            "n_loo_terms_available": "approx_n_loo_terms_available",
            "loo_subsampled": "approx_loo_subsampled",
        }
    )
    keep_exact = [
        "step_name",
        "exact_loo_fragility",
        "exact_rank",
        "exact_n_revealed",
        "exact_n_loo_terms_used",
        "exact_n_loo_terms_available",
    ]
    keep_approx = [
        "step_name",
        "approx_loo_fragility",
        "approx_rank",
        "approx_n_revealed",
        "approx_n_loo_terms_used",
        "approx_n_loo_terms_available",
        "approx_loo_subsampled",
    ]
    merged = exact[keep_exact].merge(approx[keep_approx], on="step_name", how="outer")
    return merged.sort_values("step_name").reset_index(drop=True)


def _top_steps(df: pd.DataFrame, rank_col: str, k: int) -> list[str]:
    ranked = df.loc[np.isfinite(df[rank_col])].sort_values([rank_col, "step_name"])
    return [str(step) for step in ranked.head(k)["step_name"]]


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"Unknown correlation method: {method}")
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if math.isfinite(corr) else np.nan


def _finite_mean(values: pd.Series) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else np.nan


def _finite_median(values: pd.Series) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.median()) if len(finite) else np.nan


def _empty_comparison() -> dict[str, Any]:
    return {
        "top1_exact": None,
        "top1_approx": None,
        "top1_match": False,
        "top3_exact": "",
        "top3_approx": "",
        "top3_overlap_count": 0,
        "top3_jaccard": np.nan,
        "spearman_rank_correlation": np.nan,
        "pearson_score_correlation": np.nan,
        "mean_absolute_error": np.nan,
        "max_absolute_error": np.nan,
        "mean_relative_absolute_error": np.nan,
        "max_relative_absolute_error": np.nan,
        "exact_max_score": np.nan,
        "approx_max_score": np.nan,
        "approx_exact_max_score_ratio": np.nan,
        "exact_top_step_in_approx_top3": False,
        "n_steps_compared": 0,
    }
