from __future__ import annotations

import numpy as np
import pandas as pd

from saferai_budget_recovery.fragility_audit import (
    compare_fragility_rankings,
    make_per_step_error_table,
    rank_fragility_scores,
    summarize_fragility_approximation_audit,
)


def _fragility_df(scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    for i, (step, score) in enumerate(scores.items()):
        rows.append(
            {
                "step_name": step,
                "loo_fragility": score,
                "n_revealed": 5 + i,
                "n_loo_terms_used": 5 + i,
                "n_loo_terms_available": 5 + i,
                "loo_subsampled": False,
            }
        )
    return pd.DataFrame(rows)


def test_rank_fragility_scores_assigns_rank_one_to_highest_score() -> None:
    ranked = rank_fragility_scores(_fragility_df({"a": 0.1, "b": 0.3, "c": 0.2}))
    top = ranked.loc[ranked["fragility_rank"].eq(1)].iloc[0]
    assert top["step_name"] == "b"


def test_top1_match_true_when_highest_step_matches() -> None:
    exact = _fragility_df({"a": 0.5, "b": 0.3, "c": 0.1})
    approx = _fragility_df({"a": 0.4, "b": 0.2, "c": 0.1})
    comparison = compare_fragility_rankings(exact, approx)
    assert comparison["top1_match"] is True
    assert comparison["top1_exact"] == "a"
    assert comparison["top1_approx"] == "a"


def test_top3_jaccard_is_computed_correctly() -> None:
    exact = _fragility_df({"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2})
    approx = _fragility_df({"a": 0.5, "c": 0.4, "d": 0.3, "b": 0.2})
    comparison = compare_fragility_rankings(exact, approx)
    assert comparison["top3_overlap_count"] == 2
    assert comparison["top3_jaccard"] == 0.5


def test_correlations_are_finite_for_nondegenerate_scores() -> None:
    exact = _fragility_df({"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2})
    approx = _fragility_df({"a": 0.45, "b": 0.35, "c": 0.25, "d": 0.15})
    comparison = compare_fragility_rankings(exact, approx)
    assert np.isfinite(comparison["spearman_rank_correlation"])
    assert np.isfinite(comparison["pearson_score_correlation"])


def test_undefined_correlations_are_nan_for_constant_scores() -> None:
    exact = _fragility_df({"a": 0.1, "b": 0.1, "c": 0.1})
    approx = _fragility_df({"a": 0.2, "b": 0.2, "c": 0.2})
    comparison = compare_fragility_rankings(exact, approx)
    assert np.isnan(comparison["spearman_rank_correlation"])
    assert np.isnan(comparison["pearson_score_correlation"])


def test_relative_error_handles_zero_exact_scores() -> None:
    exact = _fragility_df({"a": 0.0, "b": 0.0, "c": 0.1})
    approx = _fragility_df({"a": 0.0, "b": 0.01, "c": 0.11})
    comparison = compare_fragility_rankings(exact, approx)
    assert np.isfinite(comparison["mean_relative_absolute_error"])


def test_per_step_comparison_preserves_all_steps() -> None:
    exact = _fragility_df({"a": 0.5, "b": 0.4, "c": 0.3})
    approx = _fragility_df({"a": 0.45, "b": 0.35, "c": 0.25})
    table = make_per_step_error_table(
        exact,
        approx,
        reveal_seed=101,
        budget=45,
        max_loo_terms_per_step=20,
    )
    assert set(table["step_name"]) == {"a", "b", "c"}
    assert len(table) == 3
    assert (table["max_loo_terms_per_step"] == 20).all()


def test_summary_by_cap_aggregates_match_rates() -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "max_loo_terms_per_step": 10,
                "top1_match": True,
                "top3_jaccard": 1.0,
                "exact_top_step_in_approx_top3": True,
                "spearman_rank_correlation": 0.9,
                "pearson_score_correlation": 0.8,
                "mean_relative_absolute_error": 0.1,
                "approx_exact_max_score_ratio": 0.95,
            },
            {
                "max_loo_terms_per_step": 10,
                "top1_match": False,
                "top3_jaccard": 0.5,
                "exact_top_step_in_approx_top3": True,
                "spearman_rank_correlation": 0.2,
                "pearson_score_correlation": 0.3,
                "mean_relative_absolute_error": 0.4,
                "approx_exact_max_score_ratio": 1.1,
            },
        ]
    )
    summary = summarize_fragility_approximation_audit(comparison_df)
    assert summary["10"]["n_comparisons"] == 2
    assert summary["10"]["top1_match_rate"] == 0.5
    assert summary["10"]["mean_top3_jaccard"] == 0.75
