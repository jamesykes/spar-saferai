"""Small policy-recovery experiment primitives."""

from __future__ import annotations

import time
import json
from typing import Any

import numpy as np
import pandas as pd

from saferai_budget_recovery import config
from saferai_budget_recovery.distances import (
    empirical_quantiles,
    quantile_grid,
    squared_wasserstein2_from_quantiles,
)
from saferai_budget_recovery.policies import (
    choose_next_greedy_fragility,
    choose_next_uniform_row_random,
    choose_next_uniform_step_balanced,
)
from saferai_budget_recovery.reveal import (
    FITTED_ROW_UID_COLUMN,
    make_initial_seed_reveal,
    split_revealed_unrevealed,
    usable_fit_rows,
)
from saferai_budget_recovery.sampling import sample_full_reference_p_success, sample_p_success_from_revealed


VALID_POLICIES = {"uniform_step_balanced", "greedy_loo_fragility", "uniform_row_random"}


def run_policy_recovery(
    fit_df: pd.DataFrame,
    policy_name: str,
    budgets: list[int],
    reveal_seed: int,
    reference_seed: int,
    sample_seed_base: int,
    n_reference_samples: int = 50000,
    n_budget_samples: int = 20000,
    n_grid: int = 501,
    fragility_kwargs: dict | None = None,
    fragility_recompute_every: int = 1,
    reference_samples: np.ndarray | None = None,
    reference_quantiles: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run a small recovery loop for one policy and one reveal seed."""

    if policy_name not in VALID_POLICIES:
        raise ValueError(f"Unknown policy_name={policy_name!r}; expected one of {sorted(VALID_POLICIES)}.")
    if fragility_recompute_every <= 0:
        raise ValueError("fragility_recompute_every must be positive.")

    start = time.perf_counter()
    usable_df = usable_fit_rows(fit_df)
    initial_df = make_initial_seed_reveal(usable_df, seed=reveal_seed, strict=True)
    revealed_df, unrevealed_df = split_revealed_unrevealed(usable_df, initial_df)
    initial_size = len(revealed_df)
    total_usable = len(usable_df)
    budgets_used = sorted(set(int(budget) for budget in budgets if budget <= total_usable))
    if not budgets_used:
        raise ValueError("No requested budgets are <= total usable fitted rows.")
    if any(budget < initial_size for budget in budgets_used):
        raise ValueError(
            f"All budgets must be at least the initial seed allocation size {initial_size}."
        )

    grid = quantile_grid(n_grid)
    reference_sampling_start = time.perf_counter()
    reference_reused = reference_samples is not None or reference_quantiles is not None
    if reference_samples is None:
        reference_df = sample_full_reference_p_success(
            usable_df, n_samples=n_reference_samples, seed=reference_seed
        )
        reference_samples = reference_df["p_success"].to_numpy(dtype=float)
    else:
        reference_samples = np.asarray(reference_samples, dtype=float)
    if reference_quantiles is None:
        reference_quantiles = empirical_quantiles(reference_samples, grid)
    else:
        reference_quantiles = np.asarray(reference_quantiles, dtype=float)
    reference_sampling_seconds = time.perf_counter() - reference_sampling_start

    rng = np.random.default_rng(reveal_seed)
    selected_step_counts: dict[str, int] = {}
    fallback_count = 0
    decision_count = 0
    cached_fragility_scores: pd.DataFrame | None = None
    decisions_since_recompute = fragility_recompute_every
    fragility_runtime_seconds = 0.0
    budget_sampling_seconds = 0.0
    fragility_recomputation_count = 0
    total_loo_terms_available = 0
    total_loo_terms_used = 0
    any_loo_subsampled = False
    fragility_runtime_diagnostics: list[dict[str, Any]] = []

    rows: list[dict[str, Any]] = []
    for target_budget in budgets_used:
        while len(revealed_df) < target_budget:
            decision_count += 1
            if policy_name == "uniform_row_random":
                selected_row_id = choose_next_uniform_row_random(unrevealed_df, rng)
            elif policy_name == "uniform_step_balanced":
                selected_row_id = choose_next_uniform_step_balanced(revealed_df, unrevealed_df, rng)
            else:
                recompute = cached_fragility_scores is None or decisions_since_recompute >= fragility_recompute_every
                if recompute:
                    fragility_start = time.perf_counter()
                    selected_row_id, cached_fragility_scores = choose_next_greedy_fragility(
                        revealed_df,
                        unrevealed_df,
                        rng,
                        fragility_kwargs=fragility_kwargs or {},
                    )
                    fragility_elapsed = time.perf_counter() - fragility_start
                    fragility_runtime_seconds += fragility_elapsed
                    fragility_recomputation_count += 1
                    available = int(cached_fragility_scores["n_loo_terms_available"].sum())
                    used = int(cached_fragility_scores["n_loo_terms_used"].sum())
                    total_loo_terms_available += available
                    total_loo_terms_used += used
                    any_loo_subsampled = any_loo_subsampled or bool(cached_fragility_scores["loo_subsampled"].any())
                    finite_scores = cached_fragility_scores.loc[np.isfinite(cached_fragility_scores["loo_fragility"])]
                    if finite_scores.empty:
                        top_fragile_step = None
                        top_fragility = None
                    else:
                        top_row = finite_scores.sort_values("loo_fragility", ascending=False).iloc[0]
                        top_fragile_step = str(top_row["step_name"])
                        top_fragility = float(top_row["loo_fragility"])
                    fragility_runtime_diagnostics.append(
                        {
                            "policy_name": policy_name,
                            "reveal_seed": int(reveal_seed),
                            "recomputation_index": int(fragility_recomputation_count),
                            "revealed_count_at_recomputation": int(len(revealed_df)),
                            "runtime_seconds": float(fragility_elapsed),
                            "total_loo_terms_available": available,
                            "total_loo_terms_used": used,
                            "max_loo_terms_per_step": (fragility_kwargs or {}).get("max_loo_terms_per_step"),
                            "loo_subsampled": bool(cached_fragility_scores["loo_subsampled"].any()),
                            "top_fragile_step": top_fragile_step,
                            "top_fragility": top_fragility,
                        }
                    )
                    decisions_since_recompute = 0
                else:
                    selected_row_id, cached_fragility_scores = choose_next_greedy_fragility(
                        revealed_df,
                        unrevealed_df,
                        rng,
                        fragility_scores=cached_fragility_scores,
                    )
                decisions_since_recompute += 1
                if cached_fragility_scores.attrs.get("selection_fallback") is not None:
                    fallback_count += 1
                selected_step = cached_fragility_scores.attrs.get("selected_step")
                if selected_step is not None:
                    selected_step_counts[str(selected_step)] = selected_step_counts.get(str(selected_step), 0) + 1

            selected_row = unrevealed_df.loc[unrevealed_df[FITTED_ROW_UID_COLUMN].eq(selected_row_id)]
            if len(selected_row) != 1:
                raise ValueError(f"Selected fitted-row ID not found exactly once in unrevealed rows: {selected_row_id}")
            revealed_df = pd.concat([revealed_df, selected_row], ignore_index=True)
            unrevealed_df = unrevealed_df.loc[
                ~unrevealed_df[FITTED_ROW_UID_COLUMN].eq(selected_row_id)
            ].reset_index(drop=True)

        sample_seed = int(sample_seed_base + reveal_seed * 10_000 + target_budget)
        if len(revealed_df) == total_usable:
            budget_samples = reference_samples
            budget_quantiles = reference_quantiles
        else:
            budget_sampling_start = time.perf_counter()
            budget_df = sample_p_success_from_revealed(
                revealed_df, n_samples=n_budget_samples, seed=sample_seed
            )
            budget_samples = budget_df["p_success"].to_numpy(dtype=float)
            budget_quantiles = empirical_quantiles(budget_samples, grid)
            budget_sampling_seconds += time.perf_counter() - budget_sampling_start
        distance = squared_wasserstein2_from_quantiles(budget_quantiles, reference_quantiles, grid)
        step_counts = _step_counts(revealed_df)
        balance = _step_balance_diagnostics(step_counts, target_budget)
        summary = _p_success_summary(budget_samples)
        rows.append(
            {
                "policy_name": policy_name,
                "reveal_seed": int(reveal_seed),
                "budget": int(target_budget),
                "additional_rows_beyond_initial_seed": int(target_budget - initial_size),
                "squared_wasserstein2_to_full_reference": float(distance),
                "p_success_mean": summary["mean"],
                "p_success_p05": summary["p05"],
                "p_success_p50": summary["p50"],
                "p_success_p95": summary["p95"],
                "min_revealed_rows_per_step": int(min(step_counts.values())),
                "max_revealed_rows_per_step": int(max(step_counts.values())),
                "max_min_revealed_rows_per_step_ratio": float(max(step_counts.values()) / min(step_counts.values())),
                "step_count_std": balance["step_count_std"],
                "step_count_l1_from_perfect_balance": balance["step_count_l1_from_perfect_balance"],
                "revealed_rows_by_step": json.dumps(step_counts, sort_keys=True),
                "sample_seed": sample_seed,
                "decision_count": int(decision_count),
            }
        )

    elapsed = time.perf_counter() - start
    result_df = pd.DataFrame(rows)
    diagnostics = {
        "policy_name": policy_name,
        "reveal_seed": int(reveal_seed),
        "runtime_seconds": elapsed,
        "initial_seed_size": int(initial_size),
        "total_usable_fitted_rows": int(total_usable),
        "budgets_used": budgets_used,
        "reference_seed": int(reference_seed),
        "sample_seed_base": int(sample_seed_base),
        "n_reference_samples": int(n_reference_samples),
        "n_budget_samples": int(n_budget_samples),
        "n_grid": int(n_grid),
        "fragility_kwargs": fragility_kwargs or {},
        "fragility_recompute_every": int(fragility_recompute_every),
        "selected_step_counts": selected_step_counts,
        "fallback_count": int(fallback_count),
        "decision_count": int(decision_count),
        "reference_sampling_seconds": float(reference_sampling_seconds),
        "reference_reused": bool(reference_reused),
        "budget_sampling_seconds": float(budget_sampling_seconds),
        "fragility_runtime_seconds": float(fragility_runtime_seconds),
        "fragility_recomputation_count": int(fragility_recomputation_count),
        "avg_fragility_recompute_seconds": (
            float(fragility_runtime_seconds / fragility_recomputation_count)
            if fragility_recomputation_count
            else 0.0
        ),
        "total_loo_terms_available": int(total_loo_terms_available),
        "total_loo_terms_used": int(total_loo_terms_used),
        "loo_terms_used_fraction": (
            float(total_loo_terms_used / total_loo_terms_available)
            if total_loo_terms_available
            else None
        ),
        "any_loo_subsampled": bool(any_loo_subsampled),
        "fragility_runtime_diagnostics": fragility_runtime_diagnostics,
    }
    return result_df, diagnostics


def _p_success_summary(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "p05": float(np.percentile(samples, 5)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
    }


def _step_counts(revealed_df: pd.DataFrame) -> dict[str, int]:
    observed = revealed_df.groupby("step_name").size().to_dict()
    steps = list(config.EXPECTED_MITRE_STEP_LABELS)
    for step in observed:
        if step not in steps:
            steps.append(str(step))
    return {step: int(observed.get(step, 0)) for step in steps}


def _step_balance_diagnostics(step_counts: dict[str, int], budget: int) -> dict[str, float]:
    counts = np.asarray(list(step_counts.values()), dtype=float)
    n_steps = len(counts)
    base = budget // n_steps
    remainder = budget % n_steps
    target = np.full(n_steps, base, dtype=float)
    # Assign the +1 target counts to the largest observed counts. This is a valid
    # balanced target vector and gives the minimum L1 distance among such vectors.
    if remainder:
        largest_count_indices = np.argsort(-counts)[:remainder]
        target[largest_count_indices] += 1
    return {
        "step_count_std": float(np.std(counts)),
        "step_count_l1_from_perfect_balance": float(np.sum(np.abs(counts - target))),
    }
