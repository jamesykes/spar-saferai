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
    choose_step_epsilon_greedy_fragility,
    choose_step_exploration_bonus_fragility,
    choose_step_greedy_fragility,
    choose_step_stochastic_epsilon_greedy_fragility,
    choose_step_stochastic_exploration_bonus_fragility,
    choose_step_stochastic_normalized_fragility,
    choose_step_uniform_row_random,
    choose_step_uniform_step_balanced,
    choose_step_uniform_positive_fragility,
)
from saferai_budget_recovery.reveal import (
    FITTED_ROW_UID_COLUMN,
    initial_revealed_from_hidden_orders,
    make_hidden_reveal_orders,
    reveal_next_for_step,
    unrevealed_steps_available,
    usable_fit_rows,
)
from saferai_budget_recovery.sampling import sample_full_reference_p_success, sample_p_success_from_revealed


VALID_POLICIES = {
    "uniform_step_balanced",
    "greedy_loo_fragility",
    "epsilon_greedy_loo_fragility",
    "exploration_bonus_loo_fragility",
    "stochastic_normalized_loo_fragility",
    "stochastic_epsilon_greedy_loo_fragility",
    "stochastic_exploration_bonus_loo_fragility",
    "uniform_positive_loo_fragility",
    "uniform_row_random",
}


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
    policy_kwargs: dict | None = None,
    policy_label: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run a small recovery loop for one policy and one reveal seed."""

    if policy_name not in VALID_POLICIES:
        raise ValueError(f"Unknown policy_name={policy_name!r}; expected one of {sorted(VALID_POLICIES)}.")
    if fragility_recompute_every <= 0:
        raise ValueError("fragility_recompute_every must be positive.")
    policy_kwargs = policy_kwargs or {}
    output_policy_name = policy_label or policy_name

    start = time.perf_counter()
    usable_df = usable_fit_rows(fit_df)
    hidden_orders = make_hidden_reveal_orders(usable_df, reveal_seed=reveal_seed, strict=True)
    initial_df = initial_revealed_from_hidden_orders(usable_df, hidden_orders)
    revealed_row_ids_ordered = [str(row_id) for row_id in hidden_orders.initial_row_ids]
    revealed_row_id_set = set(revealed_row_ids_ordered)
    revealed_df = initial_df.copy().reset_index(drop=True)
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
    selected_step_counts_by_decision_type: dict[str, dict[str, int]] = {}
    decision_type_counts: dict[str, int] = {}
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
            available_steps = unrevealed_steps_available(hidden_orders, revealed_row_id_set)
            if not available_steps:
                raise ValueError(
                    "No hidden reveal-order steps have unrevealed rows before target budget "
                    f"{target_budget}."
                )
            if policy_name == "uniform_row_random":
                selected_step = choose_step_uniform_row_random(
                    available_steps,
                    _available_row_counts_by_step(hidden_orders, revealed_row_id_set),
                    rng,
                )
            elif policy_name == "uniform_step_balanced":
                selected_step = choose_step_uniform_step_balanced(revealed_df, available_steps, rng)
            else:
                recompute = cached_fragility_scores is None or decisions_since_recompute >= fragility_recompute_every
                if recompute:
                    fragility_start = time.perf_counter()
                    selected_step, cached_fragility_scores = _choose_fragility_policy(
                        policy_name=policy_name,
                        revealed_df=revealed_df,
                        available_steps=available_steps,
                        rng=rng,
                        fragility_kwargs=fragility_kwargs or {},
                        policy_kwargs=policy_kwargs,
                        fragility_scores=None,
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
                            "policy_name": output_policy_name,
                            "base_policy_name": policy_name,
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
                            "policy_kwargs": json.dumps(policy_kwargs, sort_keys=True),
                        }
                    )
                    decisions_since_recompute = 0
                else:
                    selected_step, cached_fragility_scores = _choose_fragility_policy(
                        policy_name=policy_name,
                        revealed_df=revealed_df,
                        available_steps=available_steps,
                        rng=rng,
                        fragility_kwargs=fragility_kwargs or {},
                        policy_kwargs=policy_kwargs,
                        fragility_scores=cached_fragility_scores,
                    )
                decisions_since_recompute += 1
                if cached_fragility_scores.attrs.get("selection_fallback") is not None:
                    fallback_count += 1
                selected_step = cached_fragility_scores.attrs.get("selected_step")
                decision_type = str(cached_fragility_scores.attrs.get("decision_type", "unknown"))
                decision_type_counts[decision_type] = decision_type_counts.get(decision_type, 0) + 1
                if selected_step is not None:
                    selected_step_counts[str(selected_step)] = selected_step_counts.get(str(selected_step), 0) + 1
                    by_type = selected_step_counts_by_decision_type.setdefault(decision_type, {})
                    by_type[str(selected_step)] = by_type.get(str(selected_step), 0) + 1

            selected_row_id = reveal_next_for_step(
                usable_df,
                hidden_orders=hidden_orders,
                revealed_row_ids=revealed_row_id_set,
                step_name=str(selected_step),
            )
            selected_row = usable_df.loc[usable_df[FITTED_ROW_UID_COLUMN].eq(selected_row_id)]
            if len(selected_row) != 1:
                raise ValueError(f"Selected hidden fitted-row ID not found exactly once in usable rows: {selected_row_id}")
            revealed_row_ids_ordered.append(str(selected_row_id))
            revealed_row_id_set.add(str(selected_row_id))
            revealed_df = pd.concat([revealed_df, selected_row], ignore_index=True)

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
                "policy_name": output_policy_name,
                "base_policy_name": policy_name,
                "policy_kwargs": json.dumps(policy_kwargs, sort_keys=True),
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
        "policy_name": output_policy_name,
        "base_policy_name": policy_name,
        "policy_kwargs": policy_kwargs,
        "reveal_seed": int(reveal_seed),
        "runtime_seconds": elapsed,
        "initial_seed_size": int(initial_size),
        "total_usable_fitted_rows": int(total_usable),
        "budgets_used": budgets_used,
        "uses_hidden_reveal_orders": True,
        "hidden_reveal_order_metadata": _compact_hidden_reveal_metadata(hidden_orders.metadata),
        "hidden_reveal_order_post_seed_lengths_by_step": hidden_orders.metadata.get(
            "post_seed_order_lengths_by_step", {}
        ),
        "hidden_reveal_order_coverage": hidden_orders.metadata.get("coverage", {}),
        "reference_seed": int(reference_seed),
        "sample_seed_base": int(sample_seed_base),
        "n_reference_samples": int(n_reference_samples),
        "n_budget_samples": int(n_budget_samples),
        "n_grid": int(n_grid),
        "fragility_kwargs": fragility_kwargs or {},
        "fragility_recompute_every": int(fragility_recompute_every),
        "selected_step_counts": selected_step_counts,
        "selected_step_counts_by_decision_type": selected_step_counts_by_decision_type,
        "decision_type_counts": decision_type_counts,
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


def _choose_fragility_policy(
    policy_name: str,
    revealed_df: pd.DataFrame,
    available_steps: set[str],
    rng: np.random.Generator,
    fragility_kwargs: dict,
    policy_kwargs: dict,
    fragility_scores: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame]:
    if fragility_scores is None:
        from saferai_budget_recovery.fragility import compute_loo_fragility_scores

        fragility_scores = compute_loo_fragility_scores(revealed_df, **fragility_kwargs)
    else:
        fragility_scores = fragility_scores.copy()

    if policy_name == "greedy_loo_fragility":
        selected_step = choose_step_greedy_fragility(
            revealed_df,
            available_steps,
            rng,
            fragility_scores=fragility_scores,
        )
        return selected_step, fragility_scores
    if policy_name == "stochastic_normalized_loo_fragility":
        selected_step = choose_step_stochastic_normalized_fragility(
            revealed_df,
            available_steps,
            rng,
            fragility_scores=fragility_scores,
        )
        return selected_step, fragility_scores
    if policy_name == "uniform_positive_loo_fragility":
        selected_step = choose_step_uniform_positive_fragility(
            revealed_df,
            available_steps,
            rng,
            fragility_scores=fragility_scores,
        )
        return selected_step, fragility_scores
    if policy_name == "epsilon_greedy_loo_fragility":
        selected_step = choose_step_epsilon_greedy_fragility(
            revealed_df,
            available_steps,
            rng,
            epsilon=float(policy_kwargs.get("epsilon", 0.2)),
            fragility_scores=fragility_scores,
        )
        return selected_step, fragility_scores
    if policy_name == "stochastic_epsilon_greedy_loo_fragility":
        selected_step = choose_step_stochastic_epsilon_greedy_fragility(
            revealed_df,
            available_steps,
            rng,
            epsilon=float(policy_kwargs.get("epsilon", 0.2)),
            fragility_scores=fragility_scores,
        )
        return selected_step, fragility_scores
    if policy_name == "exploration_bonus_loo_fragility":
        selected_step = choose_step_exploration_bonus_fragility(
            revealed_df,
            available_steps,
            rng,
            c=float(policy_kwargs.get("c", 0.5)),
            fragility_scores=fragility_scores,
        )
        return selected_step, fragility_scores
    if policy_name == "stochastic_exploration_bonus_loo_fragility":
        selected_step = choose_step_stochastic_exploration_bonus_fragility(
            revealed_df,
            available_steps,
            rng,
            c=float(policy_kwargs.get("c", 0.5)),
            fragility_scores=fragility_scores,
        )
        return selected_step, fragility_scores
    raise ValueError(f"Policy {policy_name!r} is not a fragility policy.")


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


def _available_row_counts_by_step(hidden_orders, revealed_row_ids: set[str]) -> dict[str, int]:
    return {
        step: int(sum(row_id not in revealed_row_ids for row_id in row_ids))
        for step, row_ids in hidden_orders.orders_by_step.items()
    }


def _compact_hidden_reveal_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep run diagnostics compact while retaining protocol/coverage evidence."""

    out = dict(metadata)
    # The full model sequence is useful for smoke audits, but repeating it in
    # every policy x seed diagnostic bloats locked experiment reports.
    out.pop("post_seed_model_sequence_by_step", None)
    return out
