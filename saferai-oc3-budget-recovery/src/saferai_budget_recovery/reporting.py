"""Helpers for producing report-oriented tables from experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from saferai_budget_recovery.analysis import DISTANCE_COL


RUN_FILE_SUFFIXES = {
    "results": "repeated_policy_results.csv",
    "auc": "policy_auc_by_seed.csv",
    "differences": "policy_differences_vs_uniform.csv",
    "concentration": "concentration_by_budget.csv",
    "selected_step_counts": "selected_step_counts.csv",
    "summary_by_budget": "policy_summary_by_budget.csv",
    "win_rate_by_budget": "win_rate_by_budget.csv",
    "win_rate_by_seed": "win_rate_by_seed.csv",
    "fragility_diagnostics": "fragility_runtime_diagnostics.csv",
    "report": "repeated_policy_experiment_report.json",
    "c_value_summary": ("exploration_bonus_c_value_summary.csv", "c_value_summary.csv"),
}


def load_experiment_outputs(base_dir: Path, run_names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Load available repeated-experiment outputs by run prefix."""

    base_dir = Path(base_dir)
    if run_names is None:
        run_names = sorted(
            path.name.removesuffix("_repeated_policy_results.csv")
            for path in base_dir.glob("*_repeated_policy_results.csv")
        )
    outputs: dict[str, dict[str, Any]] = {}
    for run_name in run_names:
        run_outputs: dict[str, Any] = {}
        for key, suffixes in RUN_FILE_SUFFIXES.items():
            if isinstance(suffixes, str):
                suffixes = (suffixes,)
            path = next(
                (base_dir / f"{run_name}_{suffix}" for suffix in suffixes if (base_dir / f"{run_name}_{suffix}").exists()),
                None,
            )
            if path is None:
                continue
            if not path.exists():
                continue
            if key == "report":
                run_outputs[key] = json.loads(path.read_text(encoding="utf-8"))
            else:
                run_outputs[key] = pd.read_csv(path)
        if "results" not in run_outputs:
            raise FileNotFoundError(
                f"Missing required repeated-policy results file for run {run_name!r} in {base_dir}"
            )
        outputs[run_name] = run_outputs
    return outputs


def build_main_policy_comparison(
    run_outputs: dict[str, dict[str, Any]],
    policy_sources: list[dict[str, str]],
    max_budget: int = 1200,
    concentration_budgets: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Build one-row-per-policy summary table while preserving source-run labels."""

    if concentration_budgets is None:
        concentration_budgets = (max_budget,)

    rows: list[dict[str, Any]] = []
    for source in policy_sources:
        source_run = source["source_run"]
        policy = source["policy"]
        outputs = _get_run(run_outputs, source_run)
        results = _policy_rows(outputs["results"], policy, source_run)
        auc = _policy_rows(outputs["auc"], policy, source_run)
        differences = outputs.get("differences", pd.DataFrame())
        concentration = outputs.get("concentration", pd.DataFrame())
        report = outputs.get("report", {})
        settings = report.get("settings", {})
        fragility_settings = settings.get("fragility_kwargs", {})

        if policy == "uniform_step_balanced":
            win_fraction = np.nan
        elif "policy_name" in differences.columns:
            diff_rows = differences.loc[differences["policy_name"].eq(policy)]
            win_fraction = float(diff_rows["policy_better"].mean()) if not diff_rows.empty else np.nan
        else:
            win_fraction = np.nan

        uses_approx_loo = bool(policy != "uniform_step_balanced" and fragility_settings.get("max_loo_terms_per_step") is not None)
        rows.append(
            {
                "policy": source.get("display_policy", policy),
                "source_run": source_run,
                "n_reveal_seeds": int(results["reveal_seed"].nunique()),
                "budgets": _format_budgets(results["budget"]),
                "average_auc": float(auc["auc_distance"].mean()),
                "median_auc": float(auc["auc_distance"].median()),
                "average_distance_all_seed_budgets": float(results[DISTANCE_COL].mean()),
                "win_fraction_vs_uniform": win_fraction,
                "uses_approx_loo": uses_approx_loo,
                "max_loo_terms_per_step": (
                    fragility_settings.get("max_loo_terms_per_step") if uses_approx_loo else np.nan
                ),
                "notes": source.get("notes", ""),
            }
        )
        for budget in concentration_budgets:
            at_budget = results.loc[results["budget"].eq(budget)]
            concentration_at_budget = concentration.loc[
                concentration.get("policy_name", pd.Series(dtype=str)).eq(policy)
                & concentration.get("budget", pd.Series(dtype=float)).eq(budget)
            ]
            concentration_row = (
                concentration_at_budget.iloc[0].to_dict() if not concentration_at_budget.empty else {}
            )
            rows[-1].update(
                {
                    f"mean_l1_imbalance_at_{budget}": _float_or_nan(
                        concentration_row.get("mean_step_count_l1_imbalance")
                    ),
                    f"mean_max_min_ratio_at_{budget}": _float_or_nan(
                        concentration_row.get("mean_max_min_revealed_row_ratio")
                    ),
                    f"min_step_count_at_{budget}_mean": _series_mean_or_nan(
                        at_budget.get("min_revealed_rows_per_step")
                    ),
                    f"max_step_count_at_{budget}_mean": _series_mean_or_nan(
                        at_budget.get("max_revealed_rows_per_step")
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_error_by_budget(
    run_outputs: dict[str, dict[str, Any]],
    policy_sources: list[dict[str, str]],
) -> pd.DataFrame:
    """Aggregate squared W2 error by source run, policy, and budget."""

    rows: list[dict[str, Any]] = []
    for source in policy_sources:
        source_run = source["source_run"]
        policy = source["policy"]
        results = _policy_rows(_get_run(run_outputs, source_run)["results"], policy, source_run)
        for budget, group in results.groupby("budget", dropna=False):
            distances = group[DISTANCE_COL].to_numpy(dtype=float)
            rows.append(
                {
                    "policy": source.get("display_policy", policy),
                    "source_run": source_run,
                    "budget": int(budget),
                    "n_reveal_seeds": int(group["reveal_seed"].nunique()),
                    "mean_squared_w2_error": float(np.mean(distances)),
                    "median_squared_w2_error": float(np.median(distances)),
                    "p25_squared_w2_error": float(np.percentile(distances, 25)),
                    "p75_squared_w2_error": float(np.percentile(distances, 75)),
                }
            )
    return pd.DataFrame(rows).sort_values(["policy", "budget"]).reset_index(drop=True)


def build_concentration_by_budget(
    run_outputs: dict[str, dict[str, Any]],
    policy_sources: list[dict[str, str]],
) -> pd.DataFrame:
    """Aggregate step-allocation concentration diagnostics from raw result rows."""

    rows: list[dict[str, Any]] = []
    for source in policy_sources:
        source_run = source["source_run"]
        policy = source["policy"]
        results = _policy_rows(_get_run(run_outputs, source_run)["results"], policy, source_run)
        for budget, group in results.groupby("budget", dropna=False):
            rows.append(
                {
                    "policy": source.get("display_policy", policy),
                    "source_run": source_run,
                    "budget": int(budget),
                    "n_reveal_seeds": int(group["reveal_seed"].nunique()),
                    "mean_l1_imbalance": _series_mean_or_nan(
                        group.get("step_count_l1_from_perfect_balance")
                    ),
                    "median_l1_imbalance": _series_median_or_nan(
                        group.get("step_count_l1_from_perfect_balance")
                    ),
                    "mean_max_min_ratio": _series_mean_or_nan(
                        group.get("max_min_revealed_rows_per_step_ratio")
                    ),
                    "mean_minimum_step_count": _series_mean_or_nan(
                        group.get("min_revealed_rows_per_step")
                    ),
                    "mean_maximum_step_count": _series_mean_or_nan(
                        group.get("max_revealed_rows_per_step")
                    ),
                    "mean_steps_still_at_initial_seed_count": _mean_step_count_condition(
                        group.get("revealed_rows_by_step"), target=5
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["policy", "budget"]).reset_index(drop=True)


def build_exploration_bonus_sensitivity(
    sensitivity_outputs: dict[str, Any],
    *,
    source_run: str | None = None,
    concentration_budget: int = 1200,
) -> pd.DataFrame:
    """Return a clean exploration-bonus sensitivity table."""

    if "c_value_summary" in sensitivity_outputs:
        df = sensitivity_outputs["c_value_summary"].copy()
    else:
        raise ValueError("Sensitivity outputs are missing c_value_summary.")
    if source_run is not None:
        df.insert(0, "source_run", source_run)

    if f"mean_l1_imbalance_at_{concentration_budget}" not in df.columns:
        concentration = sensitivity_outputs.get("concentration", pd.DataFrame())
        if not concentration.empty and "policy_name" in df.columns:
            concentration_rows = concentration.loc[concentration["budget"].eq(concentration_budget)].copy()
            concentration_rows = concentration_rows.rename(
                columns={
                    "mean_step_count_l1_imbalance": f"mean_l1_imbalance_at_{concentration_budget}",
                    "mean_max_min_revealed_row_ratio": f"mean_max_min_ratio_at_{concentration_budget}",
                    "minimum_observed_revealed_row_count_any_step": f"min_step_count_at_{concentration_budget}_mean",
                    "maximum_observed_revealed_row_count_any_step": f"max_step_count_at_{concentration_budget}_mean",
                }
            )
            join_columns = [
                "policy_name",
                f"mean_l1_imbalance_at_{concentration_budget}",
                f"mean_max_min_ratio_at_{concentration_budget}",
                f"min_step_count_at_{concentration_budget}_mean",
                f"max_step_count_at_{concentration_budget}_mean",
            ]
            available_columns = [column for column in join_columns if column in concentration_rows.columns]
            df = df.merge(concentration_rows[available_columns], on="policy_name", how="left")

    prefix_columns = ["source_run", "policy_name"] if source_run is not None else ["policy_name"]
    preferred_columns = [
        *[column for column in prefix_columns if column in df.columns],
        "c",
        "average_auc",
        "median_auc",
        "win_fraction_vs_uniform",
        f"mean_l1_imbalance_at_{concentration_budget}",
        f"mean_max_min_ratio_at_{concentration_budget}",
        f"min_step_count_at_{concentration_budget}_mean",
        f"max_step_count_at_{concentration_budget}_mean",
    ]
    for column in [
        "mean_l1_imbalance_at_1798",
        "median_l1_imbalance_at_1798",
        "max_l1_imbalance_at_1798",
        "mean_max_min_ratio_at_1798",
    ]:
        if column in df.columns:
            preferred_columns.append(column)
    missing = [column for column in preferred_columns if column not in df.columns]
    if missing:
        raise ValueError(f"c-value summary is missing required columns: {missing}")
    return df[preferred_columns].sort_values("c").reset_index(drop=True)


def write_markdown_table(df: pd.DataFrame, path: Path, float_digits: int = 6) -> None:
    """Write a small GitHub-flavored Markdown table without requiring tabulate."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = dataframe_to_markdown(df, float_digits=float_digits)
    path.write_text(markdown + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, float_digits: int = 6) -> str:
    """Convert a DataFrame to a simple Markdown table."""

    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(lambda value: _format_markdown_cell(value, float_digits))
    columns = [str(column) for column in formatted.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in formatted.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in formatted.columns) + " |")
    return "\n".join(rows)


def _get_run(run_outputs: dict[str, dict[str, Any]], source_run: str) -> dict[str, Any]:
    if source_run not in run_outputs:
        raise KeyError(f"Run outputs do not include {source_run!r}.")
    return run_outputs[source_run]


def _policy_rows(df: pd.DataFrame, policy: str, source_run: str) -> pd.DataFrame:
    if "policy_name" not in df.columns:
        raise ValueError(f"{source_run} output is missing policy_name.")
    rows = df.loc[df["policy_name"].eq(policy)].copy()
    if rows.empty:
        raise ValueError(f"Policy {policy!r} was not found in source run {source_run!r}.")
    return rows


def _format_budgets(series: pd.Series) -> str:
    budgets = sorted(int(value) for value in pd.Series(series).dropna().unique())
    return json.dumps(budgets)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return np.nan
    return float(value)


def _series_mean_or_nan(series: pd.Series | None) -> float:
    if series is None:
        return np.nan
    return float(pd.Series(series).mean())


def _series_median_or_nan(series: pd.Series | None) -> float:
    if series is None:
        return np.nan
    return float(pd.Series(series).median())


def _mean_step_count_condition(series: pd.Series | None, target: int) -> float:
    if series is None:
        return np.nan
    counts = []
    for raw in series.dropna():
        parsed = json.loads(raw)
        counts.append(sum(1 for value in parsed.values() if int(value) == target))
    return float(np.mean(counts)) if counts else np.nan


def _format_markdown_cell(value: Any, float_digits: int) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001:
            return f"{value:.{float_digits}e}"
        return f"{value:.{float_digits}f}".rstrip("0").rstrip(".")
    if isinstance(value, np.floating):
        return _format_markdown_cell(float(value), float_digits)
    if isinstance(value, np.integer):
        return str(int(value))
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
