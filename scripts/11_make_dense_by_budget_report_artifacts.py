"""Create by-budget report tables and plots for the three dense policy experiments."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "saferai_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saferai_budget_recovery import config


RESULTS_DIR = config.PROJECT_ROOT / "outputs" / "repeated_policy_experiment"
ARTIFACT_DIR = config.PROJECT_ROOT / "outputs" / "report_artifacts_dense_by_budget"
TABLE_DIR = config.PROJECT_ROOT / "report" / "tables" / "dense_by_budget"
FIGURE_DIR = config.PROJECT_ROOT / "report" / "figures" / "dense_by_budget"

ERROR_COL = "squared_wasserstein2_to_full_reference"
UNIFORM_POLICY = "uniform_step_balanced"

EXPERIMENTS = [
    {
        "key": "dense_policy_suite",
        "title": "Dense policy suite",
        "prefix": "dense_policy_suite_30_seeds_recompute10_nsamples200",
        "policy_order": [
            "uniform_step_balanced",
            "stochastic_normalized_fragility",
            "epsilon_greedy_eps0.2",
            "exploration_bonus_c1.0",
            "greedy_loo_fragility",
            "exploration_bonus_c0.5",
            "exploration_bonus_c0.25",
        ],
    },
    {
        "key": "stochastic_ablation",
        "title": "Stochastic fragility ablation",
        "prefix": "dense_stochastic_ablation_30_seeds_recompute10_nsamples200",
        "policy_order": [
            "uniform_step_balanced",
            "stochastic_epsilon_greedy_eps0.2",
            "stochastic_exploration_bonus_c1.0",
            "uniform_positive_fragility",
            "stochastic_exploration_bonus_c0.5",
            "stochastic_exploration_bonus_c0.25",
            "stochastic_normalized_fragility",
        ],
    },
    {
        "key": "softmax_temperature_ablation",
        "title": "Softmax temperature ablation",
        "prefix": "dense_softmax_temperature_ablation_30_seeds_recompute10_nsamples200",
        "policy_order": [
            "uniform_step_balanced",
            "uniform_positive_fragility",
            "stochastic_normalized_fragility",
            "softmax_normalized_fragility_temp0.25",
            "softmax_normalized_fragility_temp0.5",
            "softmax_normalized_fragility_temp1.0",
            "softmax_normalized_fragility_temp2.0",
            "softmax_normalized_fragility_temp4.0",
        ],
    },
]

POLICY_LABELS = {
    "uniform_step_balanced": "Uniform",
    "greedy_loo_fragility": "Greedy LOO",
    "epsilon_greedy_eps0.2": "Eps-greedy 0.2",
    "exploration_bonus_c0.25": "Bonus c=0.25",
    "exploration_bonus_c0.5": "Bonus c=0.5",
    "exploration_bonus_c1.0": "Bonus c=1.0",
    "stochastic_normalized_fragility": "Stoch proportional",
    "uniform_positive_fragility": "Uniform positive",
    "stochastic_epsilon_greedy_eps0.2": "Stoch eps-greedy 0.2",
    "stochastic_exploration_bonus_c0.25": "Stoch bonus c=0.25",
    "stochastic_exploration_bonus_c0.5": "Stoch bonus c=0.5",
    "stochastic_exploration_bonus_c1.0": "Stoch bonus c=1.0",
    "softmax_normalized_fragility_temp0.25": "Softmax T=0.25",
    "softmax_normalized_fragility_temp0.5": "Softmax T=0.5",
    "softmax_normalized_fragility_temp1.0": "Softmax T=1.0",
    "softmax_normalized_fragility_temp2.0": "Softmax T=2.0",
    "softmax_normalized_fragility_temp4.0": "Softmax T=4.0",
}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        metrics = _build_metrics_for_experiment(experiment)
        all_summary_rows.append(_write_experiment_artifacts(experiment, metrics))

    summary = pd.DataFrame(all_summary_rows)
    summary_path = ARTIFACT_DIR / "artifact_summary.csv"
    summary.to_csv(summary_path, index=False)
    readme_path = ARTIFACT_DIR / "README.md"
    readme_path.write_text(_build_readme(), encoding="utf-8")

    print("Dense by-budget report artifacts generated")
    print(f"Artifact directory: {ARTIFACT_DIR}")
    print(f"Table directory: {TABLE_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print(summary.to_string(index=False))
    print(f"README: {readme_path}")


def _build_metrics_for_experiment(experiment: dict[str, object]) -> pd.DataFrame:
    prefix = str(experiment["prefix"])
    policy_order = list(experiment["policy_order"])
    results_path = RESULTS_DIR / f"{prefix}_repeated_policy_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing dense results file: {results_path}")

    results = pd.read_csv(results_path)
    required = {"policy_name", "reveal_seed", "budget", ERROR_COL}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"{results_path} is missing columns: {missing}")

    present_policies = list(results["policy_name"].drop_duplicates())
    missing_policies = [policy for policy in policy_order if policy not in present_policies]
    if missing_policies:
        raise ValueError(f"{results_path} is missing expected policies: {missing_policies}")

    uniform = (
        results.loc[results["policy_name"].eq(UNIFORM_POLICY), ["reveal_seed", "budget", ERROR_COL]]
        .rename(columns={ERROR_COL: "uniform_error"})
        .copy()
    )
    if uniform.empty:
        raise ValueError(f"{results_path} has no {UNIFORM_POLICY} rows.")

    paired = results.merge(uniform, on=["reveal_seed", "budget"], how="left", validate="many_to_one")
    if paired["uniform_error"].isna().any():
        raise ValueError(f"{results_path} has rows without paired uniform errors.")

    paired["relative_error_vs_uniform"] = np.nan
    positive_uniform = paired["uniform_error"].to_numpy(dtype=float) > 0.0
    paired.loc[positive_uniform, "relative_error_vs_uniform"] = (
        paired.loc[positive_uniform, ERROR_COL].to_numpy(dtype=float)
        / paired.loc[positive_uniform, "uniform_error"].to_numpy(dtype=float)
    )
    paired["wins_vs_uniform"] = (paired[ERROR_COL] < paired["uniform_error"]).astype(float)
    paired.loc[paired["policy_name"].eq(UNIFORM_POLICY), "wins_vs_uniform"] = np.nan

    grouped = paired.groupby(["policy_name", "budget"], dropna=False)
    metrics = grouped.agg(
        n_seeds=("reveal_seed", "nunique"),
        mean_error=(ERROR_COL, "mean"),
        median_error=(ERROR_COL, "median"),
        error_p25=(ERROR_COL, lambda x: float(np.quantile(x, 0.25))),
        error_p75=(ERROR_COL, lambda x: float(np.quantile(x, 0.75))),
        policy_wins_vs_uniform=("wins_vs_uniform", lambda x: int(np.nansum(x.to_numpy(dtype=float)))),
        win_rate_vs_uniform=("wins_vs_uniform", "mean"),
        mean_relative_error_vs_uniform=("relative_error_vs_uniform", "mean"),
        median_relative_error_vs_uniform=("relative_error_vs_uniform", "median"),
        relative_error_p25_vs_uniform=("relative_error_vs_uniform", lambda x: _nanquantile_or_nan(x, 0.25)),
        relative_error_p75_vs_uniform=("relative_error_vs_uniform", lambda x: _nanquantile_or_nan(x, 0.75)),
    ).reset_index()

    metrics.loc[metrics["policy_name"].eq(UNIFORM_POLICY), "policy_wins_vs_uniform"] = pd.NA
    metrics.loc[metrics["policy_name"].eq(UNIFORM_POLICY), "win_rate_vs_uniform"] = pd.NA
    metrics["policy_label"] = metrics["policy_name"].map(lambda p: POLICY_LABELS.get(p, p))
    metrics["policy_order"] = metrics["policy_name"].map({policy: i for i, policy in enumerate(policy_order)})
    metrics = metrics.sort_values(["policy_order", "budget"]).reset_index(drop=True)
    return metrics


def _build_readme() -> str:
    return """# Dense By-Budget Report Artifacts

These files summarize the three dense 30-seed policy experiments by budget.

For each experiment, the generated metrics table contains:

- `mean_error`: mean squared Wasserstein-2 error to the full-data reference across reveal seeds.
- `median_error`: median squared Wasserstein-2 error across reveal seeds.
- `win_rate_vs_uniform`: strict paired win rate against `uniform_step_balanced` at the same reveal seed and budget.
- `mean_relative_error_vs_uniform`: mean of seedwise ratios `policy_error / uniform_error`.
- `median_relative_error_vs_uniform`: median of seedwise ratios `policy_error / uniform_error`.

Relative error is computed within reveal seed before averaging or taking medians. This preserves the paired reveal-order design.

At the terminal full budget, all policies have revealed the full available fitted SOTA dataset, so errors are zero and relative error is undefined. The tables keep this as missing. Strict win rate is zero at this terminal tie in the tables; the terminal point is omitted from win-rate plots to avoid visually suggesting late-budget deterioration.
"""


def _write_experiment_artifacts(experiment: dict[str, object], metrics: pd.DataFrame) -> dict[str, object]:
    key = str(experiment["key"])
    title = str(experiment["title"])
    policy_order = list(experiment["policy_order"])

    csv_path = ARTIFACT_DIR / f"{key}_by_budget_metrics.csv"
    md_path = ARTIFACT_DIR / f"{key}_by_budget_metrics.md"
    tex_path = TABLE_DIR / f"{key}_by_budget_metrics.tex"

    metrics.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown(metrics), encoding="utf-8")
    tex_path.write_text(_to_latex_longtable(title, key, metrics), encoding="utf-8")

    figure_paths = _write_plots(key, title, policy_order, metrics)
    return {
        "experiment": key,
        "metrics_csv": str(csv_path),
        "metrics_markdown": str(md_path),
        "metrics_latex": str(tex_path),
        "n_rows": int(len(metrics)),
        "n_policies": int(metrics["policy_name"].nunique()),
        "n_budgets": int(metrics["budget"].nunique()),
        "n_figures": int(len(figure_paths)),
    }


def _write_plots(
    key: str,
    title: str,
    policy_order: list[str],
    metrics: pd.DataFrame,
) -> list[Path]:
    plot_specs = [
        ("mean_error", "mean_error", "Mean squared W2 error", "Mean Error by Budget", False),
        ("median_error", "median_error", "Median squared W2 error", "Median Error by Budget", False),
        ("win_rate_vs_uniform", "win_rate_vs_uniform", "Win rate vs uniform", "Win Rate vs Uniform by Budget", True),
        (
            "mean_relative_error_vs_uniform",
            "mean_relative_error_vs_uniform",
            "Mean seedwise relative error vs uniform",
            "Mean Relative Error vs Uniform by Budget",
            True,
        ),
        (
            "median_relative_error_vs_uniform",
            "median_relative_error_vs_uniform",
            "Median seedwise relative error vs uniform",
            "Median Relative Error vs Uniform by Budget",
            True,
        ),
    ]

    paths: list[Path] = []
    full_tie_budgets = set(
        metrics.groupby("budget")
        .filter(lambda group: bool(np.allclose(group["mean_error"].to_numpy(dtype=float), 0.0)))
        ["budget"]
        .unique()
    )
    for stem, column, ylabel, subtitle, baseline_refs in plot_specs:
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        for policy in policy_order:
            group = metrics.loc[metrics["policy_name"].eq(policy)].sort_values("budget")
            if group.empty:
                continue
            if column == "win_rate_vs_uniform" and full_tie_budgets:
                group = group.loc[~group["budget"].isin(full_tie_budgets)]
            y = group[column].to_numpy(dtype=float)
            if np.all(np.isnan(y)):
                continue
            ax.plot(
                group["budget"].to_numpy(dtype=float),
                y,
                marker="o",
                markersize=3.2,
                linewidth=1.7,
                label=POLICY_LABELS.get(policy, policy),
            )
        if column in {"mean_relative_error_vs_uniform", "median_relative_error_vs_uniform"}:
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        if column == "win_rate_vs_uniform":
            ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
            ax.set_ylim(-0.03, 1.03)
            if full_tie_budgets:
                ax.text(
                    0.99,
                    0.03,
                    "Terminal full-budget tie omitted",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xlabel("Total revealed SOTA elicitation rows")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {subtitle}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.0, ncol=2)
        if baseline_refs and column != "win_rate_vs_uniform":
            ax.text(
                0.99,
                0.97,
                "Dashed line = uniform",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        path = FIGURE_DIR / f"{key}_{stem}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def _to_markdown(metrics: pd.DataFrame) -> str:
    cols = [
        "policy_name",
        "budget",
        "n_seeds",
        "mean_error",
        "median_error",
        "win_rate_vs_uniform",
        "mean_relative_error_vs_uniform",
        "median_relative_error_vs_uniform",
    ]
    display = metrics.loc[:, cols].copy()
    for col in ["mean_error", "median_error"]:
        display[col] = display[col].map(_fmt_sci)
    for col in ["win_rate_vs_uniform", "mean_relative_error_vs_uniform", "median_relative_error_vs_uniform"]:
        display[col] = display[col].map(_fmt_float_or_dash)
    return _simple_markdown_table(display)


def _simple_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, separator, *body]) + "\n"


def _nanquantile_or_nan(values: pd.Series, quantile: float) -> float:
    array = values.to_numpy(dtype=float)
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanquantile(array, quantile))


def _to_latex_longtable(title: str, key: str, metrics: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in metrics.iterrows():
        rows.append(
            " & ".join(
                [
                    _tex(str(row["policy_label"])),
                    str(int(row["budget"])),
                    str(int(row["n_seeds"])),
                    _fmt_sci(row["mean_error"]),
                    _fmt_sci(row["median_error"]),
                    _fmt_float_or_dash(row["win_rate_vs_uniform"]),
                    _fmt_float_or_dash(row["mean_relative_error_vs_uniform"]),
                    _fmt_float_or_dash(row["median_relative_error_vs_uniform"]),
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    label = f"tab:{key.replace('_', '-')}-by-budget-metrics"
    caption = (
        f"{title} by-budget metrics. Relative errors are computed within reveal seed against "
        "the step-balanced uniform baseline before averaging or taking medians across seeds."
    )
    return rf"""\begin{{scriptsize}}
\begin{{longtable}}{{llrrrrrr}}
\caption{{{_tex(caption)}}}\label{{{label}}}\\
\toprule
Policy & Budget & Seeds & Mean error & Median error & Win rate & Mean rel. error & Median rel. error \\
\midrule
\endfirsthead
\toprule
Policy & Budget & Seeds & Mean error & Median error & Win rate & Mean rel. error & Median rel. error \\
\midrule
\endhead
{body}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
"""


def _fmt_sci(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.3e}"


def _fmt_float_or_dash(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def _tex(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


if __name__ == "__main__":
    main()
