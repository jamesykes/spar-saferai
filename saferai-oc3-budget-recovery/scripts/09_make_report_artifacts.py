"""Create locked v8 hidden-reveal report artifacts and figures."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "saferai_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saferai_budget_recovery import config
from saferai_budget_recovery.reporting import (
    build_concentration_by_budget,
    build_error_by_budget,
    build_exploration_bonus_sensitivity,
    build_main_policy_comparison,
    load_experiment_outputs,
    write_markdown_table,
)


LOCKED_RUN_NAME = "locked_v8_hidden_reveal"
LOCKED_MODE = "LOCKED_V8_HIDDEN_REVEAL"
LOCKED_PROTOCOL = "shared_per_step_model_aware_orders"

REPEATED_OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "repeated_policy_experiment"
REPORT_OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "report_artifacts_locked_v8_hidden_reveal"
REPORT_TABLE_DIR = config.PROJECT_ROOT / "report" / "tables"
FIGURE_DIR = config.PROJECT_ROOT / "report" / "figures"

EXPECTED_LOCKED_FILES = [
    "locked_v8_hidden_reveal_repeated_policy_results.csv",
    "locked_v8_hidden_reveal_policy_summary_by_budget.csv",
    "locked_v8_hidden_reveal_policy_differences_vs_uniform.csv",
    "locked_v8_hidden_reveal_policy_auc_by_seed.csv",
    "locked_v8_hidden_reveal_win_rate_by_budget.csv",
    "locked_v8_hidden_reveal_win_rate_by_seed.csv",
    "locked_v8_hidden_reveal_concentration_by_budget.csv",
    "locked_v8_hidden_reveal_selected_step_counts.csv",
    "locked_v8_hidden_reveal_fragility_runtime_diagnostics.csv",
    "locked_v8_hidden_reveal_exploration_bonus_c_value_summary.csv",
    "locked_v8_hidden_reveal_repeated_policy_experiment_report.json",
]

POLICY_ORDER = [
    "uniform_step_balanced",
    "greedy_loo_fragility",
    "epsilon_greedy_eps0.2",
    "exploration_bonus_c0.25",
    "exploration_bonus_c0.5",
    "exploration_bonus_c1.0",
]

MAIN_POLICY_SOURCES = [
    {
        "policy": "uniform_step_balanced",
        "source_run": LOCKED_RUN_NAME,
        "notes": "Step-balanced uniform baseline.",
    },
    {
        "policy": "greedy_loo_fragility",
        "source_run": LOCKED_RUN_NAME,
        "notes": "Pure LOO-fragility heuristic.",
    },
    {
        "policy": "epsilon_greedy_eps0.2",
        "source_run": LOCKED_RUN_NAME,
        "notes": "LOO fragility with epsilon=0.2 exploration.",
    },
    {
        "policy": "exploration_bonus_c0.25",
        "source_run": LOCKED_RUN_NAME,
        "notes": "LOO fragility plus exploration bonus, c=0.25.",
    },
    {
        "policy": "exploration_bonus_c0.5",
        "source_run": LOCKED_RUN_NAME,
        "notes": "LOO fragility plus exploration bonus, c=0.5.",
    },
    {
        "policy": "exploration_bonus_c1.0",
        "source_run": LOCKED_RUN_NAME,
        "notes": "LOO fragility plus exploration bonus, c=1.0.",
    },
]

POLICY_LABELS = {
    "uniform_step_balanced": "Uniform step-balanced",
    "greedy_loo_fragility": "Greedy LOO-fragility",
    "epsilon_greedy_eps0.2": "Epsilon-greedy eps=0.2",
    "exploration_bonus_c0.25": "Exploration bonus c=0.25",
    "exploration_bonus_c0.5": "Exploration bonus c=0.5",
    "exploration_bonus_c1.0": "Exploration bonus c=1.0",
}

PLOT_POLICY_LABELS = {
    **POLICY_LABELS,
    "uniform_step_balanced": "Uniform",
}


def main() -> None:
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    _validate_locked_outputs()
    outputs = load_experiment_outputs(REPEATED_OUTPUT_DIR, [LOCKED_RUN_NAME])
    locked_outputs = outputs[LOCKED_RUN_NAME]
    locked_report = locked_outputs["report"]

    main_comparison = build_main_policy_comparison(
        outputs,
        MAIN_POLICY_SOURCES,
        max_budget=1200,
        concentration_budgets=(1200, 1798),
    )
    error_by_budget = build_error_by_budget(outputs, MAIN_POLICY_SOURCES)
    concentration_by_budget = build_concentration_by_budget(outputs, MAIN_POLICY_SOURCES)
    sensitivity = build_exploration_bonus_sensitivity(
        locked_outputs,
        source_run=LOCKED_RUN_NAME,
        concentration_budget=1200,
    )
    win_counts_by_budget = _build_win_counts_by_budget(locked_outputs)

    artifact_paths = _write_artifact_tables(
        main_comparison=main_comparison,
        error_by_budget=error_by_budget,
        concentration_by_budget=concentration_by_budget,
        sensitivity=sensitivity,
        win_counts_by_budget=win_counts_by_budget,
    )
    latex_paths = _write_latex_tables(
        main_comparison=main_comparison,
        error_by_budget=error_by_budget,
        concentration_by_budget=concentration_by_budget,
        sensitivity=sensitivity,
    )
    figure_paths = _write_figures(
        main_comparison=main_comparison,
        error_by_budget=error_by_budget,
        concentration_by_budget=concentration_by_budget,
        sensitivity=sensitivity,
    )

    draft_path = REPORT_OUTPUT_DIR / "results_summary_draft.md"
    draft_path.write_text(
        _build_summary_draft(
            main_comparison=main_comparison,
            sensitivity=sensitivity,
            locked_report=locked_report,
        ),
        encoding="utf-8",
    )
    artifact_paths["results_summary_draft"] = draft_path

    print("Locked report artifacts generated")
    print(f"Source run: {LOCKED_RUN_NAME}")
    print(f"Report mode: {locked_report.get('mode')}")
    print(f"Hidden reveal orders: {locked_report.get('uses_hidden_reveal_orders')}")
    print(f"Hidden reveal protocol: {locked_report.get('hidden_reveal_order_protocol')}")
    print(f"Settings reduced: {locked_report.get('settings_reduced_from_requested')}")
    print("Main policy comparison:")
    print(
        main_comparison[
            [
                "policy",
                "source_run",
                "average_auc",
                "median_auc",
                "win_fraction_vs_uniform",
                "mean_l1_imbalance_at_1200",
                "mean_l1_imbalance_at_1798",
            ]
        ].to_string(index=False)
    )
    print("Exploration-bonus sensitivity:")
    print(sensitivity.to_string(index=False))
    print("Win counts by budget:")
    print(win_counts_by_budget.to_string(index=False))
    print("Artifacts:")
    for label, path in artifact_paths.items():
        print(f"  {label}: {path}")
    print("LaTeX tables:")
    for path in latex_paths:
        print(f"  {path}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")


def _validate_locked_outputs() -> None:
    missing = [name for name in EXPECTED_LOCKED_FILES if not (REPEATED_OUTPUT_DIR / name).exists()]
    if missing:
        formatted = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(f"Missing expected locked output files:\n{formatted}")

    report_path = REPEATED_OUTPUT_DIR / "locked_v8_hidden_reveal_repeated_policy_experiment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = {
        "mode": report.get("mode") == LOCKED_MODE,
        "uses_hidden_reveal_orders": report.get("uses_hidden_reveal_orders") is True,
        "hidden_reveal_order_protocol": report.get("hidden_reveal_order_protocol") == LOCKED_PROTOCOL,
        "settings_reduced_from_requested": report.get("settings_reduced_from_requested") is False,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        details = {name: report.get(name) for name in checks}
        raise ValueError(f"Locked report validation failed for {failed}: {details}")


def _write_artifact_tables(
    *,
    main_comparison: pd.DataFrame,
    error_by_budget: pd.DataFrame,
    concentration_by_budget: pd.DataFrame,
    sensitivity: pd.DataFrame,
    win_counts_by_budget: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "main_policy_comparison_csv": REPORT_OUTPUT_DIR / "main_policy_comparison.csv",
        "main_policy_comparison_md": REPORT_OUTPUT_DIR / "main_policy_comparison.md",
        "error_by_budget_csv": REPORT_OUTPUT_DIR / "error_by_budget.csv",
        "error_by_budget_md": REPORT_OUTPUT_DIR / "error_by_budget.md",
        "concentration_by_budget_csv": REPORT_OUTPUT_DIR / "concentration_by_budget.csv",
        "concentration_by_budget_md": REPORT_OUTPUT_DIR / "concentration_by_budget.md",
        "exploration_bonus_sensitivity_csv": REPORT_OUTPUT_DIR / "exploration_bonus_sensitivity.csv",
        "exploration_bonus_sensitivity_md": REPORT_OUTPUT_DIR / "exploration_bonus_sensitivity.md",
        "win_counts_by_budget_csv": REPORT_OUTPUT_DIR / "win_counts_by_budget.csv",
        "win_counts_by_budget_md": REPORT_OUTPUT_DIR / "win_counts_by_budget.md",
    }
    main_comparison.to_csv(paths["main_policy_comparison_csv"], index=False)
    error_by_budget.to_csv(paths["error_by_budget_csv"], index=False)
    concentration_by_budget.to_csv(paths["concentration_by_budget_csv"], index=False)
    sensitivity.to_csv(paths["exploration_bonus_sensitivity_csv"], index=False)
    win_counts_by_budget.to_csv(paths["win_counts_by_budget_csv"], index=False)
    write_markdown_table(main_comparison, paths["main_policy_comparison_md"])
    write_markdown_table(error_by_budget, paths["error_by_budget_md"])
    write_markdown_table(concentration_by_budget, paths["concentration_by_budget_md"])
    write_markdown_table(sensitivity, paths["exploration_bonus_sensitivity_md"])
    write_markdown_table(win_counts_by_budget, paths["win_counts_by_budget_md"])
    return paths


def _write_latex_tables(
    *,
    main_comparison: pd.DataFrame,
    error_by_budget: pd.DataFrame,
    concentration_by_budget: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> list[Path]:
    paths = [
        REPORT_TABLE_DIR / "main_policy_comparison.tex",
        REPORT_TABLE_DIR / "error_by_budget.tex",
        REPORT_TABLE_DIR / "concentration_by_budget.tex",
        REPORT_TABLE_DIR / "exploration_bonus_sensitivity.tex",
    ]
    paths[0].write_text(_main_policy_latex(main_comparison), encoding="utf-8")
    paths[1].write_text(_error_by_budget_latex(error_by_budget), encoding="utf-8")
    paths[2].write_text(_concentration_by_budget_latex(concentration_by_budget), encoding="utf-8")
    paths[3].write_text(_sensitivity_latex(sensitivity), encoding="utf-8")
    return paths


def _write_figures(
    *,
    main_comparison: pd.DataFrame,
    error_by_budget: pd.DataFrame,
    concentration_by_budget: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []
    error_ordered = _sort_by_policy_order(error_by_budget)
    concentration_ordered = _sort_by_policy_order(concentration_by_budget)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for policy in POLICY_ORDER:
        group = error_ordered.loc[error_ordered["policy"].eq(policy)].sort_values("budget")
        if group.empty:
            continue
        x = group["budget"].to_numpy(dtype=float)
        y = group["mean_squared_w2_error"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=PLOT_POLICY_LABELS.get(policy, policy))
    ax.set_xlabel("Total revealed SOTA elicitation rows")
    ax.set_ylabel("Squared W2 error to full reference")
    ax.set_title("Distance to Full Reference")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)
    paths.extend(_save_figure(fig, "error_vs_budget"))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ordered_auc = main_comparison.sort_values("average_auc")
    labels = [PLOT_POLICY_LABELS.get(policy, policy) for policy in ordered_auc["policy"]]
    ax.bar(labels, ordered_auc["average_auc"].to_numpy(dtype=float))
    ax.set_ylabel("Average AUC of squared W2 error")
    ax.set_title("Locked Hidden-Reveal AUC by Policy")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    paths.extend(_save_figure(fig, "auc_by_policy"))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for policy in POLICY_ORDER:
        group = concentration_ordered.loc[concentration_ordered["policy"].eq(policy)].sort_values("budget")
        if group.empty:
            continue
        ax.plot(
            group["budget"].to_numpy(dtype=float),
            group["mean_l1_imbalance"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            label=PLOT_POLICY_LABELS.get(policy, policy),
        )
    ax.set_xlabel("Total revealed SOTA elicitation rows")
    ax.set_ylabel("Mean L1 step-count imbalance")
    ax.set_title("Allocation Concentration vs Budget")
    ax.text(
        0.99,
        0.04,
        "All policies converge at full budget 1798.",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)
    paths.extend(_save_figure(fig, "concentration_vs_budget"))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    ordered_sensitivity = sensitivity.sort_values("c")
    c_values = ordered_sensitivity["c"].to_numpy(dtype=float)
    panels = [
        ("average_auc", "Average AUC", "AUC by c"),
        ("win_fraction_vs_uniform", "Win fraction vs uniform", "Paired wins by c"),
        ("mean_l1_imbalance_at_1200", "Mean L1 imbalance at 1200", "Concentration by c"),
    ]
    for ax, (column, ylabel, title) in zip(axes, panels, strict=True):
        ax.plot(c_values, ordered_sensitivity[column].to_numpy(dtype=float), marker="o")
        ax.set_xlabel("Exploration bonus c")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Locked Exploration-Bonus Sensitivity")
    fig.tight_layout()
    paths.extend(_save_figure(fig, "exploration_bonus_sensitivity"))

    return paths


def _save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return [png_path]


def _build_win_counts_by_budget(locked_outputs: dict[str, Any]) -> pd.DataFrame:
    win_rate = locked_outputs.get("win_rate_by_budget")
    if win_rate is None or win_rate.empty:
        raise ValueError("Locked outputs are missing win_rate_by_budget data.")

    rows: list[dict[str, Any]] = []
    for budget, group in win_rate.groupby("budget", dropna=False):
        row: dict[str, Any] = {
            "budget": int(budget),
            "n_reveal_seeds": int(group["n_seeds"].max()),
        }
        for policy in POLICY_ORDER:
            if policy == "uniform_step_balanced":
                continue
            policy_rows = group.loc[group["policy_name"].eq(policy)]
            if policy_rows.empty:
                row[f"{policy}_wins"] = pd.NA
                continue
            row[f"{policy}_wins"] = int(policy_rows["policy_wins"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("budget").reset_index(drop=True)


def _main_policy_latex(df: pd.DataFrame) -> str:
    ordered = df.sort_values("average_auc")
    rows = []
    for _, row in ordered.iterrows():
        win = "--" if pd.isna(row["win_fraction_vs_uniform"]) else _fmt(row["win_fraction_vs_uniform"], 3)
        rows.append(
            " & ".join(
                [
                    _tex(POLICY_LABELS.get(row["policy"], row["policy"])),
                    _fmt(row["average_auc"], 6),
                    _fmt(row["median_auc"], 6),
                    win,
                    _fmt(row["mean_l1_imbalance_at_1200"], 1),
                    _fmt(row["mean_l1_imbalance_at_1798"], 1),
                    _tex(row["notes"]),
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
\centering
\small
\caption{{Locked main policy comparison. All rows use \texttt{{locked\_v8\_hidden\_reveal}}. AUC is the area under the squared Wasserstein-2 error curve over the selected budget schedule; lower is better. Win fraction is paired against the step-balanced uniform baseline.}}
\label{{tab:main-policy-comparison}}
\begin{{tabularx}}{{\textwidth}}{{lrrrrrX}}
\toprule
Policy & Avg. AUC & Med. AUC & Win frac. & L1 1200 & L1 1798 & Note \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table}}
"""


def _sensitivity_latex(df: pd.DataFrame) -> str:
    ordered = df.sort_values("c")
    rows = []
    for _, row in ordered.iterrows():
        rows.append(
            " & ".join(
                [
                    _fmt(row["c"], 2),
                    _fmt(row["average_auc"], 6),
                    _fmt(row["median_auc"], 6),
                    _fmt(row["win_fraction_vs_uniform"], 3),
                    _fmt(row["mean_l1_imbalance_at_1200"], 1),
                    _fmt(row["mean_max_min_ratio_at_1200"], 2),
                    _fmt(row["min_step_count_at_1200_mean"], 1),
                    _fmt(row["observed_min_step_count_at_1200"], 0),
                    _fmt(row["mean_l1_imbalance_at_1798"], 1),
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
\centering
\small
\caption{{Locked exploration-bonus sensitivity. All settings use approximate LOO fragility with \texttt{{max\_loo\_terms\_per\_step=20}}.}}
\label{{tab:exploration-bonus-sensitivity}}
\begin{{tabular}}{{rrrrrrrrr}}
\toprule
$c$ & Avg. AUC & Med. AUC & Win frac. & L1 1200 & Max/min 1200 & Mean min 1200 & Obs. min 1200 & L1 1798 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _error_by_budget_latex(df: pd.DataFrame) -> str:
    ordered = _sort_by_policy_order(df)
    rows = []
    for _, row in ordered.iterrows():
        rows.append(
            " & ".join(
                [
                    _tex(POLICY_LABELS.get(row["policy"], row["policy"])),
                    str(int(row["budget"])),
                    str(int(row["n_reveal_seeds"])),
                    _fmt_sci(row["mean_squared_w2_error"]),
                    _fmt_sci(row["median_squared_w2_error"]),
                    _fmt_sci(row["p25_squared_w2_error"]),
                    _fmt_sci(row["p75_squared_w2_error"]),
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{scriptsize}}
\begin{{longtable}}{{lrrrrrr}}
\caption{{Locked squared Wasserstein-2 error by budget. All rows use \texttt{{locked\_v8\_hidden\_reveal}}. Lower values indicate closer recovery of the full-data reference output distribution.}}\label{{tab:error-by-budget}}\\
\toprule
Policy & Budget & Seeds & Mean & Median & P25 & P75 \\
\midrule
\endfirsthead
\toprule
Policy & Budget & Seeds & Mean & Median & P25 & P75 \\
\midrule
\endhead
{body}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
"""


def _concentration_by_budget_latex(df: pd.DataFrame) -> str:
    ordered = _sort_by_policy_order(df)
    rows = []
    for _, row in ordered.iterrows():
        rows.append(
            " & ".join(
                [
                    _tex(POLICY_LABELS.get(row["policy"], row["policy"])),
                    str(int(row["budget"])),
                    str(int(row["n_reveal_seeds"])),
                    _fmt(row["mean_l1_imbalance"], 1),
                    _fmt(row["median_l1_imbalance"], 1),
                    _fmt(row["mean_max_min_ratio"], 2),
                    _fmt(row["mean_minimum_step_count"], 1),
                    _fmt(row["mean_maximum_step_count"], 1),
                    _fmt(row["mean_steps_still_at_initial_seed_count"], 1),
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{landscape}}
\begin{{scriptsize}}
\begin{{longtable}}{{lrrrrrrrr}}
\caption{{Locked allocation concentration by budget. L1 imbalance is measured against a perfectly step-balanced allocation vector. All policies reach zero L1 imbalance at budget 1798 because all available fitted SOTA rows have been revealed.}}\label{{tab:concentration-by-budget}}\\
\toprule
Policy & Budget & Seeds & Mean L1 & Med. L1 & Mean max/min & Mean min step & Mean max step & Steps at 5 \\
\midrule
\endfirsthead
\toprule
Policy & Budget & Seeds & Mean L1 & Med. L1 & Mean max/min & Mean min step & Mean max step & Steps at 5 \\
\midrule
\endhead
{body}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
\end{{landscape}}
"""


def _build_summary_draft(
    *,
    main_comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    locked_report: dict[str, Any],
) -> str:
    best_policy = main_comparison.sort_values("average_auc").iloc[0]
    uniform = main_comparison.loc[main_comparison["policy"].eq("uniform_step_balanced")].iloc[0]
    c_best = sensitivity.sort_values("average_auc").iloc[0]
    c_lowest_concentration = sensitivity.sort_values("mean_l1_imbalance_at_1200").iloc[0]
    settings = locked_report.get("settings", {})
    budgets = settings.get("budgets", [])
    reveal_seeds = settings.get("reveal_seeds", [])
    approx_terms = settings.get("fragility_kwargs", {}).get("max_loo_terms_per_step")

    return f"""# Locked V8 Hidden-Reveal Results Summary Draft

## Source Run

All report artifacts in this directory use only `{LOCKED_RUN_NAME}` outputs.

The report JSON says:

- `mode = {locked_report.get('mode')}`
- `uses_hidden_reveal_orders = {locked_report.get('uses_hidden_reveal_orders')}`
- `hidden_reveal_order_protocol = {locked_report.get('hidden_reveal_order_protocol')}`
- `settings_reduced_from_requested = {locked_report.get('settings_reduced_from_requested')}`

The locked run used the shared hidden reveal-order protocol. For each reveal seed, per-step hidden reveal orders were shared across policies, and policies selected only the MITRE-step input to reveal next.

## Locked Settings

- Policies run: {len(POLICY_ORDER)}
- Reveal seeds: {len(reveal_seeds)} ({reveal_seeds})
- Budgets: {budgets}
- Approximate LOO cap for fragility-guided policies: `max_loo_terms_per_step={approx_terms}`
- Requested settings reduced: {locked_report.get('settings_reduced_from_requested')}

## Main Recovery-Error Result

The main metric is squared Wasserstein-2 error to the full-data exchangeable reference distribution for `P(success)`. Lower AUC under the error-vs-budget curve indicates better aggregate recovery over the selected budget grid.

The lowest locked average AUC is `{best_policy['policy']}` with average AUC `{best_policy['average_auc']:.12g}`. The uniform baseline average AUC is `{uniform['average_auc']:.12g}`.

Under the corrected hidden reveal-order protocol, the best average-AUC policy was therefore an exploration-bonus fragility policy. However, all fragility-guided policies had paired win fractions below 0.5 against the uniform baseline, so the AUC advantage is aggregation-dependent and should be interpreted cautiously.

## Concentration Result

At budget 1200, the lowest concentration among exploration-bonus settings was `c={c_lowest_concentration['c']}` with mean L1 imbalance `{c_lowest_concentration['mean_l1_imbalance_at_1200']:.3g}`. Uniform remains balanced by construction, with mean L1 imbalance 0 at budget 1200.

At budget 1798, all policies have mean, median, and max L1 imbalance 0. This is expected because the full available fitted SOTA dataset has been revealed.

## Exploration-Bonus Sensitivity

The locked run included exploration-bonus settings `c=0.25`, `c=0.5`, and `c=1.0`. Among these, `c={c_best['c']}` had the lowest average AUC. Larger `c` also reduced concentration at budget 1200 in this locked run.

This suggests that stronger exploration pressure may help control over-concentration in LOO-fragility allocation. It does not show that `c={c_best['c']}` is prospectively optimal.

## Caveats

The locked run provides weak-to-moderate evidence that fragility-guided policies can improve aggregate recovery error, especially with a stronger exploration bonus. It does not show majority paired wins over uniform.

The full-data reference is not ground truth. It is the output distribution from all valid SOTA LLM elicitation rows under the exchangeable nodewise mixture approximation.

The budget unit is an LLM elicitation draw, not a human expert.

The experiment targets the OC3 DoS `P(success)` submodel, not full total-risk uncertainty.

LOO fragility measures finite-sample instability of the current nodewise mixture. It is not a value-of-information estimate, not optimal active learning, and not Bayesian experimental design.

Fragility-guided policies used approximate LOO fragility with `max_loo_terms_per_step={approx_terms}`.
"""


def _sort_by_policy_order(df: pd.DataFrame) -> pd.DataFrame:
    order = {policy: index for index, policy in enumerate(POLICY_ORDER)}
    return (
        df.assign(_policy_order=df["policy"].map(order).fillna(len(order)))
        .sort_values(["_policy_order", "budget"] if "budget" in df.columns else ["_policy_order"])
        .drop(columns=["_policy_order"])
        .reset_index(drop=True)
    )


def _fmt(value: Any, digits: int) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if value == 0:
        return "0"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_sci(value: Any) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.6e}"


def _tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
