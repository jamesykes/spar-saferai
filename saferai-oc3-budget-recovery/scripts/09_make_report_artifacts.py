"""Create report-ready tables and quick diagnostic figures from existing outputs."""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

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


REPEATED_OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "repeated_policy_experiment"
REPORT_OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "report_artifacts"
FIGURE_DIR = config.PROJECT_ROOT / "figures"

RUN_NAMES = [
    "v8_all_policies_dev",
    "exploration_bonus_sensitivity_dev",
]

MAIN_POLICY_SOURCES = [
    {
        "policy": "uniform_step_balanced",
        "source_run": "v8_all_policies_dev",
        "notes": "v8-aligned uniform baseline from the all-policy development run.",
    },
    {
        "policy": "greedy_loo_fragility",
        "source_run": "v8_all_policies_dev",
        "notes": "Pure greedy LOO-fragility policy from the all-policy development run.",
    },
    {
        "policy": "epsilon_greedy_eps0.2",
        "source_run": "v8_all_policies_dev",
        "notes": "Epsilon-greedy LOO-fragility policy with epsilon=0.2.",
    },
    {
        "policy": "exploration_bonus_c0.5",
        "source_run": "v8_all_policies_dev",
        "notes": "Exploration-bonus setting included in the all-policy development run.",
    },
    {
        "policy": "exploration_bonus_c1.0",
        "source_run": "exploration_bonus_sensitivity_dev",
        "notes": (
            "Exploration-bonus c=1.0 comes from the separate sensitivity run, not the "
            "earlier all-policy run."
        ),
    },
]

FIGURE_POLICY_SOURCES = [
    source
    for source in MAIN_POLICY_SOURCES
    if source["policy"]
    in {
        "uniform_step_balanced",
        "greedy_loo_fragility",
        "epsilon_greedy_eps0.2",
        "exploration_bonus_c1.0",
    }
]

POLICY_LABELS = {
    "uniform_step_balanced": "Uniform",
    "greedy_loo_fragility": "Greedy LOO",
    "epsilon_greedy_eps0.2": "Epsilon-greedy eps=0.2",
    "exploration_bonus_c0.5": "Exploration bonus c=0.5",
    "exploration_bonus_c1.0": "Exploration bonus c=1.0",
}


def main() -> None:
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    outputs = load_experiment_outputs(REPEATED_OUTPUT_DIR, RUN_NAMES)

    main_comparison = build_main_policy_comparison(outputs, MAIN_POLICY_SOURCES)
    error_by_budget = build_error_by_budget(outputs, MAIN_POLICY_SOURCES)
    concentration_by_budget = build_concentration_by_budget(outputs, MAIN_POLICY_SOURCES)
    sensitivity = build_exploration_bonus_sensitivity(outputs["exploration_bonus_sensitivity_dev"])

    paths = _write_tables(
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
            v8_report=outputs["v8_all_policies_dev"].get("report", {}),
            sensitivity_report=outputs["exploration_bonus_sensitivity_dev"].get("report", {}),
        ),
        encoding="utf-8",
    )
    paths["results_summary_draft"] = draft_path

    print("Report artifacts generated")
    print("Input runs:")
    for run_name in RUN_NAMES:
        report = outputs[run_name].get("report", {})
        settings = report.get("settings", {})
        reduced = report.get("settings_reduced_from_requested", "unknown")
        print(f"  {run_name}: seeds={settings.get('reveal_seeds')}, budgets={settings.get('budgets')}, reduced={reduced}")
    print("Warning: main tables combine policies from different source runs; source_run is included in every combined row.")
    print("Warning: fragility-guided policies use approximate LOO fragility with max_loo_terms_per_step=20.")
    print("Main policy comparison:")
    print(
        main_comparison[
            [
                "policy",
                "source_run",
                "average_auc",
                "win_fraction_vs_uniform",
                "mean_l1_imbalance_at_1200",
                "uses_approx_loo",
            ]
        ].to_string(index=False)
    )
    print("Exploration-bonus sensitivity:")
    print(sensitivity.to_string(index=False))
    print("Tables:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")


def _write_tables(
    main_comparison: pd.DataFrame,
    error_by_budget: pd.DataFrame,
    concentration_by_budget: pd.DataFrame,
    sensitivity: pd.DataFrame,
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
    }
    main_comparison.to_csv(paths["main_policy_comparison_csv"], index=False)
    error_by_budget.to_csv(paths["error_by_budget_csv"], index=False)
    concentration_by_budget.to_csv(paths["concentration_by_budget_csv"], index=False)
    sensitivity.to_csv(paths["exploration_bonus_sensitivity_csv"], index=False)
    write_markdown_table(main_comparison, paths["main_policy_comparison_md"])
    write_markdown_table(error_by_budget, paths["error_by_budget_md"])
    write_markdown_table(concentration_by_budget, paths["concentration_by_budget_md"])
    write_markdown_table(sensitivity, paths["exploration_bonus_sensitivity_md"])
    return paths


def _write_figures(
    main_comparison: pd.DataFrame,
    error_by_budget: pd.DataFrame,
    concentration_by_budget: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []
    figure_error = error_by_budget.loc[
        error_by_budget["policy"].isin({source["policy"] for source in FIGURE_POLICY_SOURCES})
    ].copy()
    figure_concentration = concentration_by_budget.loc[
        concentration_by_budget["policy"].isin({source["policy"] for source in FIGURE_POLICY_SOURCES})
    ].copy()
    figure_auc = main_comparison.loc[
        main_comparison["policy"].isin({source["policy"] for source in FIGURE_POLICY_SOURCES})
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, group in figure_error.groupby("policy", sort=False):
        ordered = group.sort_values("budget")
        label = POLICY_LABELS.get(policy, policy)
        x = ordered["budget"].to_numpy(dtype=float)
        y = ordered["mean_squared_w2_error"].to_numpy(dtype=float)
        p25 = ordered["p25_squared_w2_error"].to_numpy(dtype=float)
        p75 = ordered["p75_squared_w2_error"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, p25, p75, alpha=0.12)
    ax.set_xlabel("Total revealed SOTA elicitation rows")
    ax.set_ylabel("Squared W2 error to full reference")
    ax.set_title("Recovery Error vs Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    paths.extend(_save_figure(fig, "error_vs_budget"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered_auc = figure_auc.sort_values("average_auc")
    labels = [POLICY_LABELS.get(policy, policy) for policy in ordered_auc["policy"]]
    ax.bar(labels, ordered_auc["average_auc"].to_numpy(dtype=float))
    ax.set_ylabel("Average AUC of squared W2 error")
    ax.set_title("AUC by Policy")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    paths.extend(_save_figure(fig, "auc_by_policy"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, group in figure_concentration.groupby("policy", sort=False):
        ordered = group.sort_values("budget")
        ax.plot(
            ordered["budget"].to_numpy(dtype=float),
            ordered["mean_l1_imbalance"].to_numpy(dtype=float),
            marker="o",
            label=POLICY_LABELS.get(policy, policy),
        )
    ax.set_xlabel("Total revealed SOTA elicitation rows")
    ax.set_ylabel("Mean L1 step-count imbalance")
    ax.set_title("Allocation Concentration vs Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    paths.extend(_save_figure(fig, "concentration_vs_budget"))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ordered_sensitivity = sensitivity.sort_values("c")
    c_values = ordered_sensitivity["c"].to_numpy(dtype=float)
    axes[0].plot(c_values, ordered_sensitivity["average_auc"].to_numpy(dtype=float), marker="o")
    axes[0].set_xlabel("Exploration bonus c")
    axes[0].set_ylabel("Average AUC")
    axes[0].set_title("AUC by c")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(
        c_values,
        ordered_sensitivity["mean_l1_imbalance_at_1200"].to_numpy(dtype=float),
        marker="o",
        color="tab:orange",
    )
    axes[1].set_xlabel("Exploration bonus c")
    axes[1].set_ylabel("Mean L1 imbalance at 1200")
    axes[1].set_title("Concentration by c")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Exploration-Bonus Sensitivity")
    fig.tight_layout()
    paths.extend(_save_figure(fig, "exploration_bonus_sensitivity"))

    return paths


def _save_figure(fig, stem: str) -> list[Path]:
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _build_summary_draft(
    main_comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    v8_report: dict,
    sensitivity_report: dict,
) -> str:
    best_policy = main_comparison.sort_values("average_auc").iloc[0]
    uniform = main_comparison.loc[main_comparison["policy"].eq("uniform_step_balanced")].iloc[0]
    c_best = sensitivity.sort_values("average_auc").iloc[0]
    c_lowest_concentration = sensitivity.sort_values("mean_l1_imbalance_at_1200").iloc[0]
    approx_terms = sensitivity_report.get("settings", {}).get("fragility_kwargs", {}).get(
        "max_loo_terms_per_step", 20
    )
    v8_reduced = v8_report.get("settings_reduced_from_requested", "unknown")

    return f"""# Results Summary Draft

## What Was Compared

In this development run, the OC3 DoS `P(success)` budget-recovery experiment compared the v8-aligned uniform baseline against fragility-guided allocation policies. The main policy table combines the earlier `v8_all_policies_dev` run with the later `exploration_bonus_sensitivity_dev` run. Source runs are labelled explicitly in the tables because `exploration_bonus_c1.0` comes from the separate sensitivity run.

The policies summarized are:

- `uniform_step_balanced`
- `greedy_loo_fragility`
- `epsilon_greedy_eps0.2`
- `exploration_bonus_c0.5`
- `exploration_bonus_c1.0`

## Main Recovery-Error Result

The main metric is squared Wasserstein-2 error to the full-data exchangeable reference distribution for `P(success)`. Lower AUC under the error-vs-budget curve indicates better recovery in this retrospective budget-recovery setup.

In these development outputs, the lowest average AUC in the combined main table is `{best_policy['policy']}` with average AUC `{best_policy['average_auc']:.6g}`. The uniform baseline average AUC is `{uniform['average_auc']:.6g}`.

This should be read cautiously because the combined table uses different source runs for some policies. It is useful for report drafting and diagnostics, not a replacement for a single final run with all selected settings.

## Concentration Result

The uniform baseline remains step-balanced by construction. Fragility-guided policies concentrate allocation more strongly, especially pure greedy and lower exploration-bonus `c` values.

At budget 1200 in the exploration-bonus sensitivity run, the lowest concentration among the tested exploration-bonus settings was `c={c_lowest_concentration['c']}` with mean L1 imbalance `{c_lowest_concentration['mean_l1_imbalance_at_1200']:.3g}`. This is still much more concentrated than the uniform baseline, whose L1 imbalance is zero or near zero at the same budget.

## Exploration-Bonus Sensitivity

The exploration-bonus sensitivity run tested `c` values 0.25, 0.5, and 1.0. In this development run, `c={c_best['c']}` had the lowest average AUC among those settings. Increasing `c` reduced concentration in the generated sensitivity table.

This suggests that stronger exploration pressure may be useful for controlling the severe over-concentration seen in pure greedy LOO fragility. It does not prove that `c={c_best['c']}` is prospectively optimal.

## Caveats

All fragility-guided policies summarized here used approximate LOO fragility with `max_loo_terms_per_step={approx_terms}`. Exact LOO remains the mathematical v8 definition.

The full-data reference is not ground truth. It is the output distribution from all valid SOTA LLM elicitation rows under the exchangeable nodewise mixture approximation.

The budget unit is an LLM elicitation draw, not a human expert.

The experiment targets the OC3 DoS `P(success)` submodel, not full total-risk uncertainty.

LOO fragility measures finite-sample instability of the current nodewise mixture. It is not a value-of-information estimate, not optimal active learning, and not Bayesian experimental design.

The `v8_all_policies_dev` run reports `settings_reduced_from_requested={v8_reduced}`. Any final report result should use a predeclared final configuration or clearly describe any runtime-driven reductions.
"""


if __name__ == "__main__":
    main()
