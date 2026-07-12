"""Create selected smoothed figures for the draft Results section."""

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


ARTIFACT_DIR = config.PROJECT_ROOT / "outputs" / "report_artifacts_dense_by_budget"
DRAFT_FIGURE_DIR = config.PROJECT_ROOT / "report" / "drafts" / "original_draft_latex_repo" / "figures" / "results"
REPORT_FIGURE_DIR = config.PROJECT_ROOT / "report" / "figures" / "results"

ROLLING_WINDOW = 5
ROLLING_MIN_PERIODS = 3
EXCLUDED_PLOT_BUDGETS = {45, 1798}

POLICY_LABELS = {
    "greedy_loo_fragility": "Greedy LOO",
    "epsilon_greedy_eps0.2": "Epsilon-greedy",
    "exploration_bonus_c1.0": "Exploration bonus c=1.0",
    "stochastic_normalized_fragility": "Proportional fragility",
    "uniform_positive_fragility": "Uniform positive",
    "stochastic_epsilon_greedy_eps0.2": "Stochastic epsilon-greedy",
    "stochastic_exploration_bonus_c1.0": "Stochastic bonus c=1.0",
    "softmax_normalized_fragility_temp0.25": "Softmax T=0.25",
    "softmax_normalized_fragility_temp2.0": "Softmax T=2.0",
    "softmax_normalized_fragility_temp4.0": "Softmax T=4.0",
}

POLICY_COLORS = {
    "stochastic_normalized_fragility": "#1f77b4",
    "epsilon_greedy_eps0.2": "#ff7f0e",
    "exploration_bonus_c1.0": "#2ca02c",
    "greedy_loo_fragility": "#d62728",
    "uniform_positive_fragility": "#9467bd",
    "stochastic_epsilon_greedy_eps0.2": "#ff7f0e",
    "stochastic_exploration_bonus_c1.0": "#2ca02c",
    "softmax_normalized_fragility_temp0.25": "#d62728",
    "softmax_normalized_fragility_temp2.0": "#8c564b",
    "softmax_normalized_fragility_temp4.0": "#e377c2",
}

FIGURE_SPECS = [
    {
        "stem": "initial_suite_smoothed_median_relative_error",
        "title": "Initial Dense Policy Suite",
        "metrics_file": "dense_policy_suite_by_budget_metrics.csv",
        "metric": "median_relative_error_vs_uniform",
        "ylabel": "Median relative error vs uniform",
        "reference": 1.0,
        "policies": [
            "stochastic_normalized_fragility",
            "epsilon_greedy_eps0.2",
            "exploration_bonus_c1.0",
            "greedy_loo_fragility",
        ],
        "ylim": (0.5, 7.8),
    },
    {
        "stem": "initial_suite_smoothed_win_rate",
        "title": "Initial Dense Policy Suite",
        "metrics_file": "dense_policy_suite_by_budget_metrics.csv",
        "metric": "win_rate_vs_uniform",
        "ylabel": "Paired win rate vs uniform",
        "reference": 0.5,
        "policies": [
            "stochastic_normalized_fragility",
            "epsilon_greedy_eps0.2",
            "exploration_bonus_c1.0",
            "greedy_loo_fragility",
        ],
        "ylim": (0.0, 0.8),
    },
    {
        "stem": "stochastic_ablation_smoothed_median_relative_error",
        "title": "Stochastic Fragility Ablation",
        "metrics_file": "stochastic_ablation_by_budget_metrics.csv",
        "metric": "median_relative_error_vs_uniform",
        "ylabel": "Median relative error vs uniform",
        "reference": 1.0,
        "policies": [
            "stochastic_normalized_fragility",
            "uniform_positive_fragility",
            "stochastic_epsilon_greedy_eps0.2",
            "stochastic_exploration_bonus_c1.0",
        ],
        "ylim": (0.65, 1.6),
    },
    {
        "stem": "stochastic_ablation_smoothed_win_rate",
        "title": "Stochastic Fragility Ablation",
        "metrics_file": "stochastic_ablation_by_budget_metrics.csv",
        "metric": "win_rate_vs_uniform",
        "ylabel": "Paired win rate vs uniform",
        "reference": 0.5,
        "policies": [
            "stochastic_normalized_fragility",
            "uniform_positive_fragility",
            "stochastic_epsilon_greedy_eps0.2",
            "stochastic_exploration_bonus_c1.0",
        ],
        "ylim": (0.15, 0.75),
    },
    {
        "stem": "softmax_ablation_smoothed_median_relative_error",
        "title": "Softmax Temperature Ablation",
        "metrics_file": "softmax_temperature_ablation_by_budget_metrics.csv",
        "metric": "median_relative_error_vs_uniform",
        "ylabel": "Median relative error vs uniform",
        "reference": 1.0,
        "policies": [
            "uniform_positive_fragility",
            "stochastic_normalized_fragility",
            "softmax_normalized_fragility_temp0.25",
            "softmax_normalized_fragility_temp2.0",
            "softmax_normalized_fragility_temp4.0",
        ],
        "ylim": (0.65, 1.85),
    },
    {
        "stem": "softmax_ablation_smoothed_win_rate",
        "title": "Softmax Temperature Ablation",
        "metrics_file": "softmax_temperature_ablation_by_budget_metrics.csv",
        "metric": "win_rate_vs_uniform",
        "ylabel": "Paired win rate vs uniform",
        "reference": 0.5,
        "policies": [
            "uniform_positive_fragility",
            "stochastic_normalized_fragility",
            "softmax_normalized_fragility_temp0.25",
            "softmax_normalized_fragility_temp2.0",
            "softmax_normalized_fragility_temp4.0",
        ],
        "ylim": (0.15, 0.75),
    },
]


def main() -> None:
    DRAFT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for spec in FIGURE_SPECS:
        written.extend(_write_figure(spec))

    print("Results-section figures generated:")
    for path in written:
        print(f"  {path}")


def _write_figure(spec: dict[str, object]) -> list[Path]:
    metrics_path = ARTIFACT_DIR / str(spec["metrics_file"])
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    data = pd.read_csv(metrics_path)
    data = data.loc[~data["budget"].isin(EXCLUDED_PLOT_BUDGETS)].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for policy in spec["policies"]:
        policy_data = data.loc[data["policy_name"].eq(policy)].sort_values("budget").copy()
        if policy_data.empty:
            raise ValueError(f"Missing policy {policy} in {metrics_path}")
        y = (
            policy_data[str(spec["metric"])]
            .rolling(window=ROLLING_WINDOW, center=True, min_periods=ROLLING_MIN_PERIODS)
            .mean()
            .to_numpy(dtype=float)
        )
        ax.plot(
            policy_data["budget"].to_numpy(dtype=float),
            y,
            linewidth=1.35,
            color=POLICY_COLORS.get(policy),
            label=POLICY_LABELS.get(policy, policy),
        )

    reference = float(spec["reference"])
    ax.axhline(reference, color="black", linewidth=0.95, linestyle="--", alpha=0.65)
    ax.set_title(str(spec["title"]))
    ax.set_xlabel("Total revealed SOTA elicitation rows")
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_xlim(80, 1770)
    ax.set_ylim(*spec["ylim"])
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.legend(fontsize=7.5, frameon=True)
    fig.tight_layout()

    written: list[Path] = []
    for directory in (DRAFT_FIGURE_DIR, REPORT_FIGURE_DIR):
        pdf_path = directory / f"{spec['stem']}.pdf"
        png_path = directory / f"{spec['stem']}.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=220)
        written.extend([pdf_path, png_path])
    plt.close(fig)
    return written


if __name__ == "__main__":
    main()
