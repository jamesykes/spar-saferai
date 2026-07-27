"""Create review-only plots for alternative by-budget metrics."""

from __future__ import annotations

import os
import sys
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


ARTIFACT_DIR = config.PROJECT_ROOT / "outputs" / "report_artifacts_dense_by_budget"
OUTPUT_DIR = config.PROJECT_ROOT / "outputs" / "results_review_plots"

ROLLING_WINDOW = 5
ROLLING_MIN_PERIODS = 3
EXCLUDED_PLOT_BUDGETS = {45, 1798}

POLICY_LABELS = {
    "uniform_step_balanced": "Uniform",
    "greedy_loo_fragility": "Greedy LOO",
    "epsilon_greedy_eps0.2": "Epsilon-greedy",
    "exploration_bonus_c0.25": "Bonus c=0.25",
    "exploration_bonus_c0.5": "Bonus c=0.5",
    "exploration_bonus_c1.0": "Bonus c=1.0",
    "stochastic_normalized_fragility": "Proportional fragility",
    "uniform_positive_fragility": "Uniform positive",
    "stochastic_epsilon_greedy_eps0.2": "Stochastic epsilon-greedy",
    "stochastic_exploration_bonus_c0.25": "Stochastic bonus c=0.25",
    "stochastic_exploration_bonus_c0.5": "Stochastic bonus c=0.5",
    "stochastic_exploration_bonus_c1.0": "Stochastic bonus c=1.0",
    "softmax_normalized_fragility_temp0.25": "Softmax T=0.25",
    "softmax_normalized_fragility_temp0.5": "Softmax T=0.5",
    "softmax_normalized_fragility_temp1.0": "Softmax T=1.0",
    "softmax_normalized_fragility_temp2.0": "Softmax T=2.0",
    "softmax_normalized_fragility_temp4.0": "Softmax T=4.0",
}

EXPERIMENTS = [
    {
        "key": "dense_policy_suite",
        "title": "Initial Dense Policy Suite",
        "metrics_file": "dense_policy_suite_by_budget_metrics.csv",
        "policies": [
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
        "title": "Stochastic Fragility Ablation",
        "metrics_file": "stochastic_ablation_by_budget_metrics.csv",
        "policies": [
            "uniform_step_balanced",
            "stochastic_normalized_fragility",
            "uniform_positive_fragility",
            "stochastic_epsilon_greedy_eps0.2",
            "stochastic_exploration_bonus_c1.0",
            "stochastic_exploration_bonus_c0.5",
            "stochastic_exploration_bonus_c0.25",
        ],
    },
    {
        "key": "softmax_ablation",
        "title": "Softmax Temperature Ablation",
        "metrics_file": "softmax_temperature_ablation_by_budget_metrics.csv",
        "policies": [
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

METRICS = [
    {
        "column": "mean_error",
        "stem": "mean_error",
        "ylabel": "Mean squared W2 error",
        "reference": None,
        "exclude_uniform": False,
    },
    {
        "column": "median_error",
        "stem": "median_error",
        "ylabel": "Median squared W2 error",
        "reference": None,
        "exclude_uniform": False,
    },
    {
        "column": "mean_relative_error_vs_uniform",
        "stem": "mean_relative_error_vs_uniform",
        "ylabel": "Mean seedwise relative error vs uniform",
        "reference": 1.0,
        "exclude_uniform": True,
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for experiment in EXPERIMENTS:
        data = _load_experiment_metrics(experiment)
        for metric in METRICS:
            written.extend(_write_metric_plot(experiment, metric, data))

    print("Review-only plots generated:")
    for path in written:
        print(f"  {path}")


def _load_experiment_metrics(experiment: dict[str, object]) -> pd.DataFrame:
    path = ARTIFACT_DIR / str(experiment["metrics_file"])
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    data = pd.read_csv(path)
    return data.loc[~data["budget"].isin(EXCLUDED_PLOT_BUDGETS)].copy()


def _write_metric_plot(
    experiment: dict[str, object],
    metric: dict[str, object],
    data: pd.DataFrame,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for policy in experiment["policies"]:
        if metric["exclude_uniform"] and policy == "uniform_step_balanced":
            continue
        policy_data = data.loc[data["policy_name"].eq(policy)].sort_values("budget").copy()
        if policy_data.empty:
            raise ValueError(f"Missing policy {policy} in {experiment['key']}")
        y = (
            policy_data[str(metric["column"])]
            .rolling(window=ROLLING_WINDOW, center=True, min_periods=ROLLING_MIN_PERIODS)
            .mean()
            .to_numpy(dtype=float)
        )
        ax.plot(
            policy_data["budget"].to_numpy(dtype=float),
            y,
            linewidth=1.2,
            label=POLICY_LABELS.get(policy, policy),
        )

    if metric["reference"] is not None:
        ax.axhline(float(metric["reference"]), color="black", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.set_title(f"{experiment['title']}: {metric['ylabel']}")
    ax.set_xlabel("Total revealed SOTA elicitation rows")
    ax.set_ylabel(str(metric["ylabel"]))
    ax.set_xlim(80, 1770)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.legend(fontsize=7.2, frameon=True, ncol=2)
    fig.tight_layout()

    stem = f"{experiment['key']}_{metric['stem']}"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [png_path]


if __name__ == "__main__":
    main()
