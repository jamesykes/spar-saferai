"""Smoke test the full-data exchangeable OC3 DoS P(success) reference sampler."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saferai_budget_recovery import config
from saferai_budget_recovery.sampling import sample_full_reference_p_success


N_SAMPLES = 100_000
SEED = 12345
SUMMARY_PATH = config.FORWARD_MODEL_SMOKE_TEST_DIR / "full_reference_p_success_summary.json"
HEAD_PATH = config.FORWARD_MODEL_SMOKE_TEST_DIR / "full_reference_p_success_head.csv"


def main() -> None:
    if not config.SOTA_BETA_FITS_PATH.exists():
        raise FileNotFoundError(
            f"SOTA Beta fits not found: {config.SOTA_BETA_FITS_PATH}. "
            "Run scripts/02_fit_beta_distributions.py first."
        )

    config.FORWARD_MODEL_SMOKE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    fit_df = pd.read_csv(config.SOTA_BETA_FITS_PATH)
    sample_df = sample_full_reference_p_success(fit_df, n_samples=N_SAMPLES, seed=SEED)
    sample_df.head(50).to_csv(HEAD_PATH, index=False)

    summary = _make_summary(sample_df, n_samples=N_SAMPLES, seed=SEED)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Forward-model smoke test summary")
    print(f"Samples: {summary['n_samples']}")
    print(f"Seed: {summary['seed']}")
    print(f"P(success) mean: {summary['mean']}")
    print(f"P(success) p50: {summary['p50']}")
    print(f"P(success) p05-p95: {summary['p05']} to {summary['p95']}")
    print(f"Non-finite samples: {summary['number_of_non_finite_samples']}")
    print(f"Samples outside [0,1]: {summary['number_of_samples_outside_0_1']}")
    print(f"Summary JSON: {SUMMARY_PATH}")
    print(f"Sample head CSV: {HEAD_PATH}")


def _make_summary(sample_df: pd.DataFrame, n_samples: int, seed: int) -> dict[str, Any]:
    p_success = sample_df["p_success"].to_numpy(dtype=float)
    finite = np.isfinite(p_success)
    outside = finite & ((p_success < 0.0) | (p_success > 1.0))
    leaf_cols = [col for col in sample_df.columns if col.startswith("p_") and col not in _TACTIC_COLUMNS()]
    return {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "min": _stat(p_success, np.min),
        "max": _stat(p_success, np.max),
        "mean": _stat(p_success, np.mean),
        "std": _stat(p_success, np.std),
        "p01": _percentile(p_success, 1),
        "p05": _percentile(p_success, 5),
        "p25": _percentile(p_success, 25),
        "p50": _percentile(p_success, 50),
        "p75": _percentile(p_success, 75),
        "p95": _percentile(p_success, 95),
        "p99": _percentile(p_success, 99),
        "number_of_non_finite_samples": int((~finite).sum()),
        "number_of_samples_outside_0_1": int(outside.sum()),
        "per_leaf_sample_means": {
            col: float(sample_df[col].mean()) for col in leaf_cols
        },
        "per_tactic_sample_means": {
            col: float(sample_df[col].mean()) for col in _TACTIC_COLUMNS() if col in sample_df.columns
        },
    }


def _TACTIC_COLUMNS() -> tuple[str, ...]:
    return ("p_rec", "p_res", "p_def", "p_c2_tactic", "p_imp", "p_success")


def _stat(values: np.ndarray, fn: Any) -> float | None:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return None
    return float(fn(finite_values))


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return None
    return float(np.percentile(finite_values, percentile))


if __name__ == "__main__":
    main()

