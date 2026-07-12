# Dense By-Budget Report Artifacts

These files summarize the three dense 30-seed policy experiments by budget.

For each experiment, the generated metrics table contains:

- `mean_error`: mean squared Wasserstein-2 error to the full-data reference across reveal seeds.
- `median_error`: median squared Wasserstein-2 error across reveal seeds.
- `win_rate_vs_uniform`: strict paired win rate against `uniform_step_balanced` at the same reveal seed and budget.
- `mean_relative_error_vs_uniform`: mean of seedwise ratios `policy_error / uniform_error`.
- `median_relative_error_vs_uniform`: median of seedwise ratios `policy_error / uniform_error`.

Relative error is computed within reveal seed before averaging or taking medians. This preserves the paired reveal-order design.

At the terminal full budget, all policies have revealed the full available fitted SOTA dataset, so errors are zero and relative error is undefined. The tables keep this as missing. Strict win rate is zero at this terminal tie in the tables; the terminal point is omitted from win-rate plots to avoid visually suggesting late-budget deterioration.
