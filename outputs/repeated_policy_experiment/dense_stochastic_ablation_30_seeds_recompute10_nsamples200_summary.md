# Dense Stochastic Fragility Ablation

## Purpose

This experiment tested whether the recent improvement from `stochastic_normalized_fragility` came from using the magnitude of LOO fragility scores, or from the weaker choice to randomize among steps that currently look fragile.

The earlier deterministic fragility policies used argmax-style selection: choose the step with the largest fragility-derived score. This ablation instead tested stochasticized variants that sample among plausible fragile steps. The goal was to separate three possible effects:

- Whether positive LOO fragility contains useful signal.
- Whether deterministic argmax over-exploits noisy fragility estimates.
- Whether raw fragility magnitudes are well calibrated enough to use as sampling weights.

The old deterministic policies were not rerun here. The comparison target in this ablation is the step-balanced uniform baseline plus stochastic fragility variants.

## Source Outputs

The main artifacts are in `saferai-oc3-budget-recovery/outputs/repeated_policy_experiment/` with prefix:

`dense_stochastic_ablation_30_seeds_recompute10_nsamples200`

Key files:

- `dense_stochastic_ablation_30_seeds_recompute10_nsamples200_repeated_policy_results.csv`
- `dense_stochastic_ablation_30_seeds_recompute10_nsamples200_policy_auc_by_seed.csv`
- `dense_stochastic_ablation_30_seeds_recompute10_nsamples200_win_rate_by_budget.csv`
- `dense_stochastic_ablation_30_seeds_recompute10_nsamples200_repeated_policy_experiment_report.json`

The run produced 8,400 result rows, equal to 7 policies x 30 reveal seeds x 40 budgets.

## Shared Experiment Settings

- Mode: `DENSE_STOCHASTIC_ABLATION_30_SEEDS_RECOMPUTE10_NSAMPLES200`
- Reveal seeds: 30 seeds, from `101` through `3030`
- Budgets: 40 dense budgets, every 45 rows from `45` through `1755`, plus full reveal at `1798`
- Reference samples: `n_reference_samples = 50000`
- Budget samples: `n_budget_samples = 20000`
- Main sampling grid: `n_grid = 501`
- Fragility samples: `n_samples = 200`
- Fragility grid: `n_grid = 101`
- LOO cap: `max_loo_terms_per_step = 20`
- Fragility recomputation cadence: every 10 reveals
- Hidden reveal protocol: shared hidden reveal orders by seed, so policies choose only which MITRE step to reveal next

The metric is squared Wasserstein-2 distance from the full-data exchangeable reference distribution for `P(success)`. Lower error is better. AUC is the area under the error-vs-budget curve; lower AUC is better.

## Policies

### `uniform_step_balanced`

This is the baseline. At each reveal, it chooses uniformly among the currently available steps with the smallest revealed count. It therefore keeps the reveal allocation balanced across MITRE steps by construction.

### `stochastic_normalized_fragility`

This samples an available step with probability proportional to its finite positive LOO fragility:

`p(step) proportional to max(loo_fragility(step), 0)`

If there is no positive finite fragility mass, it falls back to `uniform_step_balanced`.

This is the original stochastic fragility policy in the ablation. It uses raw fragility magnitudes as sampling weights.

### `uniform_positive_fragility`

This first identifies available steps with finite positive LOO fragility, then samples uniformly among those steps:

`p(step) = 1 / number_of_positive_fragility_steps`

If there are no finite positive fragility steps, it falls back to `uniform_step_balanced`.

This is the diagnostic ablation for the magnitude weighting in `stochastic_normalized_fragility`. If this performs similarly to or better than `stochastic_normalized_fragility`, that suggests that the sign/filtering of fragility is useful but the raw magnitudes may be poorly calibrated as probabilities.

### `stochastic_epsilon_greedy_eps0.2`

This is an epsilon-greedy stochastic fragility policy:

- With probability `epsilon = 0.2`, choose by `uniform_step_balanced`.
- With probability `0.8`, sample proportional to finite positive LOO fragility, as in `stochastic_normalized_fragility`.

This tests whether preserving some explicit balanced exploration improves the stochastic fragility policy.

### `stochastic_exploration_bonus_c0.25`

This samples proportional to a fragility-plus-exploration acquisition score:

`score(step) = positive_fragility(step) + lambda / sqrt(n_step)`

where:

- `positive_fragility(step) = max(loo_fragility(step), 0)`
- `n_step` is the current revealed count for that step, lower bounded at 1
- `lambda = c * median_positive_fragility`
- `c = 0.25`

The policy samples with probability proportional to this score. If the score has no positive mass, it falls back to `uniform_step_balanced`.

### `stochastic_exploration_bonus_c0.5`

Same as `stochastic_exploration_bonus_c0.25`, but with `c = 0.5`.

### `stochastic_exploration_bonus_c1.0`

Same as `stochastic_exploration_bonus_c0.25`, but with `c = 1.0`.

This was included because the earlier deterministic exploration-bonus run suggested that stronger exploration pressure could reduce over-concentration.

## Overall AUC Summary

This table averages over all budgets, including early budgets and the full-reveal endpoint. Lower AUC and lower average distance are better. The win column compares each policy against `uniform_step_balanced` at matched seed-budget pairs.

| policy | average_auc | median_auc | avg_distance | all-budget win vs uniform |
| --- | ---: | ---: | ---: | ---: |
| `uniform_step_balanced` | 0.011815 | 0.010616 | 7.190e-06 |  |
| `stochastic_epsilon_greedy_eps0.2` | 0.012552 | 0.011485 | 7.599e-06 | 48.8% |
| `stochastic_exploration_bonus_c1.0` | 0.012958 | 0.012003 | 7.824e-06 | 46.2% |
| `uniform_positive_fragility` | 0.013013 | 0.011642 | 7.855e-06 | 48.5% |
| `stochastic_exploration_bonus_c0.5` | 0.013088 | 0.013153 | 7.897e-06 | 46.1% |
| `stochastic_exploration_bonus_c0.25` | 0.013196 | 0.012938 | 7.957e-06 | 46.0% |
| `stochastic_normalized_fragility` | 0.013369 | 0.012865 | 8.053e-06 | 45.4% |

On the all-budget AUC summary, `uniform_step_balanced` remains best. However, this aggregate is not the only relevant view, because early budgets and the full-reveal endpoint can mask behavior in the budget range we may actually care about.

## Budget-Window Results

For higher budgets, the stochastic policies look better relative to `uniform_step_balanced`. The table below excludes the full-reveal budget `1798`, where all policies tie because all fitted rows have been revealed.

The `avg mean diff` column is the average of:

`policy distance - uniform_step_balanced distance`

Negative values mean lower error than the uniform baseline.

### Budgets `>= 900` and `< 1798`

| policy | avg win | min win | max win | avg mean diff |
| --- | ---: | ---: | ---: | ---: |
| `stochastic_epsilon_greedy_eps0.2` | 56.3% | 40.0% | 70.0% | -1.857e-07 |
| `uniform_positive_fragility` | 56.3% | 36.7% | 70.0% | -4.143e-07 |
| `stochastic_exploration_bonus_c0.25` | 54.0% | 33.3% | 70.0% | -2.304e-07 |
| `stochastic_exploration_bonus_c1.0` | 53.3% | 33.3% | 70.0% | -2.726e-07 |
| `stochastic_normalized_fragility` | 52.0% | 40.0% | 63.3% | -2.010e-07 |
| `stochastic_exploration_bonus_c0.5` | 51.7% | 36.7% | 70.0% | -2.532e-07 |

### Budgets `>= 1215` and `< 1798`

| policy | avg win | min win | max win | avg mean diff |
| --- | ---: | ---: | ---: | ---: |
| `uniform_positive_fragility` | 57.2% | 46.7% | 66.7% | -4.838e-07 |
| `stochastic_epsilon_greedy_eps0.2` | 56.2% | 40.0% | 70.0% | -3.431e-07 |
| `stochastic_exploration_bonus_c0.5` | 52.6% | 40.0% | 70.0% | -2.823e-07 |
| `stochastic_exploration_bonus_c1.0` | 52.6% | 33.3% | 66.7% | -2.779e-07 |
| `stochastic_exploration_bonus_c0.25` | 52.3% | 33.3% | 63.3% | -2.255e-07 |
| `stochastic_normalized_fragility` | 51.5% | 40.0% | 63.3% | -1.533e-07 |

In these higher-budget windows, `uniform_positive_fragility` is the strongest-looking diagnostic variant. It has the best average mean-distance improvement in both windows and the best average win rate in the `>= 1215` window.

## Representative Budget-Level Win Rates

Each cell is the paired win rate against `uniform_step_balanced` at that budget, across 30 reveal seeds. A win means the policy had lower squared Wasserstein-2 error than the baseline for the same seed and budget.

| budget | eps0.2 | bonus0.25 | bonus0.5 | bonus1.0 | stoch_norm | uniform_positive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 90 | 13% | 30% | 37% | 37% | 33% | 13% |
| 225 | 67% | 40% | 50% | 53% | 47% | 50% |
| 450 | 43% | 40% | 60% | 43% | 43% | 50% |
| 900 | 57% | 70% | 63% | 70% | 53% | 70% |
| 1215 | 50% | 53% | 47% | 53% | 50% | 67% |
| 1350 | 40% | 43% | 43% | 43% | 47% | 63% |
| 1485 | 43% | 63% | 47% | 37% | 50% | 60% |
| 1620 | 60% | 57% | 57% | 60% | 57% | 63% |
| 1710 | 53% | 57% | 40% | 33% | 60% | 47% |
| 1755 | 57% | 47% | 53% | 60% | 50% | 53% |

## Interpretation

The all-budget AUC summary says that `uniform_step_balanced` remains the best aggregate policy in this run. That statement is true, but incomplete.

For the higher-budget region, especially budgets above roughly 900 or 1215, stochastic fragility policies often beat the uniform baseline. The strongest evidence in this ablation is for `uniform_positive_fragility`, not `stochastic_normalized_fragility`.

That suggests the useful part of the fragility signal may be:

- identifying which steps have positive finite LOO fragility, and
- avoiding deterministic over-selection of the single largest noisy score,

rather than treating the raw fragility magnitudes as calibrated sampling probabilities.

The result is therefore consistent with the hypothesis that raw fragility magnitudes are noisy or poorly calibrated, even if the sign or broad ranking of fragility contains useful information.

## Caveats

- This is still an empirical ablation on the OC3 DoS `P(success)` submodel, not a proof of optimal active learning.
- The full-data reference is not ground truth; it is the full fitted-row exchangeable reference distribution.
- Budget-window conclusions depend on which budgets are operationally relevant.
- The full-reveal budget `1798` should not be used to distinguish policies, because all policies reveal all rows by that point.
- LOO fragility is an approximation here, with `max_loo_terms_per_step = 20`.
- The stochastic policies share hidden reveal orders by seed, which makes paired comparisons more meaningful but does not remove Monte Carlo uncertainty.
