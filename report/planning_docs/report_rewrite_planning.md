# Report Rewrite Planning Pack

Status: planning document, not report prose.

Purpose: help decide what to write, what to show, and what to leave out. This is intended as an evidence map and report-design aid, not a draft that should be copied wholesale into the report.

## 1. Recommended Framing

### Central Question

The cleanest central question is:

> How useful is LOO output fragility as a budget-allocation signal for recovering the full-data OC3 DoS `P(success)` reference distribution, and under what sampling rules does it help?

This is broader and more defensible than asking whether one named policy is "best". It lets the report document learning across the full sequence of experiments.

### Main Empirical Lesson

The current evidence points toward this interpretation:

- LOO fragility contains some useful signal.
- Deterministic or too-sharp use of fragility is fragile in the bad sense: it can over-concentrate and perform poorly.
- Randomized use of positive fragility is more promising.
- Raw fragility magnitudes do not look well calibrated as direct sampling probabilities.
- Softmax temperature tuning improves some magnitude-weighted variants, but does not clearly beat uniform sampling over positive-fragility steps.
- Policy performance is budget-dependent, so the most informative presentation is a performance frontier over budgets, not only a single AUC ranking.

### Tone

The report should mainly document what was learned. It can still be paper-like if it is structured around questions and evidence rather than around a diary of what happened.

Suggested tone:

- "We evaluate..." not "we prove..."
- "This suggests..." not "this establishes..."
- "Exploratory ablation" for the stochastic and softmax follow-up runs.
- "Empirical frontier" or "budget-dependent policy performance" rather than "the optimal policy".

## 2. Narrative Structure Options

### Option A: Chronological Narrative

Structure:

1. Initial deterministic and stochastic policy suite.
2. Stochasticized fragility ablation.
3. Softmax temperature ablation.

Advantages:

- Honest about the research process.
- Easy to explain why later ablations were run.
- Useful for an internal or semi-technical audience.

Disadvantages:

- Can feel like a lab notebook.
- Risks overemphasizing historical false starts.
- Harder to make the final report feel like a paper.

### Option B: Question-Driven Narrative

Structure:

1. Does naive LOO-fragility allocation help?
2. Does deterministic argmax over-exploit noisy fragility?
3. Are fragility magnitudes calibrated enough to use as probabilities?
4. What does the budget-dependent policy frontier look like?

Advantages:

- More paper-like.
- Turns the sequence of experiments into a coherent empirical argument.
- Makes it easier to decide which figures are core.

Disadvantages:

- Requires careful signposting so the reader understands the experiment chronology.
- `stochastic_normalized_fragility` appears in more than one role.

### Option C: Main Result Plus Ablations

Structure:

1. Present the strongest current family of policies and the budget frontier.
2. Then present deterministic, stochastic, and softmax ablations as explanations.

Advantages:

- Efficient if the report needs to be short.
- Foregrounds the actionable empirical result.

Disadvantages:

- Less transparent about how the conclusions were reached.
- Could feel too polished relative to the exploratory nature of the work.

### Recommendation

Use Option B.

The report can still briefly say that the experiments were run sequentially, but the top-level results should be organized by the questions they answer. A short "Experiment sequence" table can preserve chronology without making chronology the main structure.

## 3. Where `stochastic_normalized_fragility` Belongs

`stochastic_normalized_fragility` is a bridge policy.

It belongs in:

- Experiment 1, because it was included in the dense policy suite and is the only policy in that suite that clearly changed the story relative to deterministic fragility.
- Experiment 2, because it is the reference proportional-fragility stochastic policy being ablated.
- Experiment 3, because softmax temperature is an attempt to improve the weighting rule inside the same conceptual family.

How to describe it:

- In Experiment 1: "The unexpectedly interesting result was not the deterministic fragility family, but the stochastic proportional-fragility variant."
- In Experiment 2: "We then asked whether proportional weighting by raw fragility was necessary."
- In Experiment 3: "We then asked whether the raw magnitudes could be made more usable by softmax temperature scaling."

Avoid treating it as a single experiment endpoint. It is better understood as the policy that exposed the calibration question.

## 4. Experiment Inventory

All three dense experiments share these core settings:

| setting | value |
| --- | --- |
| SOTA fitted rows | 1798 |
| initial seed allocation | 45 rows, 5 per MITRE step |
| reveal seeds | 30, from 101 to 3030 |
| budgets | 40 budgets: 45, 90, ..., 1755, 1798 |
| reference samples | 50000 |
| budget samples | 20000 |
| main W2 grid | 501 |
| fragility samples | 200 |
| fragility grid | 101 |
| max LOO terms per step | 20 |
| fragility recompute cadence | every 10 reveals |
| comparison baseline | `uniform_step_balanced` |

### Experiment 1: Dense Policy Suite

Output prefix:

`outputs/repeated_policy_experiment/dense_policy_suite_30_seeds_recompute10_nsamples200_*`

Policy set:

| policy | role |
| --- | --- |
| `uniform_step_balanced` | main baseline |
| `greedy_loo_fragility` | deterministic argmax fragility |
| `epsilon_greedy_eps0.2` | deterministic fragility with balanced exploration |
| `exploration_bonus_c0.25` | deterministic acquisition score with weak balance bonus |
| `exploration_bonus_c0.5` | deterministic acquisition score with medium balance bonus |
| `exploration_bonus_c1.0` | deterministic acquisition score with stronger balance bonus |
| `stochastic_normalized_fragility` | samples proportional to positive fragility |

Question answered:

- Does the original deterministic fragility family work when tested densely with 30 seeds?
- Does stochastic proportional fragility look different from deterministic argmax-style use?

Main result:

- `uniform_step_balanced` is best by all-budget AUC.
- `stochastic_normalized_fragility` is the only non-baseline policy that looks competitive in the higher-budget region.
- Deterministic argmax-style policies perform poorly overall, especially greedy and exploration-bonus variants.

AUC ranking:

| policy | avg_auc | median_auc | mean win vs uniform |
| --- | ---: | ---: | ---: |
| `uniform_step_balanced` | 0.011815 | 0.010616 |  |
| `stochastic_normalized_fragility` | 0.013369 | 0.012865 | 45.4% |
| `epsilon_greedy_eps0.2` | 0.019059 | 0.016813 | 34.6% |
| `exploration_bonus_c1.0` | 0.041150 | 0.034017 | 15.7% |
| `greedy_loo_fragility` | 0.052392 | 0.039662 | 12.1% |
| `exploration_bonus_c0.5` | 0.055868 | 0.043951 | 9.0% |
| `exploration_bonus_c0.25` | 0.057834 | 0.052389 | 10.6% |

Useful budget-window diagnostic:

| policy | avg win vs uniform, budgets >=900 and <1798 | avg win vs uniform, budgets >=1215 and <1798 |
| --- | ---: | ---: |
| `stochastic_normalized_fragility` | 52.0% | 51.5% |
| `epsilon_greedy_eps0.2` | 42.7% | 44.6% |
| `exploration_bonus_c1.0` | 20.3% | 24.4% |
| `greedy_loo_fragility` | 15.3% | 21.0% |
| `exploration_bonus_c0.25` | 14.0% | 17.7% |
| `exploration_bonus_c0.5` | 12.8% | 16.2% |

Interpretation:

- Deterministic selection based on the largest fragility score looks too brittle.
- The proportionally stochastic policy is much more interesting than deterministic fragility.
- This motivates asking whether randomization, rather than magnitude-weighting, is the important ingredient.

### Experiment 2: Stochastic Fragility Ablation

Output prefix:

`outputs/repeated_policy_experiment/dense_stochastic_ablation_30_seeds_recompute10_nsamples200_*`

Policy set:

| policy | role |
| --- | --- |
| `uniform_step_balanced` | main baseline |
| `stochastic_normalized_fragility` | original proportional positive-fragility policy |
| `uniform_positive_fragility` | samples uniformly among positive-fragility steps |
| `stochastic_epsilon_greedy_eps0.2` | mixture of balanced exploration and stochastic proportional fragility |
| `stochastic_exploration_bonus_c0.25` | samples proportional to fragility plus weak balance bonus |
| `stochastic_exploration_bonus_c0.5` | samples proportional to fragility plus medium balance bonus |
| `stochastic_exploration_bonus_c1.0` | samples proportional to fragility plus stronger balance bonus |

Question answered:

- Is the useful part of fragility the raw magnitude, or just identifying plausible positive-fragility steps and randomizing?

Main result:

- All stochasticized variants are far more competitive than deterministic variants.
- `uniform_positive_fragility` is the most interesting diagnostic result.
- `uniform_positive_fragility` often outperforms `stochastic_normalized_fragility`, suggesting raw fragility magnitudes are poorly calibrated as probabilities.

AUC ranking:

| policy | avg_auc | median_auc | mean win vs uniform |
| --- | ---: | ---: | ---: |
| `uniform_step_balanced` | 0.011815 | 0.010616 |  |
| `stochastic_epsilon_greedy_eps0.2` | 0.012552 | 0.011485 | 48.8% |
| `stochastic_exploration_bonus_c1.0` | 0.012958 | 0.012003 | 46.2% |
| `uniform_positive_fragility` | 0.013013 | 0.011642 | 48.5% |
| `stochastic_exploration_bonus_c0.5` | 0.013088 | 0.013153 | 46.1% |
| `stochastic_exploration_bonus_c0.25` | 0.013196 | 0.012938 | 46.0% |
| `stochastic_normalized_fragility` | 0.013369 | 0.012865 | 45.4% |

Budget-window diagnostic:

| policy | avg win vs uniform, budgets >=900 and <1798 | avg win vs uniform, budgets >=1215 and <1798 |
| --- | ---: | ---: |
| `uniform_positive_fragility` | 56.3% | 57.2% |
| `stochastic_epsilon_greedy_eps0.2` | 56.3% | 56.2% |
| `stochastic_exploration_bonus_c0.25` | 54.0% | 52.3% |
| `stochastic_exploration_bonus_c1.0` | 53.3% | 52.6% |
| `stochastic_normalized_fragility` | 52.0% | 51.5% |
| `stochastic_exploration_bonus_c0.5` | 51.7% | 52.6% |

Interpretation:

- Randomization matters.
- Magnitude weighting by raw positive fragility is not clearly better than simply using positive fragility as an eligibility filter.
- The apparent signal is closer to "this step is currently fragile" than "this step is 3.7 times as important as that one".

### Experiment 3: Softmax Temperature Ablation

Output prefix:

`outputs/repeated_policy_experiment/dense_softmax_temperature_ablation_30_seeds_recompute10_nsamples200_*`

Policy set:

| policy | role |
| --- | --- |
| `uniform_step_balanced` | main baseline |
| `stochastic_normalized_fragility` | proportional positive-fragility reference |
| `uniform_positive_fragility` | high-temperature / magnitude-agnostic reference |
| `softmax_normalized_fragility_temp0.25` | sharp softmax, closer to greedy |
| `softmax_normalized_fragility_temp0.5` | moderately sharp softmax |
| `softmax_normalized_fragility_temp1.0` | median-scaled natural temperature |
| `softmax_normalized_fragility_temp2.0` | flatter softmax |
| `softmax_normalized_fragility_temp4.0` | high-temperature softmax, closer to uniform-positive |

Question answered:

- Can we rescue magnitude-weighted fragility sampling by applying a softmax transformation with temperature?

Main result:

- Very sharp softmax is bad.
- Higher temperatures are better than low temperatures.
- `temp1.0` and `temp2.0` look best by late-budget win rates.
- `temp4.0` has the best softmax all-budget AUC.
- None of the softmax variants clearly beats `uniform_positive_fragility` in the budget-window diagnostics.

AUC ranking:

| policy | avg_auc | median_auc | mean win vs uniform |
| --- | ---: | ---: | ---: |
| `uniform_step_balanced` | 0.011815 | 0.010616 |  |
| `softmax_normalized_fragility_temp4.0` | 0.012585 | 0.012098 | 47.2% |
| `softmax_normalized_fragility_temp2.0` | 0.012858 | 0.012411 | 47.2% |
| `uniform_positive_fragility` | 0.013013 | 0.011642 | 48.5% |
| `stochastic_normalized_fragility` | 0.013369 | 0.012865 | 45.4% |
| `softmax_normalized_fragility_temp1.0` | 0.013500 | 0.013373 | 45.4% |
| `softmax_normalized_fragility_temp0.5` | 0.014137 | 0.012361 | 43.6% |
| `softmax_normalized_fragility_temp0.25` | 0.016917 | 0.013854 | 36.3% |

Budget-window diagnostic:

| policy | avg win vs uniform, budgets >=900 and <1798 | avg win vs uniform, budgets >=1215 and <1798 |
| --- | ---: | ---: |
| `uniform_positive_fragility` | 56.3% | 57.2% |
| `softmax_normalized_fragility_temp2.0` | 52.7% | 54.1% |
| `stochastic_normalized_fragility` | 52.0% | 51.5% |
| `softmax_normalized_fragility_temp4.0` | 51.8% | 50.3% |
| `softmax_normalized_fragility_temp1.0` | 51.0% | 54.4% |
| `softmax_normalized_fragility_temp0.5` | 48.2% | 47.2% |
| `softmax_normalized_fragility_temp0.25` | 43.2% | 45.9% |

Interpretation:

- Softmax temperature confirms the calibration story.
- Lower temperatures over-exploit noisy fragility magnitude.
- Higher temperatures move toward `uniform_positive_fragility`.
- The best softmax variants are competitive, but not a clear improvement over the simpler uniform-positive rule.

## 5. Policy Taxonomy

This taxonomy may be useful in the Methods or Results section.

### Baseline

`uniform_step_balanced`

- Chooses among available MITRE-step inputs with the smallest current revealed count.
- Keeps allocation balanced by construction.
- Main comparator for all win-rate calculations.

### Deterministic Fragility Policies

`greedy_loo_fragility`

- Chooses the step with largest finite LOO fragility.
- Tests the naive "use fragility as an acquisition score" idea.
- Performs poorly in dense runs.

`epsilon_greedy_eps0.2`

- Usually chooses the largest-fragility step, but sometimes explores using the balanced baseline.
- Less brittle than pure greedy, but still not competitive with stochasticized variants.

`exploration_bonus_c0.25`, `exploration_bonus_c0.5`, `exploration_bonus_c1.0`

- Deterministic acquisition score combining fragility with under-sampling bonus.
- In the dense 30-seed run, these perform poorly overall.
- Useful mainly as evidence that deterministic acquisition-score maximization is not enough.

### Stochastic Fragility Policies

`stochastic_normalized_fragility`

- Samples among positive-fragility steps with probability proportional to positive fragility.
- Important bridge policy.
- Shows that randomized fragility can be more promising than deterministic fragility.

`uniform_positive_fragility`

- Samples uniformly among steps with finite positive fragility.
- Fallback: balanced uniform if no positive fragility exists.
- Strongest simple diagnostic policy in the stochastic and softmax ablations.

`stochastic_epsilon_greedy_eps0.2`

- With probability 0.2, uses balanced exploration.
- With probability 0.8, samples proportional to positive fragility.
- Competitive, especially in Experiment 2.

`stochastic_exploration_bonus_c*`

- Samples proportional to a fragility-plus-balance acquisition score.
- More competitive than deterministic exploration-bonus policies.
- Does not obviously dominate `uniform_positive_fragility`.

### Softmax Calibration Policies

`softmax_normalized_fragility_temp*`

- Samples positive-fragility steps using median-scaled softmax weights.
- Temperature is dimensionless.
- Low temperature approaches greedy behavior.
- High temperature approaches uniform-positive behavior.
- Empirically, too-low temperature is poor; higher temperatures are better but not clearly better than uniform-positive.

## 6. Metrics and How to Present Them

### Primary Error Metric

Squared Wasserstein-2 distance to the full-data exchangeable reference distribution.

Use this as the main recovery-error metric.

### AUC

Use AUC as a useful aggregate diagnostic, but avoid making it the only result.

Why:

- AUC depends on the chosen budget grid.
- It can hide budget-dependent reversals.
- It includes early budgets and the full-reveal endpoint unless carefully defined.

Possible use:

- One compact table of average AUC by policy.
- Maybe one AUC figure in appendix or a summary figure in the main text.

### Paired Win Rate vs Uniform

Use paired win rate vs `uniform_step_balanced` as a budget-level diagnostic.

Why:

- It is easy to interpret.
- It respects paired reveal seeds.
- It highlights budget-dependent behavior.

Limitations:

- It ignores magnitude of improvement.
- A 51% win rate with tiny improvements is different from 51% with large improvements.

Pair with:

- Mean policy-minus-uniform distance at each budget.

### Policy-Minus-Uniform Mean Difference

Use mean difference in squared W2:

`policy distance - uniform distance`

Negative means the policy is better than uniform on average at that budget.

This should accompany win-rate plots/tables.

### Budget-Dependent Policy Frontier

This is likely central for the rewritten report.

Possible definitions:

- Best policy by mean distance at each budget.
- Best policy by paired win rate at each budget, with mean distance difference as a secondary column.
- Set of policies within a tolerance of the best policy at each budget.

Recommendation:

- Use a "best by budget" figure/table as a descriptive frontier.
- Be explicit that selecting the best policy per budget using these same data is exploratory. If a practitioner knows their budget in advance, this analysis suggests budget-dependent policy choice may matter.

### Concentration Diagnostics

Use as secondary diagnostics, especially to explain why deterministic policies fail.

Potential metrics:

- L1 imbalance from perfect balance by budget.
- min/max revealed rows per step.
- max/min revealed-row ratio.
- selected step counts by policy.

Main role:

- Explain over-concentration.
- Support the claim that deterministic argmax-style policies can over-exploit noisy fragility.

### Runtime Diagnostics

Probably appendix only.

Use if you want to discuss practicality:

- Fragility recomputation count.
- Average recomputation time.
- Total runtime.
- LOO terms used fraction.

## 7. Candidate Figures

This section lists everything you could show. "Core" means likely main-text figure. "Appendix" means useful but not necessary. "Optional" means include only if it clarifies a specific claim.

### Figure 1: Schematic of Retrospective Reveal Protocol

Priority: Core or optional.

What it shows:

- Full SOTA fitted dataset exists but is hidden.
- Initial 45-row seed allocation.
- Policy chooses MITRE step.
- Hidden reveal order returns next row for that step.
- Current mixture is evaluated against full-data reference.

Claim supported:

- The experiment is a retrospective budget-recovery design, not online elicitation.

Source:

- Would need to be drawn manually or with a simple script.

Notes:

- This could replace a lot of explanatory text.

### Figure 2: Error vs Budget, Experiment 1

Priority: Core if discussing deterministic policies in main text; otherwise appendix.

What it shows:

- Mean squared W2 error by budget for dense policy suite.

Claim supported:

- Deterministic fragility policies do not dominate uniform.
- `stochastic_normalized_fragility` is the only policy in the initial suite that looks competitive.

Source:

- `dense_policy_suite_30_seeds_recompute10_nsamples200_policy_summary_by_budget.csv`

Potential issue:

- Seven lines may be too many. Consider splitting deterministic and stochastic/reference policies.

### Figure 3: Error vs Budget, Stochastic Ablation

Priority: Core.

What it shows:

- Mean squared W2 error by budget for stochasticized policies.

Claim supported:

- Stochasticization makes fragility-guided allocation more competitive.
- `uniform_positive_fragility` is competitive with or better than proportional fragility over many budgets.

Source:

- `dense_stochastic_ablation_30_seeds_recompute10_nsamples200_policy_summary_by_budget.csv`

Potential issue:

- Seven lines. Could show only `uniform_step_balanced`, `stochastic_normalized_fragility`, `uniform_positive_fragility`, and one or two stochastic bonus/epsilon variants.

### Figure 4: Softmax Temperature Curves

Priority: Core if softmax gets a full results subsection.

What it shows:

- Error vs budget for `uniform_step_balanced`, `uniform_positive_fragility`, `stochastic_normalized_fragility`, and softmax temperatures.

Claim supported:

- Low temperatures are too sharp.
- Higher temperatures improve the softmax family.
- Uniform-positive remains hard to beat.

Source:

- `dense_softmax_temperature_ablation_30_seeds_recompute10_nsamples200_policy_summary_by_budget.csv`

Potential issue:

- Many lines. Consider a focused temperature figure excluding deterministic policies.

### Figure 5: Win Rate vs Uniform by Budget

Priority: Core.

What it shows:

- Paired win fraction against uniform across budgets.
- Could be one panel per experiment or one focused panel for the most relevant policies.

Claim supported:

- Performance is budget-dependent.
- AUC alone is not sufficient.
- There are budget regions where fragility-guided policies outperform uniform more often than not.

Source:

- `*_win_rate_by_budget.csv`

Recommended version:

- Main text: show a combined plot with `stochastic_normalized_fragility`, `uniform_positive_fragility`, `softmax_temp1.0`, `softmax_temp2.0`, `softmax_temp4.0`.
- Appendix: full version with all policies.

### Figure 6: Policy-Minus-Uniform Distance by Budget

Priority: Core or appendix.

What it shows:

- Mean difference in squared W2 error relative to uniform.
- Negative is better than uniform.

Claim supported:

- Complements win rate by showing magnitude.

Source:

- `*_win_rate_by_budget.csv` contains `mean_policy_minus_baseline_distance`.

Recommended version:

- Pair with win-rate plot, possibly as two stacked panels.

### Figure 7: Budget-Dependent Empirical Frontier

Priority: Core.

What it shows:

- For each budget, the empirically best policy or top policy set.
- Could be a colored strip over budget.
- Could be based on mean distance or paired win rate.

Claim supported:

- There is no single universally dominant fragility policy.
- Budget-dependent policy choice is a natural way to interpret the results.

Source:

- Results or summary CSVs from all dense experiments.

Recommended caution:

- Label as "empirical frontier" or "exploratory best-by-budget summary", not "oracle policy" unless explicitly framed as post hoc.

### Figure 8: AUC Ranking Bar Chart

Priority: Appendix or compact main summary.

What it shows:

- Average AUC by policy.

Claim supported:

- Uniform is strongest by all-budget AUC.
- Softmax high temperatures improve over low temperatures.
- Deterministic policies perform badly.

Source:

- `*_policy_auc_by_seed.csv`

Potential issue:

- AUC can obscure budget dependence. Do not let this be the only main figure.

### Figure 9: Concentration vs Budget

Priority: Appendix or explanatory main figure.

What it shows:

- Step-count imbalance by budget.

Claim supported:

- Deterministic fragility policies over-concentrate.
- Stochasticization and uniform-positive reduce pathological concentration.

Source:

- `*_concentration_by_budget.csv`

Recommended use:

- Good supporting figure for "why deterministic fragility fails".

### Figure 10: Selected Step Counts by Policy

Priority: Appendix.

What it shows:

- Which MITRE steps each policy selected.

Claim supported:

- Fragility-guided policies focus on particular steps.
- Over-concentration is visible in allocation patterns.

Source:

- `*_selected_step_counts.csv`

Potential issue:

- Step names are long. Use horizontal bars or grouped summaries.

### Figure 11: Softmax Temperature Summary

Priority: Core or appendix.

What it shows:

- Temperature on x-axis, performance metric on y-axis.
- Could use AUC, mean win rate over all budgets, or average best-window performance.

Claim supported:

- Performance improves as softmax becomes less sharp up to a point.
- Too-low temperature is bad.

Source:

- `dense_softmax_temperature_ablation_30_seeds_recompute10_nsamples200_policy_auc_by_seed.csv`
- `dense_softmax_temperature_ablation_30_seeds_recompute10_nsamples200_win_rate_by_budget.csv`

Recommended version:

- One panel for AUC by temperature.
- One panel for win rate by budget or average win in chosen budget group.

### Figure 12: LOO Approximation Audit

Priority: Appendix.

What it shows:

- Cap 20 vs exact LOO agreement.

Claim supported:

- The LOO approximation is reasonable enough for a pilot but not exact.

Source:

- `outputs/fragility_approximation_audit/loo_approximation_audit_report.json`

Could be a table instead.

## 8. Candidate Tables

### Table 1: Experiment Scope and Data

Priority: Core.

Possible rows:

- OC3 DoS only.
- SOTA only.
- `P(success)` submodel.
- 3600 raw rows.
- 3587 usable rows.
- 1798 usable SOTA rows.
- 9 MITRE steps.
- 5 LLM forecaster models.
- Initial budget 45.

Source:

- `outputs/sanity_checks/cleaning_report.json`

### Table 2: Beta Fit and Forward Model Sanity Checks

Priority: Appendix or compact methods table.

Possible rows:

- 1798 SOTA fitted rows.
- 1790 ok fits.
- 8 warn fits.
- 0 failed fits.
- median RMSE 0.00222.
- p95 RMSE 0.00688.
- full-reference mean `P(success)` 0.2244.
- full-reference median `P(success)` 0.2197.

Source:

- `outputs/fitted_distributions/sota_beta_fit_report.json`
- `outputs/forward_model_smoke_tests/full_reference_p_success_summary.json`

### Table 3: Policy Definitions

Priority: Core.

Columns:

- Policy name.
- Family.
- Selection rule.
- What question it tests.
- Which experiment(s) include it.

This table will help readers navigate the many policies.

### Table 4: Experiment Inventory

Priority: Core.

Columns:

- Experiment.
- Output prefix.
- Policies.
- Seeds.
- Budgets.
- Main question.

This prevents the results section from becoming confusing.

### Table 5: AUC Ranking by Experiment

Priority: Core or appendix.

Rows:

- One table per experiment, or a compact combined table.

Use:

- Show all-budget aggregate.

Caution:

- Explain that AUC is not the full story.

### Table 6: Budget-Window Diagnostics

Priority: Optional.

Rows:

- Average win rate vs uniform and average mean difference for selected budget windows.

Caution:

- Since you no longer want to restrict to fixed windows, this should be used as a diagnostic, not the central framing.

Better main alternative:

- Budget frontier table/figure.

### Table 7: Best Policy by Budget

Priority: Core.

Possible columns:

- Budget.
- Best policy by mean squared W2.
- Best policy by win rate vs uniform.
- Mean distance difference from uniform.
- Win rate vs uniform.
- Runner-up / near-tie policies.

Why it matters:

- Directly addresses the idea that methodology could be budget-dependent.

Caution:

- This is exploratory if the same data are used to select the best policy per budget.

### Table 8: Softmax Temperature Summary

Priority: Core if softmax is a main result.

Columns:

- Temperature.
- Average AUC.
- Average win rate vs uniform.
- Average win rate over budgets where the softmax family is competitive.
- Average mean difference vs uniform.

Main use:

- Shows that lower temperatures are too sharp and higher temperatures are better.

### Table 9: LOO Approximation Audit

Priority: Appendix.

Rows:

- cap 10.
- cap 20.

Columns:

- top1 match rate.
- mean top3 Jaccard.
- mean Spearman.
- warning flag.

Values:

| cap | top1 match | mean top3 Jaccard | mean Spearman | flag |
| ---: | ---: | ---: | ---: | --- |
| 10 | 0.833 | 0.583 | 0.547 | questionable |
| 20 | 1.000 | 0.833 | 0.833 | appears acceptable |

### Table 10: Runtime and Practicality

Priority: Appendix.

Possible rows:

- Experiment 1 runtime: 26364.5 seconds.
- Experiment 2 runtime: 23772.6 seconds.
- Experiment 3 runtime: 27879.1 seconds.

Use:

- Practical note: LOO fragility is expensive; approximations and recompute cadence matter.

## 9. Possible Final Report Structure

This is a structure, not prose.

### Abstract

Purpose:

- One-sentence problem.
- Retrospective budget-recovery setup.
- Main finding: deterministic fragility fails; stochastic positive-fragility helps in budget-dependent ways; raw magnitudes are poorly calibrated.

Do not overclaim:

- Avoid saying "we found the optimal policy".

### 1. Introduction

Content:

- Why elicitation budget allocation matters.
- Why output fragility is a tempting signal.
- Why retrospective budget recovery is a reasonable pilot.
- Main questions:
  - Does fragility help?
  - Does deterministic argmax over-exploit?
  - Are fragility magnitudes calibrated?
  - How does performance vary by budget?

### 2. Dataset and Model

Content:

- OC3 DoS SOTA `P(success)` scope.
- Data size and cleaning.
- Beta fitting.
- Forward model formula.
- Full-data exchangeable reference.

Keep:

- Mostly methods and factual setup.

Move to appendix:

- Detailed row counts by MITRE step.
- Worst Beta fits.

### 3. Retrospective Budget-Recovery Design

Content:

- Initial seed allocation.
- Hidden reveal orders.
- Budget grid.
- Evaluation metric.
- Paired seeds.
- LOO fragility definition.
- Approximate LOO details.

Important:

- This should be clear enough that the policy comparison is credible.

### 4. Policy Families

Content:

- Present policy taxonomy.
- Use a compact policy definition table.

Recommended grouping:

- Baseline.
- Deterministic fragility.
- Stochastic fragility.
- Softmax calibration.

### 5. Results: Does Naive Fragility Allocation Help?

Experiment:

- Dense policy suite.

Core evidence:

- AUC ranking.
- Error vs budget or win vs uniform.
- Concentration diagnostic, if needed.

Main message:

- Deterministic argmax-style fragility performs poorly.
- `stochastic_normalized_fragility` is the interesting exception.

### 6. Results: Is Randomization More Important Than Fragility Magnitude?

Experiment:

- Stochastic fragility ablation.

Core evidence:

- `uniform_positive_fragility` vs `stochastic_normalized_fragility` vs uniform.
- Win-rate by budget.
- Mean difference by budget.

Main message:

- Uniform over positive-fragility steps is competitive and often stronger than proportional weighting.
- Raw fragility magnitude is suspect as a probability weight.

### 7. Results: Can Softmax Temperature Calibrate the Magnitudes?

Experiment:

- Softmax temperature ablation.

Core evidence:

- Temperature summary.
- Softmax error/win curves.
- Comparison to `uniform_positive_fragility`.

Main message:

- Too-sharp softmax is bad.
- Higher temperatures improve performance.
- Softmax does not clearly beat uniform-positive.

### 8. Budget-Dependent Policy Frontier

This could also come before Sections 5-7 if you want the frontier to be the headline result.

Content:

- Best/near-best policy by budget.
- A plot or table showing how preferred policy changes with budget.

Main message:

- There is no single robustly dominant fragility policy.
- If the available budget is known, budget-dependent policy choice may be more appropriate than selecting a global winner by AUC.

Caution:

- This is an empirical frontier estimated on the same data; it should be interpreted exploratorily unless validated on another task/model.

### 9. Discussion

Content:

- What was learned about fragility:
  - Signal exists.
  - Magnitudes are not calibrated.
  - Randomization is important.
  - Deterministic maximization is risky.
- Practical implications:
  - Start with balanced uniform as a strong baseline.
  - If using fragility, prefer randomized positive-fragility policies.
  - Avoid pure greedy fragility.
- Why results are not definitive:
  - One submodel.
  - Full-data reference is not truth.
  - Exchangeable nodewise mixture approximation.
  - Post hoc policy/budget selection.
  - Approximate LOO.

### 10. Limitations

Possible subsections:

- Scope limitation: OC3 DoS SOTA `P(success)` only.
- Reference limitation: full-data reference is not ground truth.
- Mixture limitation: no cross-node forecaster coherence.
- LOO limitation: fragility is finite-sample instability, not VOI.
- Multiple comparisons / exploratory policy search.
- Runtime/practicality of LOO fragility.

### 11. Conclusion

Purpose:

- Short, sober summary.
- Emphasize learning, not final optimal method.

Possible conclusion-level claims:

- LOO fragility is not useless, but naive deterministic use is poor.
- Randomized use of positive fragility is the most promising simple direction.
- The strongest evidence is for fragility as a filter/ranking signal, not a calibrated probability magnitude.
- Future work should validate on additional submodels/tasks and consider better-calibrated acquisition functions.

## 10. Recommended Core Exhibits

If the report needs to be concise, I would choose:

1. Scope/data table.
2. Policy definition table.
3. Experiment inventory table.
4. One figure showing deterministic policy suite results or a compact AUC/win summary.
5. One figure showing stochastic ablation win rate or policy-minus-uniform by budget.
6. One figure showing softmax temperature performance.
7. One budget-dependent frontier figure/table.
8. One appendix table for LOO approximation audit.

If only three main figures are allowed:

1. Budget-dependent win-rate and mean-difference plot for key policies.
2. Softmax temperature summary plot.
3. Empirical policy frontier by budget.

If only two main tables are allowed:

1. Policy/experiment inventory table.
2. Key numerical summary table: AUC plus selected budget-frontier diagnostics.

## 11. Recommended Appendix Exhibits

Appendix A: Data cleaning and Beta fitting.

- Raw/usable rows.
- SOTA rows by MITRE step.
- Beta fit RMSE summary.

Appendix B: Forward model details.

- Full OC3 DoS `P(success)` formula.
- Variable-to-MITRE-step mapping.

Appendix C: LOO approximation audit.

- Cap 10 vs cap 20 table.
- Runtime reduction notes.

Appendix D: Full result tables.

- AUC by seed.
- Win rate by budget.
- Policy summary by budget.
- Concentration by budget.

Appendix E: Selected step counts.

- Which steps were selected by each policy.
- Useful for diagnosing concentration.

## 12. Claims: Strong, Moderate, Weak

### Strong Claims

These are well supported by the current evidence:

- Step-balanced uniform is a strong baseline.
- Deterministic greedy fragility and deterministic exploration-bonus policies perform poorly in the dense 30-seed run.
- Stochasticized fragility policies are much more competitive than deterministic argmax-style policies.
- Very sharp softmax temperature is poor relative to flatter softmax variants.
- Performance varies meaningfully by budget.

### Moderate Claims

These are plausible but should be worded carefully:

- Positive LOO fragility contains useful allocation signal.
- Raw LOO fragility magnitudes are poorly calibrated as direct sampling probabilities.
- `uniform_positive_fragility` is the best simple diagnostic policy among the stochastic ablations.
- Softmax temperature can improve magnitude-weighted fragility sampling, but not enough to clearly beat uniform-positive.

### Weak or Exploratory Claims

These need caveats:

- "The best policy at budget X is policy Y."
- "A practitioner should use a budget-dependent policy schedule."
- "Uniform-positive is generally better than softmax."
- "LOO fragility is the right active learning signal."

Better wording:

- "In this retrospective dataset, the empirical frontier suggests..."
- "This motivates testing..."
- "This is consistent with..."

### Claims to Avoid

Avoid:

- "Optimal allocation."
- "Value of information."
- "Bayesian experimental design."
- "Ground truth recovery."
- "Fragility proves which input matters most."
- "Softmax calibration failed" without qualification.

## 13. Unresolved Decisions for You

### Decision 1: Main Story Order

Recommended:

- Question-driven structure with experiment chronology as a supporting table.

Alternative:

- Chronological structure if the report is meant to document the research process more explicitly.

### Decision 2: How Central Should the Budget Frontier Be?

My recommendation:

- Make budget dependence a central result.
- Do not restrict attention to one preselected budget range.
- Show the frontier and discuss it as exploratory.

### Decision 3: How Much of Experiment 1 to Show?

Options:

- Core: show only enough to establish deterministic fragility problems and motivate stochasticization.
- Appendix: full deterministic policy results and concentration diagnostics.

Recommendation:

- Keep Experiment 1 in main text, but do not let it dominate.

### Decision 4: How Much of Softmax to Show?

Options:

- Main text: compact softmax temperature summary.
- Appendix: full softmax curves/tables.

Recommendation:

- Main text should include softmax because it directly tests the magnitude-calibration hypothesis.

### Decision 5: Whether to Make a Policy Recommendation

Options:

- Strong recommendation: use `uniform_positive_fragility`.
- Cautious recommendation: if using fragility, avoid deterministic argmax and prefer randomized positive-fragility rules.
- No recommendation: document empirical behavior only.

Recommendation:

- Use cautious recommendation.

### Decision 6: How to Handle Multiple Comparisons

Important:

- Many policies and temperatures were evaluated.
- Best-by-budget summaries are post hoc.

Recommendation:

- Acknowledge this.
- Frame later ablations as exploratory calibration/diagnostic experiments.

### Decision 7: Whether to Keep Old Locked-Run Material

The current `report/main.tex` contains older locked-run framing and report artifacts. The rewrite should probably use the three dense 30-seed experiments as the main empirical basis.

Recommendation:

- Treat older dev/locked artifacts as historical or appendix only, unless needed for methods validation.

## 14. Suggested Figure/Table Generation Work

Before writing prose, it would be useful to generate the following artifacts:

1. `policy_definitions_table.csv` or `.tex`
2. `experiment_inventory_table.csv` or `.tex`
3. `dense_all_auc_summary.csv`
4. `budget_frontier_by_mean_distance.csv`
5. `budget_frontier_by_win_rate.csv`
6. `key_policy_win_rate_by_budget.png`
7. `key_policy_mean_diff_by_budget.png`
8. `softmax_temperature_summary.png`
9. `concentration_deterministic_vs_stochastic.png`

Suggested key policies for combined plots:

- `uniform_step_balanced`
- `stochastic_normalized_fragility`
- `uniform_positive_fragility`
- `softmax_normalized_fragility_temp1.0`
- `softmax_normalized_fragility_temp2.0`
- `softmax_normalized_fragility_temp4.0`
- optionally `greedy_loo_fragility` as a cautionary deterministic contrast

## 15. Possible Main-Text Evidence Sequence

This is the sequence I would use if writing the report myself:

1. Scope and retrospective design.
2. Policy taxonomy.
3. Dense policy suite:
   - deterministic fragility mostly fails;
   - stochastic proportional fragility is the interesting exception.
4. Stochastic ablation:
   - randomization helps;
   - uniform-positive suggests magnitude calibration problem.
5. Softmax ablation:
   - temperature improves magnitude use;
   - low temperature is bad;
   - no clear win over uniform-positive.
6. Budget frontier:
   - policy choice depends on budget;
   - AUC is useful but incomplete.
7. Limitations and future work.

## 16. Possible Abstract-Level Bullet Skeleton

This is not prose, just content to include eventually:

- Problem: budgeted elicitation in structured risk models.
- Setting: OC3 DoS SOTA `P(success)` retrospective budget recovery.
- Method: compare uniform allocation with LOO-fragility-guided policies under shared hidden reveal orders.
- Experiments: deterministic suite, stochastic ablation, softmax temperature ablation.
- Findings:
  - deterministic fragility over-exploits;
  - stochastic positive-fragility policies are more competitive;
  - raw fragility magnitudes are poorly calibrated;
  - budget-dependent frontier is more informative than global AUC alone.
- Caveat: exploratory, one submodel, full-data reference not truth.

## 17. File Map

Key source files:

- `scripts/01_clean_data.py`
- `scripts/02_fit_beta_distributions.py`
- `scripts/03_smoke_test_forward_model.py`
- `scripts/03b_audit_beta_and_forward_model.py`
- `scripts/07_run_repeated_policy_experiment.py`
- `scripts/08_audit_loo_approximation.py`
- `scripts/09_make_report_artifacts.py`
- `src/saferai_budget_recovery/policies.py`
- `src/saferai_budget_recovery/experiment.py`
- `src/saferai_budget_recovery/reveal.py`

Key data/audit outputs:

- `outputs/sanity_checks/cleaning_report.json`
- `outputs/fitted_distributions/sota_beta_fit_report.json`
- `outputs/forward_model_smoke_tests/full_reference_p_success_summary.json`
- `outputs/fragility_approximation_audit/loo_approximation_audit_report.json`

Key experiment outputs:

- `outputs/repeated_policy_experiment/dense_policy_suite_30_seeds_recompute10_nsamples200_*`
- `outputs/repeated_policy_experiment/dense_stochastic_ablation_30_seeds_recompute10_nsamples200_*`
- `outputs/repeated_policy_experiment/dense_softmax_temperature_ablation_30_seeds_recompute10_nsamples200_*`

Existing current report:

- `report/main.tex`

This should probably be treated as material to mine rather than as a base to lightly edit. The new report will likely need a substantial rewrite.

## 18. Immediate Next Steps

Recommended next step before drafting:

1. Decide the top-level structure: question-driven vs chronological.
2. Decide which figures are core.
3. Generate the frontier tables/figures.
4. Generate updated report artifact tables from the three dense experiments.
5. Only then start a LaTeX outline.

My recommended first concrete work item:

- Generate a combined "key results artifacts" directory containing:
  - all-experiment AUC table;
  - budget frontier by mean distance;
  - budget frontier by win rate;
  - key-policy win-rate plot;
  - key-policy policy-minus-uniform plot;
  - softmax temperature summary.

That would make the actual writing much easier and reduce the risk of choosing figures ad hoc while drafting.
