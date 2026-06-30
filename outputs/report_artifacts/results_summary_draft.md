# Results Summary Draft

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

In these development outputs, the lowest average AUC in the combined main table is `exploration_bonus_c1.0` with average AUC `0.0210991`. The uniform baseline average AUC is `0.0294275`.

This should be read cautiously because the combined table uses different source runs for some policies. It is useful for report drafting and diagnostics, not a replacement for a single final run with all selected settings.

## Concentration Result

The uniform baseline remains step-balanced by construction. Fragility-guided policies concentrate allocation more strongly, especially pure greedy and lower exploration-bonus `c` values.

At budget 1200 in the exploration-bonus sensitivity run, the lowest concentration among the tested exploration-bonus settings was `c=1.0` with mean L1 imbalance `537`. This is still much more concentrated than the uniform baseline, whose L1 imbalance is zero or near zero at the same budget.

## Exploration-Bonus Sensitivity

The exploration-bonus sensitivity run tested `c` values 0.25, 0.5, and 1.0. In this development run, `c=1.0` had the lowest average AUC among those settings. Increasing `c` reduced concentration in the generated sensitivity table.

This suggests that stronger exploration pressure may be useful for controlling the severe over-concentration seen in pure greedy LOO fragility. It does not prove that `c=1.0` is prospectively optimal.

## Caveats

All fragility-guided policies summarized here used approximate LOO fragility with `max_loo_terms_per_step=20`. Exact LOO remains the mathematical v8 definition.

The full-data reference is not ground truth. It is the output distribution from all valid SOTA LLM elicitation rows under the exchangeable nodewise mixture approximation.

The budget unit is an LLM elicitation draw, not a human expert.

The experiment targets the OC3 DoS `P(success)` submodel, not full total-risk uncertainty.

LOO fragility measures finite-sample instability of the current nodewise mixture. It is not a value-of-information estimate, not optimal active learning, and not Bayesian experimental design.

The `v8_all_policies_dev` run reports `settings_reduced_from_requested=True`. Any final report result should use a predeclared final configuration or clearly describe any runtime-driven reductions.
