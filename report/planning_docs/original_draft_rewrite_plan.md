# Original Draft Rewrite Plan

Status: working planning memo for discussion, not report prose.

Baseline draft: `report/main.tex`

## 1. What This Memo Is For

The aim is to help rewrite the report while preserving the parts of the original draft that were written and reviewed more carefully. This memo is meant to support your writing process, not replace it.

The report should now explain a larger empirical story than the original draft:

- The original deterministic fragility policies were not enough.
- A stochastic proportional-fragility policy changed the story.
- Follow-up ablations suggested that the useful signal may be positive fragility plus randomization, not raw fragility magnitude as a calibrated probability.
- Softmax temperature scaling partly helped magnitude-weighted sampling, but did not clearly beat the simpler uniform-positive-fragility rule.
- Performance depends strongly on the budget, so a budget-dependent view is more informative than a single average AUC ranking.

The target tone should be somewhere between an internal research report and a paper-style empirical study: transparent about the exploratory sequence, but organized around questions rather than chronology alone.

## 2. Current Draft Diagnosis

### Abstract

Current location: lines 37-39.

Keep the core setup, but the conclusion is now too vague and slightly out of date. It says the results do not clearly support "this particular fragility-guided heuristic", whereas the new evidence is more specific:

- Deterministic or too-sharp fragility allocation performs poorly.
- Stochastic use of positive fragility is more competitive.
- Raw fragility magnitudes appear poorly calibrated as direct sampling probabilities.
- The best empirical policy depends on budget.

Suggested rewrite goal:

- State the scoped retrospective setup.
- State that three dense experiment stages were run.
- State the main lesson in calibrated language: LOO fragility contains useful signal, but how the signal is converted into allocation probabilities matters.

Do not overclaim that fragility-guided allocation beats uniform overall. Uniform remains a strong baseline.

### Section 1: Introduction

Current location: lines 44-50.

Mostly keep.

What works:

- The motivation from elicitation burden is clear.
- The active-learning/experimental-design analogy is useful.
- The final paragraph correctly frames this as a budget-allocation problem.

What needs changing:

- The final paragraph should not imply that the report only compares a small set of original LOO policies.
- It should preview the empirical question more accurately: not only "does the heuristic help?", but "under what policy transformations does the fragility signal help?"

Suggested section-level role:

- Introduce the problem: limited elicitation budget in SaferAI-style quantitative risk modeling.
- Introduce the retrospective budget-recovery evaluation.
- Introduce LOO fragility as an output-sensitive heuristic.
- Preview the main empirical finding: naive deterministic use is brittle, stochastic/calibrated use is more promising, and budget-specific comparisons matter.

### Section 2: Methods

Current location: lines 52-203.

This is the strongest part of the original draft and should remain the backbone of the rewritten report. The main required work is updating the policy taxonomy and experiment settings.

#### Experiment Scope

Current location: lines 53-62.

Mostly keep.

Possible edits:

- Be consistent about whether the output is called `Successful Attack Rate` or `P(success)`.
- The phrase "This is sufficient for our objectives" may be too strong. Safer wording: this makes the first investigation tractable and isolates allocation across elicited node distributions.
- Consider adding a compact scope table here or just after the text:
  - Risk model: OC3 Denial of Service.
  - Component: `P(success)` submodel.
  - Capability level: SOTA only.
  - Budget unit: one valid SOTA LLM elicitation row.
  - Reference: full-data exchangeable nodewise mixture.

#### Fitting Elicitation Distributions

Current location: lines 64-83.

Mostly keep.

Required update:

- The old draft says "1800 rows" and notes two invalid rows. The cleaned SOTA subset used in the dense experiments has 1798 usable rows. Make this precise.
- If you want one extra reproducibility detail, mention that each valid elicitation row is fit independently as a Beta distribution by quartile matching.

Possible appendix material:

- Detailed fit-quality diagnostics.
- Invalid-row details.
- Endpoint clipping details, if relevant.

#### OC3 DoS P(success) Submodel

Current location: lines 85-124.

Keep.

This is a clear self-contained description of the forward model. It could be improved only by tightening language and maybe avoiding long display tables in the main text if space is tight.

Possible addition:

- A short sentence emphasizing that this forward model is fixed across all experiments; only the revealed nodewise mixtures change.

#### Exchangeable Nodewise Mixtures

Current location: lines 126-132.

Keep, but consider adding one limitation sentence:

- This approximation preserves nodewise marginal variation across elicitation rows, but not cross-node coherence from the same forecaster/repeat.

That point can also live in Limitations instead.

#### Full-Data Reference Distribution

Current location: lines 134-147.

Keep.

Make sure the report repeatedly distinguishes:

- full-data reference distribution,
- not ground truth,
- retrospective recovery target.

This distinction is important because the metric is recovery of a constructed full-data reference, not verification against reality.

#### Budgeted Reveal Protocol

Current location: lines 149-151.

Keep but expand slightly.

The dense experiments used the same basic paired hidden-reveal idea, but the important common settings should be stated clearly:

- initial budget 45 rows = five per node, one from each of five LLM forecaster models,
- 30 reveal seeds,
- 40 budgets: 45, 90, ..., 1755, 1798,
- policies compared under the same reveal seed share hidden within-node reveal orders,
- policies choose only the next node/step, not the identity of the row within that node.

This is a good place for a small algorithm block if you want one.

#### LOO Output Fragility

Current location: lines 153-171.

Keep.

This is mathematically central and already well explained.

Required update:

- Add that in the dense runs fragility was recomputed every 10 reveals.
- Add the Monte Carlo approximation settings either here or in an experiment settings table:
  - fragility `n_samples = 200`,
  - fragility `n_grid = 101`,
  - `max_loo_terms_per_step = 20`.

Tone caution:

- Keep saying "finite-sample instability heuristic", not "expected value of information".

#### Allocation Policies

Current location: lines 173-191.

This needs the largest Methods rewrite.

The current list only covers the original deterministic family. The report now needs a taxonomy covering all policy families, without making Methods unreadably long.

Recommended structure:

1. Define the shared ingredients:
   - eligible node set,
   - current counts `n_j`,
   - positive finite fragility scores,
   - fallback behavior when fragility is missing or non-positive.

2. Present a compact policy table in the main text.

3. Put exact formulas for the main policy classes in text or an appendix:
   - step-balanced uniform,
   - deterministic argmax fragility,
   - deterministic epsilon-greedy / exploration-bonus,
   - stochastic proportional positive fragility,
   - uniform-positive fragility,
   - stochastic epsilon/bonus variants,
   - softmax-normalized fragility.

Recommended table columns:

| family | policy names | selection rule | role in report |
| --- | --- | --- | --- |
| Baseline | `uniform_step_balanced` | balance counts across nodes | main comparator |
| Deterministic fragility | `greedy_loo_fragility`, deterministic epsilon/bonus | choose argmax-style fragility score | tests naive use of fragility |
| Stochastic proportional | `stochastic_normalized_fragility` | sample proportional to positive fragility | bridge policy |
| Uniform positive | `uniform_positive_fragility` | sample uniformly among positive-fragility nodes | tests sign/set vs magnitude |
| Stochastic bonus/epsilon | stochastic epsilon/bonus variants | randomize scores that include exploration | tests randomization with balance pressure |
| Softmax calibrated | `softmax_normalized_fragility_temp*` | median-scaled softmax over positive fragility | tests magnitude calibration |

Important policy interpretation:

- `stochastic_normalized_fragility` is the bridge policy. It belongs in all three experiment stories:
  - In Experiment 1, it is the interesting exception to deterministic fragility failure.
  - In Experiment 2, it is the proportional-fragility rule being ablated.
  - In Experiment 3, it motivates temperature-scaling the weighting rule.

#### Evaluation Metric

Current location: lines 193-203.

Keep.

Add one caution:

- AUC over budgets is useful but can hide budget-specific behavior. This motivates showing win rate and error difference by budget, not only all-budget AUC.

## 3. Results Section Replacement

Current location: lines 205-230.

This section should be almost entirely replaced.

The old Results section describes:

- 10 reveal seeds,
- sparse budget grid `[45,90,180,360,720,1200,1798]`,
- six policies,
- old conclusion that exploration-bonus `c=1.0` had lowest average AUC.

That is now superseded by the three dense 30-seed experiments.

### Recommended Results Structure

Use a question-driven structure, with a short experiment-inventory table near the start.

Suggested outline:

```text
3. Results
3.1 Common dense-run setup
3.2 Experiment 1: Does naive LOO-fragility allocation help?
3.3 Experiment 2: Is randomization more important than raw magnitude weighting?
3.4 Experiment 3: Can softmax temperature scaling improve proportional fragility sampling?
3.5 Budget-dependent policy performance
3.6 Summary of empirical lessons
```

This is more paper-like than a diary, but it still preserves the logic of the research process.

### Results Opening

Start Results with a compact table like:

| experiment | purpose | policy set | output prefix |
| --- | --- | --- | --- |
| Dense policy suite | deterministic fragility vs uniform, plus stochastic bridge | original policy suite | `dense_policy_suite_30_seeds_recompute10_nsamples200_*` |
| Stochastic ablation | test stochasticized variants and uniform-positive fragility | stochastic policy family | `dense_stochastic_ablation_30_seeds_recompute10_nsamples200_*` |
| Softmax ablation | test temperature scaling of proportional fragility weights | softmax temperatures plus references | `dense_softmax_temperature_ablation_30_seeds_recompute10_nsamples200_*` |

Then state common settings once:

| setting | value |
| --- | --- |
| reveal seeds | 30 |
| budgets | 40 budgets: 45, 90, ..., 1755, 1798 |
| reference samples | 50000 |
| budget samples | 20000 |
| main W2 grid | 501 |
| fragility samples | 200 |
| fragility grid | 101 |
| max LOO terms per step | 20 |
| fragility recomputation | every 10 reveals |
| baseline | `uniform_step_balanced` |

### Experiment 1: Dense Policy Suite

Question:

- Does the original deterministic fragility family work when tested with 30 seeds and a dense budget grid?

Policies:

- `uniform_step_balanced`
- `greedy_loo_fragility`
- `epsilon_greedy_eps0.2`
- `exploration_bonus_c0.25`
- `exploration_bonus_c0.5`
- `exploration_bonus_c1.0`
- `stochastic_normalized_fragility`

Main result:

- Uniform is best by all-budget AUC.
- Deterministic argmax-style fragility policies perform poorly overall.
- `stochastic_normalized_fragility` is the interesting exception and motivates the stochastic ablations.

Numbers to use:

| policy | avg AUC | mean win vs uniform |
| --- | ---: | ---: |
| `uniform_step_balanced` | 0.011815 | baseline |
| `stochastic_normalized_fragility` | 0.013369 | 45.4% |
| `epsilon_greedy_eps0.2` | 0.019059 | 34.6% |
| `exploration_bonus_c1.0` | 0.041150 | 15.7% |
| `greedy_loo_fragility` | 0.052392 | 12.1% |
| `exploration_bonus_c0.5` | 0.055868 | 9.0% |
| `exploration_bonus_c0.25` | 0.057834 | 10.6% |

Possible main-text figure:

- Error vs budget or policy-minus-uniform error for selected policies:
  - `uniform_step_balanced`,
  - `stochastic_normalized_fragility`,
  - `greedy_loo_fragility`,
  - `epsilon_greedy_eps0.2`,
  - maybe `exploration_bonus_c1.0`.

Do not let this experiment dominate the report. Its role is to motivate the later ablations.

### Experiment 2: Stochastic Fragility Ablation

Question:

- Is the useful part of fragility the raw magnitude, or merely identifying plausible positive-fragility nodes and randomizing among them?

Policies:

- `uniform_step_balanced`
- `stochastic_normalized_fragility`
- `uniform_positive_fragility`
- `stochastic_epsilon_greedy_eps0.2`
- `stochastic_exploration_bonus_c0.25`
- `stochastic_exploration_bonus_c0.5`
- `stochastic_exploration_bonus_c1.0`

Main result:

- Stochasticized variants are much more competitive than deterministic argmax-style policies.
- `uniform_positive_fragility` often beats proportional weighting by raw fragility.
- This suggests raw LOO fragility magnitudes are not well calibrated as sampling probabilities, even if positive fragility contains useful signal.

Numbers to use:

| policy | avg AUC | mean win vs uniform |
| --- | ---: | ---: |
| `uniform_step_balanced` | 0.011815 | baseline |
| `stochastic_epsilon_greedy_eps0.2` | 0.012552 | 48.8% |
| `stochastic_exploration_bonus_c1.0` | 0.012958 | 46.2% |
| `uniform_positive_fragility` | 0.013013 | 48.5% |
| `stochastic_exploration_bonus_c0.5` | 0.013088 | 46.1% |
| `stochastic_exploration_bonus_c0.25` | 0.013196 | 46.0% |
| `stochastic_normalized_fragility` | 0.013369 | 45.4% |

Budget-window diagnostic:

| policy | avg win vs uniform, budgets >=900 and <1798 | avg win vs uniform, budgets >=1215 and <1798 |
| --- | ---: | ---: |
| `uniform_positive_fragility` | 56.3% | 57.2% |
| `stochastic_epsilon_greedy_eps0.2` | 56.3% | 56.2% |
| `stochastic_exploration_bonus_c1.0` | about 53-54% | about 54% |
| `stochastic_normalized_fragility` | 52.0% | 51.5% |

The exact window values for the stochastic bonus variants should be pulled from the CSV if they become main-text claims.

Possible main-text figure:

- A compact plot of win rate vs budget against uniform for:
  - `stochastic_normalized_fragility`,
  - `uniform_positive_fragility`,
  - `stochastic_epsilon_greedy_eps0.2`,
  - `stochastic_exploration_bonus_c1.0`.

### Experiment 3: Softmax Temperature Ablation

Question:

- Can raw fragility magnitudes be made more useful by transforming them with a softmax temperature?

Policies:

- `uniform_step_balanced`
- `stochastic_normalized_fragility`
- `uniform_positive_fragility`
- `softmax_normalized_fragility_temp0.25`
- `softmax_normalized_fragility_temp0.5`
- `softmax_normalized_fragility_temp1.0`
- `softmax_normalized_fragility_temp2.0`
- `softmax_normalized_fragility_temp4.0`

Main result:

- Very sharp softmax is poor.
- Higher-temperature softmax variants are better.
- The best softmax variants are competitive but do not clearly beat `uniform_positive_fragility`.
- This supports the calibration story: raw magnitude has some information, but too much sharpness over-exploits it.

Numbers to use:

| policy | avg AUC | mean win vs uniform |
| --- | ---: | ---: |
| `uniform_step_balanced` | 0.011815 | baseline |
| `softmax_normalized_fragility_temp4.0` | 0.012585 | 47.2% |
| `softmax_normalized_fragility_temp2.0` | 0.012858 | 47.2% |
| `uniform_positive_fragility` | 0.013013 | 48.5% |
| `stochastic_normalized_fragility` | 0.013369 | 45.4% |
| `softmax_normalized_fragility_temp1.0` | 0.013500 | 45.4% |
| `softmax_normalized_fragility_temp0.5` | 0.014137 | 43.6% |
| `softmax_normalized_fragility_temp0.25` | 0.016917 | 36.3% |

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

Possible main-text figure:

- Softmax temperature summary:
  - x-axis: temperature,
  - y-axis: average AUC or mean paired win rate,
  - include horizontal reference lines for `uniform_step_balanced`, `uniform_positive_fragility`, and `stochastic_normalized_fragility`.

This figure would be one of the clearest ways to show the calibration story.

### Budget-Dependent Policy Performance

This should be a short but important subsection.

The user-facing point is:

- In practice, if someone knows their elicitation budget, they need the method that performs best at that budget, not the method with best average performance across all budgets.

Recommended presentation:

- Show a budget-dependent empirical frontier:
  - for each budget, which policy has the lowest mean error,
  - optionally also show the top 2-3 policies at each budget.

Alternative if the frontier is visually messy:

- Show policy-minus-uniform error by budget for a selected policy set.
- Then describe the frontier qualitatively.

Policies to include in a main frontier plot:

- `uniform_step_balanced`,
- `stochastic_normalized_fragility`,
- `uniform_positive_fragility`,
- `stochastic_epsilon_greedy_eps0.2`,
- `softmax_normalized_fragility_temp2.0`,
- `softmax_normalized_fragility_temp4.0`.

The point is not to crown a universal winner. The point is to show that policy choice should be budget-aware.

## 4. Limitations and Future Work

Current location: lines 232-244.

This section is useful and should largely survive, but it needs updating.

### Keep

- Restriction to OC3 Denial of Service submodel.
- Use of only SOTA inputs.
- Assumption that each elicitation draw has identical cost.
- Full-data reference is not ground truth.
- LOO fragility is an instability heuristic, not expected information gain.
- Computational limitations and approximation caveats.

### Update

- "The number of reveal seeds is modest" should specify that the dense experiments use 30 reveal seeds. This is still modest, but less weak than the original 10-seed run.
- Replace "batching was used" with the actual dense setting: fragility recomputed every 10 reveals.
- Mention `max_loo_terms_per_step = 20`.
- Add a limitation about multiple exploratory comparisons: the later ablations were designed after seeing earlier results, so they should be read as exploratory rather than confirmatory.
- Add a limitation about the policy search space: softmax temperatures were hand-selected, not optimized prospectively.

### Possible Future Work Additions

- Replicate on other SaferAI risk models and other capability settings.
- Move from nodewise exchangeable mixtures to model-coherent mixtures that preserve forecaster/repeat structure.
- Develop an expected-information-gain or Bayesian experimental-design version of the acquisition rule.
- Use bootstrap or uncertainty-aware fragility estimates rather than point estimates.
- Study cost models where eliciting multiple inputs from the same human expert is cheaper than recruiting independent experts.
- Pre-register a smaller set of promising policies for a confirmatory run.

## 5. Conclusion Replacement

Current location: lines 246-247.

Replace fully.

The new conclusion should make four points:

1. This report studied a retrospective budget-recovery problem for OC3 DoS `P(success)`.
2. LOO fragility has some signal, but deterministic use of that signal is brittle.
3. Randomized use of positive fragility is more promising, and raw magnitudes appear poorly calibrated as direct probabilities.
4. The result is not a finished allocation method; it is evidence that output-sensitive elicitation allocation is worth studying with better-calibrated and budget-aware policies.

Avoid:

- "The best policy is X."
- "Softmax solves calibration."
- "Fragility-guided allocation beats uniform."
- "This approximates value of information."

## 6. Recommended Figure Set

### Core Figures

These are the figures I would try to include in the main text.

#### Figure 1: Overview of Budget-Recovery Setup

Purpose:

- Explain the retrospective reveal protocol visually.

Could show:

- hidden pool of elicitation rows by node,
- initial 5-per-node seed allocation,
- policy chooses next node,
- output distribution compared to full-data reference.

This could be a schematic rather than a data plot.

#### Figure 2: Dense Policy Suite Error or Difference vs Budget

Purpose:

- Show that deterministic fragility policies perform poorly and motivate stochastic follow-up.

Recommended selected policies:

- `uniform_step_balanced`,
- `stochastic_normalized_fragility`,
- `greedy_loo_fragility`,
- `epsilon_greedy_eps0.2`,
- `exploration_bonus_c1.0`.

Preferred y-axis:

- Policy-minus-uniform mean error, if readable.
- Otherwise absolute mean W2 error.

#### Figure 3: Stochastic Ablation Win Rate vs Budget

Purpose:

- Show that `uniform_positive_fragility` and stochasticized variants are competitive at budget ranges where they matter.

Recommended selected policies:

- `stochastic_normalized_fragility`,
- `uniform_positive_fragility`,
- `stochastic_epsilon_greedy_eps0.2`,
- `stochastic_exploration_bonus_c1.0`.

#### Figure 4: Softmax Temperature Summary

Purpose:

- Show that too-sharp softmax is bad and flatter softmax is better, but not clearly better than uniform-positive fragility.

Recommended form:

- Temperature on x-axis.
- Average AUC or selected-budget win rate on y-axis.
- Horizontal references for `uniform_step_balanced`, `uniform_positive_fragility`, and `stochastic_normalized_fragility`.

#### Figure 5: Budget-Dependent Empirical Frontier

Purpose:

- Address the user's important point that the relevant question is not only average win rate, but which methods work at budgets one might actually choose.

Possible forms:

- best policy by budget,
- top policy family by budget,
- selected policy curves against uniform.

This may be a main-text figure if it is clean, otherwise appendix.

### Appendix Figures

- Full error-vs-budget curves for all policies in each experiment.
- Full win-rate-by-budget curves for all policies.
- Allocation concentration by budget for deterministic vs stochastic policies.
- Fragility runtime diagnostics.
- Step-count allocation heatmaps for selected policies/seeds, if available and useful.

## 7. Recommended Table Set

### Core Tables

#### Table 1: Scope and Common Dense Settings

Purpose:

- Prevent repeated prose about seeds, budgets, MC samples, fragility settings.

#### Table 2: Policy Taxonomy

Purpose:

- Let the reader understand all policies without long prose.

This is probably more useful than trying to define every policy in paragraph form.

#### Table 3: Experiment Inventory

Purpose:

- Preserve the research chronology without making the report read chronologically.

Columns:

- experiment,
- question,
- policies,
- common settings,
- main lesson.

#### Table 4: AUC Summary by Experiment

Purpose:

- Compact scalar summary.

Caution:

- The text should explicitly say AUC is not the whole story.

### Appendix Tables

- Full AUC by seed.
- Full win rate by budget.
- Full policy summary by budget.
- Full concentration by budget.
- Exact experiment settings from JSON outputs.

## 8. Algorithm Blocks

The active-testing paper uses compact algorithm blocks to make the sampling procedure concrete. This report could benefit from one or two.

Recommended:

### Algorithm 1: Retrospective Budget-Recovery Reveal Loop

Inputs:

- fitted row pools by node,
- hidden reveal orders,
- initial rows,
- policy,
- budget schedule.

Loop:

- compute or reuse fragility scores,
- select next node according to policy,
- reveal next hidden row for that node,
- at evaluation budgets, sample current output and compute W2 error to full-data reference.

This algorithm would clarify the whole experimental setup.

Optional:

### Algorithm 2: Softmax-Normalized Positive-Fragility Sampling

Inputs:

- fragility scores,
- temperature,
- eligible nodes.

Steps:

- restrict to finite positive fragility scores,
- scale by median positive fragility,
- compute softmax weights,
- sample next node,
- fallback to balanced uniform if no usable fragility scores.

Do not add too many algorithm blocks. One core reveal-loop algorithm plus a policy table may be enough.

## 9. Claims: Strong, Moderate, Avoid

### Strong Claims

These are well supported:

- Uniform step-balanced allocation is a strong baseline.
- Deterministic argmax-style LOO fragility policies perform poorly in the dense 30-seed suite.
- Stochasticized fragility policies are much more competitive than deterministic fragility policies.
- Very sharp softmax temperature performs poorly.
- Policy performance is budget-dependent, so AUC alone is incomplete.

### Moderate Claims

These are plausible but should be worded carefully:

- Positive LOO fragility appears to contain useful allocation signal.
- Raw fragility magnitudes are poorly calibrated as direct sampling probabilities.
- `uniform_positive_fragility` is a strong simple diagnostic policy in the stochastic and softmax ablations.
- Softmax temperature scaling improves over too-sharp proportional weighting, but does not clearly dominate uniform-positive fragility.

### Claims to Avoid

Avoid these:

- "Fragility-guided allocation beats uniform."
- "We found the optimal elicitation policy."
- "LOO fragility estimates value of information."
- "The full-data reference is ground truth."
- "Softmax calibration solves the problem."
- "Average AUC determines the policy one should use in practice."

## 10. Concrete Editing Checklist

### Preamble

- Fix duplicated/conflicting bibliography packages: the original draft currently loads `natbib` twice and also loads `biblatex`.
- Decide whether the report will use `natbib` or `biblatex`, not both.
- Add algorithm packages only if you decide to include algorithm blocks.

### Abstract

- Rewrite to mention the three dense experiment stages and the main calibration/randomization lesson.

### Introduction

- Keep the first two paragraphs mostly intact.
- Update the final paragraph to preview:
  - deterministic fragility,
  - stochastic fragility,
  - magnitude calibration via softmax,
  - budget-dependent performance.

### Methods

- Preserve experiment scope, forward model, mixtures, reference distribution, LOO fragility, and W2 metric.
- Expand budgeted reveal protocol with the dense-run common settings.
- Replace the current allocation-policy list with a policy taxonomy table and selected formulas.
- Add a core reveal-loop algorithm if desired.

### Results

- Delete the current 10-seed sparse-grid Results text.
- Add common dense-run settings.
- Add three question-driven experiment subsections.
- Add budget-dependent frontier subsection.
- Use AUC as one diagnostic, not the main narrative endpoint.

### Limitations and Future Work

- Keep the existing limitation categories.
- Update seed count, recomputation cadence, and approximation settings.
- Add exploratory/post-hoc ablation caveat.
- Add calibration and policy-selection future work.

### Conclusion

- Replace fully.
- Keep it short and restrained.
- Emphasize what was learned, not just whether one policy won.

## 11. Proposed New Table of Contents

This is the structure I would recommend starting from:

```text
1. Introduction
2. Methods
   2.1 Experiment scope
   2.2 Fitting elicitation distributions
   2.3 OC3 DoS P(success) submodel
   2.4 Exchangeable nodewise mixtures
   2.5 Full-data reference distribution
   2.6 Budgeted reveal protocol
   2.7 LOO output fragility
   2.8 Allocation policy families
   2.9 Evaluation metrics
3. Results
   3.1 Common dense-run setup
   3.2 Initial dense policy suite: does naive LOO fragility help?
   3.3 Stochastic ablation: is raw magnitude weighting necessary?
   3.4 Softmax ablation: can temperature scaling improve fragility probabilities?
   3.5 Budget-dependent policy performance
4. Limitations and Future Work
5. Conclusion
Appendix A. Additional tables
Appendix B. Full policy definitions and implementation details
Appendix C. Additional figures
```

This keeps the original draft's basic architecture, but replaces the obsolete Results section with the full empirical story.

## 12. First Revision Priorities

If we revise this plan together, I suggest deciding these in order:

1. Whether the Results section should be exactly question-driven, or slightly more chronological.
2. Whether `softmax` deserves a full main-text subsection or a shorter calibration-ablation subsection.
3. Which figure should carry the main empirical story:
   - stochastic win-rate-by-budget,
   - softmax temperature summary,
   - budget frontier,
   - or a combined selected-policy error plot.
4. Whether to add an algorithm block in Methods.
5. How much of the deterministic policy failure belongs in main text vs appendix.

My current recommendation:

- Use question-driven Results.
- Keep softmax in main text because it directly tests the calibration hypothesis.
- Include one algorithm block for the reveal loop.
- Include the deterministic suite in main text, but compactly.
- Make `uniform_positive_fragility` and the budget-dependent view central to the empirical interpretation.
