# Results Reporting Assessment

Status: planning memo for discussion, not report prose.

Baseline report draft: `report/drafts/original_draft_latex_repo/main.tex`

## 1. Diagnosis

The first-pass plots are not usable as report figures. The issue is not that the experiment is uninformative; it is that the presentation is trying to show too much at once.

The main problems are:

- Too many lines per plot. Seven or eight policy curves with similar colors and repeated markers are not readable.
- The pointwise budget curves are noisy. With 40 budgets, the line charts show high-frequency variation that is probably not substantively meaningful.
- Some plots combine policies with very different performance scales. In the initial dense suite, poor deterministic policies stretch the y-axis and make the more interesting uniform-vs-stochastic comparison hard to see.
- The terminal full-budget point is a boundary condition, not useful performance evidence. At budget 1798 all policies have revealed the full dataset, so errors are zero and relative error is undefined.
- Raw relative-error plots can be visually unstable because relative error is seedwise `policy_error / uniform_error`; this is the right paired calculation, but the plotted curve is noisy unless smoothed or summarized.

The fix is to make the main text selective. The exact by-budget metrics can still be available in appendix tables, but the main Results section should not try to display every policy and every metric simultaneously.

## 2. Reporting Principle

The main text should show the empirical story, not the entire output dump.

I would use:

- exact by-budget tables in the appendix;
- smoothed selected-policy curves in the main text;
- at most four or five policies per plotted panel;
- one main baseline-relative metric for the main story;
- win rate as a secondary reliability check.

The report should be explicit that smoothing is only for display. The unsmoothed by-budget values remain the underlying reported data.

## 3. Metric Choice

For main-text plots, I would use median seedwise relative error vs uniform:

\[
\mathrm{median}_s\left(\frac{E_{p,s,b}}{E_{U,s,b}}\right).
\]

Reasons:

- It uses the paired reveal-seed structure.
- It is easier to interpret than raw squared-Wasserstein error because uniform is always the reference line.
- It is less sensitive than mean relative error to seeds where the uniform error is very small.
- It directly addresses whether a policy is better or worse than the baseline at a given budget.

I would also show win rate vs uniform, but only as a secondary plot or secondary panel:

\[
\frac{1}{S}\sum_s \mathbf{1}\{E_{p,s,b} < E_{U,s,b}\}.
\]

Win rate answers a different question: reliability rather than magnitude. It should not replace error magnitude.

Mean and median absolute error by budget should still be generated and made available, but I would not make them the central plots unless the relative-error plots turn out to be misleading.

## 4. Smoothing Convention

I recommend a centered five-budget moving average for plotted curves.

Because budgets are spaced every 45 rows, this corresponds to a 225-row display window. The smoothing would be applied after computing the by-budget metric, not to raw per-seed errors.

For example:

- compute median relative error for each policy at each budget;
- exclude the boundary budgets 45 and 1798 from the displayed curve;
- apply a centered rolling mean across five adjacent budget points;
- use `min_periods=3` at the ends of the displayed curve.

Why five budget points?

- Three points still leaves too much high-frequency noise.
- Seven points starts to hide changes over the budget range.
- Five points is a reasonable first display choice given the 45-row budget spacing.

This is not a new statistical estimator. It is a readability convention for the figures.

## 5. Plot Selection

### Figure 1: Initial Dense Policy Suite

Purpose:

- Show that deterministic argmax-style fragility use performs poorly.
- Show that `stochastic_normalized_fragility` is the interesting exception in the initial suite.

Metric:

- Smoothed median relative error vs uniform.

Policies:

- `stochastic_normalized_fragility`
- `epsilon_greedy_eps0.2`
- `exploration_bonus_c1.0`
- `greedy_loo_fragility`

Do not plot all exploration-bonus constants in the main figure. `c=1.0` is enough to represent the best deterministic bonus variant.

Optional secondary panel:

- Smoothed win rate vs uniform for the same policies.

### Figure 2: Stochastic Fragility Ablation

Purpose:

- Test whether stochastic use of positive fragility is more important than raw proportional weighting.

Metric:

- Smoothed median relative error vs uniform.

Policies:

- `stochastic_normalized_fragility`
- `uniform_positive_fragility`
- `stochastic_epsilon_greedy_eps0.2`
- `stochastic_exploration_bonus_c1.0`

Optional secondary panel:

- Smoothed win rate vs uniform for the same policies.

This is probably the most important figure in the report.

### Figure 3: Softmax Temperature Ablation

Purpose:

- Test whether temperature scaling improves magnitude-weighted fragility sampling.

Metric:

- Smoothed median relative error vs uniform.

Policies:

- `uniform_positive_fragility`
- `stochastic_normalized_fragility`
- `softmax_normalized_fragility_temp0.25`
- `softmax_normalized_fragility_temp2.0`
- `softmax_normalized_fragility_temp4.0`

This keeps the plot to five non-uniform policies. It includes:

- the simple uniform-positive diagnostic policy,
- the original proportional-fragility policy,
- one too-sharp softmax,
- two flatter softmax policies.

`temp0.5` and `temp1.0` can go in appendix tables or a supplementary softmax-only plot if needed.

## 6. Scale and Styling

Use thinner lines than the first pass:

- line width around 1.2 or 1.4;
- no markers, or very small markers;
- no more than five policies per panel;
- direct labels at line ends if practical, otherwise a compact legend.

For relative-error plots:

- horizontal reference line at 1;
- y-axis should be chosen per figure, not shared globally;
- avoid letting one very poor policy make all other lines indistinguishable.

For the initial dense suite, it may be useful to use a wider y-axis because greedy and exploration-bonus variants are genuinely much worse. For stochastic and softmax ablations, use a tighter y-axis around the region where policies cluster near uniform.

I would not use the full-budget point in relative-error or win-rate plots.

## 7. What the Smoothed Summaries Say

These are not final report metrics, but they are useful for deciding the story.

Using a centered five-budget moving average of median relative error, excluding budgets 45 and 1798:

### Initial Dense Policy Suite

- `stochastic_normalized_fragility` stays close to uniform: smoothed median relative error ranges roughly from 0.88 to 1.52, with median about 1.02.
- `epsilon_greedy_eps0.2` is worse: roughly 0.98 to 2.53, median about 1.33.
- `exploration_bonus_c1.0` is much worse: roughly 1.10 to 4.67, median about 3.75.
- `greedy_loo_fragility` is worse still: roughly 1.09 to 7.27, median about 5.33.

Interpretation:

- Direct deterministic use of LOO fragility is not competitive.
- The stochastic proportional-fragility policy is the policy that motivates the next experiment.

### Stochastic Fragility Ablation

- `uniform_positive_fragility`: roughly 0.75 to 1.29, median about 0.96.
- `stochastic_epsilon_greedy_eps0.2`: roughly 0.85 to 1.14, median about 0.99.
- `stochastic_normalized_fragility`: roughly 0.88 to 1.52, median about 1.02.
- `stochastic_exploration_bonus_c1.0`: roughly 0.89 to 1.42, median about 1.01.

Interpretation:

- The stochastic variants are all much closer to uniform than the deterministic variants.
- `uniform_positive_fragility` and stochastic epsilon-greedy look at least as good as proportional fragility.
- This supports the idea that positive fragility is informative, but raw magnitude weighting is not clearly calibrated.

### Softmax Temperature Ablation

- `softmax_temp0.25`: roughly 1.02 to 1.75, median about 1.23.
- `softmax_temp2.0`: roughly 0.86 to 1.45, median about 1.01.
- `softmax_temp4.0`: roughly 0.85 to 1.34, median about 0.98.
- `uniform_positive_fragility`: roughly 0.75 to 1.29, median about 0.96.
- `stochastic_normalized_fragility`: roughly 0.88 to 1.52, median about 1.02.

Interpretation:

- Too-sharp softmax is bad.
- Flatter softmax is better.
- The best softmax variants are competitive, but they do not obviously improve on `uniform_positive_fragility`.

## 8. Recommended Results Narrative

I would structure the Results section as follows:

1. Reporting convention.
   - Explain mean/median error, paired win rate, and seedwise relative error.
   - Explain that main figures use smoothed curves for readability and appendix tables give exact by-budget values.

2. Initial dense policy suite.
   - Show deterministic LOO-fragility policies perform poorly.
   - Emphasize that `stochastic_normalized_fragility` is the interesting exception.

3. Stochastic ablation.
   - Compare proportional fragility, uniform-positive fragility, stochastic epsilon-greedy, and stochastic bonus.
   - Main lesson: randomization among positive-fragility nodes appears more robust than raw proportional weighting.

4. Softmax ablation.
   - Compare too-sharp softmax to flatter softmax and to uniform-positive.
   - Main lesson: temperature helps avoid over-sharp weighting, but does not clearly beat the simpler uniform-positive rule.

5. Summary.
   - LOO fragility appears to contain signal.
   - Deterministic or too-sharp use of that signal is brittle.
   - Randomized positive-fragility policies are the most promising follow-up direction.
   - The evidence remains exploratory.

## 9. What Not To Do

Do not:

- plot all policies in the main text;
- plot all five metrics for every experiment in the main text;
- rely on AUC as the main result;
- use arbitrary high-budget windows as headline evidence;
- present a "best policy at each budget" switching rule;
- show unsmoothed jagged curves as if the wiggles are meaningful.

## 10. Concrete Next Plotting Step

The next plotting script should generate exactly these main-text figures:

1. `initial_suite_smoothed_median_relative_error`
2. `initial_suite_smoothed_win_rate`
3. `stochastic_ablation_smoothed_median_relative_error`
4. `stochastic_ablation_smoothed_win_rate`
5. `softmax_ablation_smoothed_median_relative_error`
6. optionally `softmax_ablation_smoothed_win_rate`

If the stochastic ablation figure is still too busy, split it into two panels:

- proportional vs uniform-positive;
- stochastic epsilon/bonus variants.

But I would first try the four-policy version.
