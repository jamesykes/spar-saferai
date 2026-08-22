# Metrics and Budget Aggregation Plan

Status: planning memo for discussion, not report prose.

Purpose: decide how to evaluate policy performance over budgets before deciding figures, tables, or final Results structure.

Baseline report draft: `report/main.tex`

## 1. The Core Evaluation Object

The basic unit of evaluation is a paired policy comparison at a reveal seed and budget.

Let

- \(p\) denote an allocation policy,
- \(s\) denote a reveal seed,
- \(b\) denote a revealed-row budget,
- \(U\) denote the `uniform_step_balanced` baseline,
- \(E_{p,s,b}\) denote squared Wasserstein-2 error to the full-data reference:

\[
E_{p,s,b}
=
W_2^2\left(T(M_{p,s,b}), T(M^{\mathrm{full}})\right).
\]

The paired structure matters. For a fixed seed \(s\), all policies share the same hidden reveal order within each node. Therefore comparisons should mostly be made using paired quantities such as

\[
E_{p,s,b} - E_{U,s,b}
\]

or

\[
\frac{E_{p,s,b}}{E_{U,s,b}}.
\]

This is preferable to comparing only unpaired averages, because each seed can have a different inherent difficulty.

## 2. First Principle: Curves Before Aggregates

The main empirical object should be the performance curve over budgets, not a single scalar summary.

Reason:

- The question is about budgeted recovery.
- The policy that looks best at low budgets may not look best at moderate budgets.
- Near the full budget, all policies must converge because almost all rows have been revealed.
- A scalar average can hide where a policy helps or hurts.

So the Results section should first establish budget-dependent behavior, and only then use aggregate metrics as summaries.

This does not mean showing every curve for every policy in the main text. It means the logic of the analysis should treat the budget curve as primary.

## 3. Pointwise Metrics by Budget

These metrics answer: at a fixed budget \(b\), how does policy \(p\) compare with uniform?

### 3.1 Mean Error

\[
\bar{E}_{p,b}
=
\frac{1}{S}\sum_s E_{p,s,b}.
\]

Pros:

- Directly tied to the loss function.
- Easy to interpret as average recovery error.
- Natural for error-vs-budget curves.

Cons:

- Sensitive to outlier seeds.
- Absolute values can be hard to interpret.
- Does not use the paired comparison against uniform directly.

Use:

- Good as the main absolute performance curve.
- Should usually be accompanied by uncertainty bands or paired comparisons.

### 3.2 Median Error

\[
\mathrm{median}_s(E_{p,s,b}).
\]

Pros:

- More robust to outlier seeds.
- Often useful when errors are skewed.

Cons:

- Less directly connected to expected squared error.
- Median of each policy separately is not a paired comparison.

Use:

- Good secondary check.
- Not necessarily the main metric unless the seed-level error distribution is very skewed.

### 3.3 Paired Error Difference vs Uniform

\[
\Delta_{p,s,b}
=
E_{p,s,b} - E_{U,s,b}.
\]

Aggregate by mean or median over seeds:

\[
\bar{\Delta}_{p,b}
=
\frac{1}{S}\sum_s \Delta_{p,s,b}.
\]

Negative is better than uniform.

Pros:

- Uses paired design.
- Keeps the metric in original squared-Wasserstein units.
- Shows the magnitude of improvement or harm.

Cons:

- Absolute differences can be dominated by budgets where errors are naturally larger.
- Can be visually harder to compare across budgets if the error scale changes a lot.

Use:

- Very good main comparison metric.
- Especially useful for selected-policy plots.

### 3.4 Relative Error vs Uniform

\[
R_{p,s,b}
=
\frac{E_{p,s,b}}{E_{U,s,b}}.
\]

Values below 1 are better than uniform.

Pros:

- Normalizes by the difficulty of the seed and budget.
- Common in active-testing-style presentations: performance relative to uniform-random evaluation is often more readable than absolute error.
- Makes it easier to see proportional improvement.

Cons:

- Can become unstable when \(E_{U,s,b}\) is very small, especially near full budget.
- Ratios can be skewed.
- A large ratio can occur from a tiny denominator even when absolute differences are small.

Use:

- Strong candidate as a main or secondary metric.
- Probably exclude the full budget \(b=1798\) from ratio summaries because all methods are forced toward zero error there.
- Consider plotting medians or log ratios rather than raw mean ratios.

### 3.5 Log Relative Error vs Uniform

\[
L_{p,s,b}
=
\log E_{p,s,b} - \log E_{U,s,b}
=
\log\frac{E_{p,s,b}}{E_{U,s,b}}.
\]

Negative is better than uniform.

Pros:

- Symmetric for multiplicative effects.
- More stable to summarize than raw ratios.
- The average log ratio corresponds to a geometric mean relative error.

Cons:

- Requires handling zero or near-zero errors.
- Less immediately intuitive for nontechnical readers.

Use:

- Excellent for aggregate summaries if errors are never exactly zero or if a small numerical floor is used.
- Could be too technical for the main text unless explained carefully.

### 3.6 Paired Win Rate vs Uniform

\[
\mathrm{WinRate}_{p,b}
=
\frac{1}{S}
\sum_s
\mathbf{1}\{E_{p,s,b} < E_{U,s,b}\}.
\]

Pros:

- Simple and robust.
- Answers whether a policy usually beats uniform, not just whether it has lower average error.
- Less sensitive to outlier magnitudes.

Cons:

- Throws away magnitude.
- A policy can win slightly often but lose badly occasionally.
- Ties need a convention.

Use:

- Good secondary reliability metric.
- Should not be the only metric.

### 3.7 Paired Improvement Probability Between Two Non-Uniform Policies

For policies \(p\) and \(q\):

\[
\frac{1}{S}
\sum_s
\mathbf{1}\{E_{p,s,b} < E_{q,s,b}\}.
\]

Pros:

- Useful for targeted ablations, e.g. `uniform_positive_fragility` vs `stochastic_normalized_fragility`.

Cons:

- Too many pairwise comparisons can clutter the report.

Use:

- Use sparingly for the key conceptual comparison:
  - uniform-positive vs proportional-fragility,
  - softmax temperatures vs proportional-fragility,
  - softmax temperatures vs uniform-positive.

## 4. Aggregating Across Budgets

Budget aggregation is where the report needs the most care.

The problem:

- We do not care equally about all budgets automatically.
- The full budget is not operationally informative, because all policies converge.
- A single aggregate can obscure budget-dependent behavior.

Therefore any aggregate should have an explicitly stated budget weighting.

### 4.1 Unweighted Mean Across Evaluation Budgets

\[
\frac{1}{K}\sum_{k=1}^K m(b_k),
\]

where \(m(b_k)\) is a budget-specific metric such as mean error, mean paired difference, win rate, or log relative error.

Pros:

- Simple.
- Easy to reproduce.
- Fine when evaluation budgets are equally spaced and intentionally chosen.

Cons:

- Treats each recorded budget equally, regardless of interval width or operational importance.
- Sensitive to arbitrary choice of evaluation grid.
- Including the terminal full budget can dilute or distort the result.

Use:

- Acceptable as a descriptive summary if the budget grid is dense and nearly uniform.
- Should exclude the full budget or report with/without it.

### 4.2 Area Under the Error Curve

For mean error:

\[
\mathrm{AUC}_p
=
\frac{1}{b_{\max}-b_{\min}}
\int_{b_{\min}}^{b_{\max}} \bar{E}_{p,b}\,db,
\]

estimated by a trapezoidal rule over the evaluated budget grid.

Pros:

- Standard learning-curve/sample-efficiency summary.
- Uses the whole curve.
- Accounts for spacing between budgets.

Cons:

- Still depends on the chosen budget interval.
- Can overweight high-budget regions if the interval is long.
- Absolute AUC can hide paired behavior.
- If the full budget is included, it adds a forced-convergence endpoint.

Use:

- Fine as a scalar summary, but not the main story.
- Better reported as "area under the budget-recovery error curve" rather than as decisive evidence of best policy.
- Consider a truncated AUC excluding the final full-budget point.

### 4.3 Area Under Paired Difference Curve

\[
\mathrm{AUDC}_p
=
\frac{1}{b_{\max}-b_{\min}}
\int_{b_{\min}}^{b_{\max}}
\bar{\Delta}_{p,b}\,db.
\]

Negative is better than uniform.

Pros:

- Paired against the baseline.
- Keeps magnitude information.
- Directly answers average improvement/harm relative to uniform over a specified budget interval.

Cons:

- Same budget-weighting issues as AUC.
- Absolute difference scale can be dominated by regions of larger natural error.

Use:

- Stronger than raw policy AUC for baseline comparison.
- Good candidate aggregate if the report wants one main scalar.

### 4.4 Area Under Log Relative Error Curve

\[
\mathrm{AULR}_p
=
\frac{1}{b_{\max}-b_{\min}}
\int_{b_{\min}}^{b_{\max}}
\frac{1}{S}\sum_s
\log\frac{E_{p,s,b}}{E_{U,s,b}}
\,db.
\]

Negative is better than uniform.

Pros:

- Paired and scale-normalized.
- Multiplicative interpretation.
- Reduces domination by high-error budgets.

Cons:

- Needs careful zero-error handling.
- Harder to explain.

Use:

- Good technical appendix metric.
- Potentially good main metric if the audience is comfortable with it.

### 4.5 Area Under Win-Rate Curve

\[
\frac{1}{b_{\max}-b_{\min}}
\int_{b_{\min}}^{b_{\max}}
\mathrm{WinRate}_{p,b}
\,db.
\]

Pros:

- Summarizes reliability over budgets.
- Robust to outlier magnitudes.

Cons:

- Ignores effect size.
- Can make a policy that wins by tiny margins look better than one that wins less often but by more.

Use:

- Secondary aggregate, not primary.

### 4.6 Budget-to-Threshold

Pick a target error threshold \(\tau\), then compute the first budget at which a policy reaches it:

\[
B_p(\tau)
=
\min\{b : \bar{E}_{p,b} \le \tau\}.
\]

Or do this per seed, then summarize over seeds.

Pros:

- Very common in sample-efficiency settings.
- Directly answers: how much budget is needed to reach a quality target?

Cons:

- Requires meaningful thresholds.
- Curves may not be monotone due to Monte Carlo noise and reveal randomness.
- If many policies never reach the threshold, comparison becomes awkward.

Use:

- Potentially useful if we can define thresholds from the uniform baseline, e.g. "budget needed to reach the error achieved by uniform at budget X".
- Probably not the main metric unless we find a threshold that has a clear practical interpretation.

### 4.7 Regret to Empirical Best Policy

At each budget, define the best observed policy by mean error:

\[
p^*(b) = \arg\min_p \bar{E}_{p,b}.
\]

Then compute regret:

\[
\bar{E}_{p,b} - \bar{E}_{p^*(b),b}.
\]

Pros:

- Shows how far each policy is from the empirical best available policy at each budget.
- Useful for diagnosing budget-dependent behavior.

Cons:

- Post-hoc and optimistic.
- Can overfit noise in the experiment.
- Not appropriate as a headline metric for an exploratory report.

Use:

- Maybe appendix only.
- Avoid presenting this as a policy-selection rule.

### 4.8 Mean Rank by Budget

At each budget, rank policies by mean error and average ranks over budgets.

Pros:

- Common in benchmarking.
- Simple when comparing many methods.

Cons:

- Ignores effect size.
- Small irrelevant differences affect rank.
- With many related policy variants, ranks can be misleading.

Use:

- Not recommended as a main metric.

## 5. Budget Weighting Choices

The key unresolved issue is not just the metric, but the implicit weighting over budgets.

### Option A: Uniform Weight Over Evaluated Budgets

Treat each recorded budget equally.

Pros:

- Simple.
- Natural because the dense grid is almost evenly spaced.

Cons:

- Still arbitrary.
- Includes high budgets where all methods converge unless truncated.

Recommendation:

- Acceptable for descriptive summaries, but exclude the full-budget point from aggregates.

### Option B: Uniform Weight Over Budget Interval

Use trapezoidal integration over actual budget values.

Pros:

- Standard for learning curves.
- Accounts for interval widths.

Cons:

- Similar to Option A here because the grid is nearly uniform.
- Still gives substantial weight to late budgets unless truncated.

Recommendation:

- Fine for AUC/AUDC, but define the budget interval explicitly.

### Option C: Log-Budget Weighting

Aggregate over \(\log b\) rather than \(b\).

Pros:

- Gives relatively more emphasis to earlier budgets.
- Common in some sample-efficiency contexts where multiplicative budget differences matter.

Cons:

- Less natural here because the initial budget is fixed at 45 and subsequent budgets are roughly linear.
- Harder to justify unless early-budget performance is explicitly central.

Recommendation:

- Probably not needed unless we decide low-budget behavior is the main operational concern.

### Option D: Operational Budget Set

Preselect a small set of budgets that represent practically plausible elicitation budgets.

Pros:

- Directly tied to decision-making.
- Avoids arbitrary averaging over unimportant regions.

Cons:

- Requires a real justification for the chosen budgets.
- We may not currently know what budgets are operationally plausible.

Recommendation:

- Good if SaferAI has practical budget regimes in mind.
- Otherwise avoid inventing them.

### Option E: Truncated Budget Range

Aggregate over all dense budgets except the terminal full-data budget, or over a justified pre-full interval.

Pros:

- Avoids the forced-convergence endpoint.
- Still uses most of the curve.

Cons:

- The truncation point can look arbitrary if not explained.

Recommendation:

- Exclude the full budget \(1798\) from all aggregate metrics.
- Be cautious about any further truncation unless justified.

## 6. Recommended Metric Package

My current recommendation is a small hierarchy, not one metric.

### Primary Object

Budget curves:

1. Mean or median error by budget.
2. Paired difference vs uniform by budget.
3. Relative or log-relative error vs uniform by budget.

The Results should be written around these curves conceptually, even before deciding the exact visuals.

### Primary Aggregate

Use one of:

1. Truncated area under mean error curve, excluding full budget.
2. Truncated area under paired difference curve vs uniform, excluding full budget.
3. Truncated area under mean log-relative error curve vs uniform, excluding full budget.

My preference:

- For a broad audience: truncated area under paired difference curve vs uniform.
- For a more technical/statistical audience: truncated area under log-relative error curve vs uniform.

Why not only raw AUC?

- Raw AUC is useful, but it does not directly use the paired uniform baseline and may not answer the central comparative question as cleanly.

### Secondary Reliability Metrics

Use:

- win rate vs uniform by budget,
- median paired difference vs uniform,
- seed-level paired differences or confidence intervals if needed.

These help distinguish "large average improvement driven by few seeds" from "small but reliable improvement".

### Optional Sample-Efficiency Metric

Budget-to-threshold could be useful if we define thresholds carefully.

Possible thresholds:

- Error reached by uniform at budget 900.
- Error reached by uniform at budget 1200.
- A fixed quantile discrepancy threshold if one has substantive meaning.

But unless a threshold has a clear interpretation, do not make this central.

## 7. How This Affects the Results Structure

The Results section should probably not begin with a table of experiments or AUC rankings.

Instead, a cleaner flow is:

### 3.1 Common Experimental Setup

Briefly remind the reader that all dense experiments use the same seeds, budgets, Monte Carlo settings, and uniform baseline. Most details belong in Methods, so this should be short.

### 3.2 Initial Dense Policy Suite

Question:

- What happens when LOO fragility is used directly?

Metrics:

- error curve,
- paired difference or relative error vs uniform,
- compact aggregate only after showing/understanding the budget behavior.

Interpretation:

- deterministic use is poor;
- stochastic proportional fragility is the interesting exception.

### 3.3 Stochastic Ablation

Question:

- Is raw proportional weighting necessary?

Metrics:

- compare `stochastic_normalized_fragility` and `uniform_positive_fragility` directly,
- use relative/log-relative error vs uniform,
- use win rate only as secondary reliability.

Interpretation:

- randomization among positive-fragility nodes appears to matter more than raw magnitude weighting.

### 3.4 Softmax Temperature Ablation

Question:

- Can magnitude weighting be improved by temperature scaling?

Metrics:

- compare temperatures by the chosen aggregate and budget curves,
- include references to `uniform_positive_fragility` and `stochastic_normalized_fragility`.

Interpretation:

- too-sharp weighting is poor;
- flatter softmax is better;
- no clear evidence that softmax beats uniform-positive.

### 3.5 What the Budget Dependence Means

This should not be "best policy at each budget" as a prescription.

Better framing:

- The conclusions are budget-sensitive.
- Therefore future work should treat budget as part of the policy-evaluation problem.
- We should avoid claiming one universal best allocation rule from these experiments.

## 8. What I Would Avoid

Avoid making any of these the main reporting strategy:

- Average win rate over arbitrary ranges like budgets \(\ge 900\).
- Best policy at each budget as if it were a recommended switching rule.
- Raw all-budget AUC as the sole headline.
- Mean rank across policies.
- Post-hoc empirical frontier as the central claim.

These can be diagnostic tools, but they do not fit the report's stated aim as well as paired baseline-relative curves and carefully defined budget aggregates.

## 9. Decisions To Make Next

The key decisions before writing Results are:

1. Should the main baseline-relative metric be absolute difference, relative ratio, or log-relative ratio?
2. Should aggregates exclude only the full budget, or use another justified truncation?
3. Do we want one main aggregate metric, or two complementary aggregates?
4. Do we want to include budget-to-threshold as a sample-efficiency metric?
5. Should results be organized by experiment stage or by metric?

My current answers:

1. Use paired absolute difference for readability, and log-relative ratio as a technical robustness check.
2. Exclude the full budget \(1798\) from aggregates; do not use arbitrary high-budget windows.
3. Use one main aggregate plus win rate as secondary.
4. Skip budget-to-threshold unless a meaningful threshold emerges.
5. Organize by experiment stage, but interpret each stage using the same metric package.
