# SaferAI OC3 DoS budget-recovery plan: exchangeable mixture fragility for SOTA P(success)

## Purpose

This note specifies the implementation plan for the SPAR final-report experiment on prioritising elicitation effort in SaferAI-style quantitative cyber risk models.

The aim is deliberately modest:

> Given a fixed elicitation budget, can a fragility-guided allocation heuristic recover a full-data reference output distribution more efficiently than uniform allocation?

This is **not** claimed to be optimal active learning, Bayesian experimental design, or a true value-of-information method. It is an active-learning-inspired, retrospective **fragility-guided budget recovery** experiment.

The intended use of this note is to guide notebook implementation and to give Codex / LLM coding assistants a precise specification of what to implement. It should specify the method clearly, but it is not intended to be line-by-line code.

---

## Final-report scope

The final-report experiment is scoped to:

```text
Risk model:        OC3 Denial of Service
Component:         P(success) submodel only
Capability level:  SOTA only
Output:            distribution of P(success)
```

Jakub elicited only the MITRE-step probabilities relevant to the attack-success component, not the remaining multiplicative factors in the total risk equation such as number of actors, attempts per actor, or damage conditional on success.

This is acceptable for the final report. The proof of concept focuses on the part of the model where the interesting allocation dynamics lie:

- elicited MITRE-step probability inputs;
- AND/OR-style aggregation;
- nontrivial downstream effects from changing specific node distributions.

The primary output is:

$$
T(M) = \mathcal{L}\left(P_M(\mathrm{success})\right),
$$

where:

- \(M\) denotes the current collection of nodewise elicitation mixtures;
- \(P_M(\mathrm{success})\) is the attack-success probability induced by those mixtures and the OC3 DoS gate structure;
- \(\mathcal{L}(X)\) means the **law**, i.e. the probability distribution, of the random variable \(X\). It does **not** mean likelihood.

Do **not** describe the primary experiment as recovering the full total-risk distribution. It recovers the attack-success output distribution for the OC3 DoS \(P(\mathrm{success})\) submodel.

A careful final-report phrasing is:

> We restrict attention to the \(P(\mathrm{success})\) submodel of the OC3 Denial of Service risk model. This is where the elicited MITRE-step probabilities enter and where the nontrivial AND/OR aggregation structure creates meaningful differences between allocation strategies. Extending the method to the remaining multiplicative risk factors is left for future work.

A log-scale output,

$$
T_{\log}(M) = \mathcal{L}\left(\log P_M(\mathrm{success})\right),
$$

is a natural future robustness check, especially if later work reincorporates the full multiplicative total-risk model. It is **not** the primary output for the final-report experiment.

---

## Dataset

The dataset is:

```text
detailed_estimates_OC3_DDoS_sota_saturated_40_repeats_5_LLMs_1_expert.csv
```

It contains:

```text
40 repeats × 5 LLM forecaster models × 9 MITRE-step inputs × 2 capability levels = 3600 rows
```

There are a small number of invalid rows from API timeouts, missing quartiles, or similar issues. These should be excluded before fitting distributions.

The five LLM forecaster models are:

```text
claude-sonnet-4-6
gpt-5-mini
gemini-3-flash-preview
claude-haiku-4-5-20251001
claude-opus-4-7
```

The elicited probability summaries are:

```text
percentile_25th
percentile_50th
percentile_75th
```

The `estimate` column is the median estimate for usable rows and should be treated as equivalent to `percentile_50th`.

SaferAI has moved toward eliciting 25th/50th/75th percentiles, so this project should use those quartiles rather than 5th/50th/95th intervals.

---

## Preprocessing

### Draw identifiers

Do **not** use `repeat_index` alone as the repeat identifier. Some models appear across multiple `run_id`s, and repeat indices can repeat across runs.

Use a globally unique draw identifier based on the tuple:

```text
(model, run_id, repeat_index)
```

Within each LLM forecaster model, the pair `(run_id, repeat_index)` identifies a stochastic repeat. Including `model` makes the identifier globally unique.

### Valid rows

Exclude rows if any of the following hold:

```text
has_error == True
percentile_25th is missing
percentile_50th is missing
percentile_75th is missing
percentile_25th > percentile_50th
percentile_50th > percentile_75th
any percentile is outside [0, 1]
```

The notebook should report:

- total rows;
- rows excluded due to errors;
- rows excluded due to missing quartiles;
- rows excluded due to invalid ordering or out-of-range values;
- final usable row count;
- usable row count for the SOTA subset.

### Capability-level mapping

The two elicited capability levels are **SOTA** and **saturated**.

The CSV represents these levels through task names. For the final-report experiment, keep only SOTA rows.

Current mapping:

```text
SOTA:
  - Paddle
  - Labyrinth Linguist

saturated:
  - pytorchLightning
  - Randsubware
```

This mapping is step-specific: each MITRE-step input has one SOTA task and one saturated task, not all four tasks.

Operationally, add a `capability_level` column during preprocessing and filter to:

```text
capability_level == "SOTA"
```

The SOTA-only subset should contain approximately:

```text
9 MITRE-step inputs × 5 LLM forecaster models × 40 repeats = 1800 rows
```

minus invalid rows.

---

## OC3 DoS P(success) forward model

The OC3 DoS success-probability submodel has nine elicited leaf inputs, corresponding to MITRE-step / technique-level success probabilities:

| Short variable | CSV / model label |
|---|---|
| \(p_{\mathrm{active}}\) | `T1595 - Reconnaissance: Active Scanning` |
| \(p_{\mathrm{gather}}\) | `T1590 - Reconnaissance: Gather Victim Network Information` |
| \(p_{\mathrm{acquire}}\) | `T1583.005 - Resource Development: Acquire Botnet` |
| \(p_{\mathrm{build}}\) | `T1584.005 - Resource Development: Build/Compromise Botnet` |
| \(p_{\mathrm{masquerading}}\) | `T1036 - Defense Evasion: Masquerading` |
| \(p_{\mathrm{port}}\) | `T1571 - Defense Evasion: Non-Standard Port` |
| \(p_{\mathrm{c2}}\) | `TA0011 - Command-and-Control` |
| \(p_{\mathrm{direct}}\) | `T1498.001 - Impact: Direct Network Flood` |
| \(p_{\mathrm{reflection}}\) | `T1498.002 - Impact: Reflection/Amplification Attack` |

These are aggregated into tactic-level probabilities as follows.

### Reconnaissance

Reconnaissance succeeds if either active scanning succeeds or gathering victim network information succeeds:

$$
p_{\mathrm{rec}}
=
p_{\mathrm{active}} + p_{\mathrm{gather}} - p_{\mathrm{active}}p_{\mathrm{gather}}.
$$

This is an OR gate.

### Resource Development

Resource Development succeeds if either botnet acquisition succeeds or botnet compromise/build succeeds:

$$
p_{\mathrm{res}}
=
p_{\mathrm{build}} + p_{\mathrm{acquire}} - p_{\mathrm{build}}p_{\mathrm{acquire}}.
$$

This is an OR gate.

### Defense Evasion

Defense Evasion succeeds only if both non-standard port use and masquerading succeed:

$$
p_{\mathrm{def}}
=
p_{\mathrm{port}}p_{\mathrm{masquerading}}.
$$

This is an AND gate.

### Command and Control

Command and Control is elicited directly:

$$
p_{\mathrm{c2}}.
$$

### Impact

Impact succeeds if either direct network flood succeeds or reflection/amplification succeeds:

$$
p_{\mathrm{imp}}
=
p_{\mathrm{direct}} + p_{\mathrm{reflection}} - p_{\mathrm{direct}}p_{\mathrm{reflection}}.
$$

This is an OR gate.

### Final success probability

The overall successful attack probability is the product of the five tactic-level probabilities:

$$
P(\mathrm{success})
=
p_{\mathrm{rec}}
\cdot
p_{\mathrm{res}}
\cdot
p_{\mathrm{def}}
\cdot
p_{\mathrm{c2}}
\cdot
p_{\mathrm{imp}}.
$$

Equivalently:

$$
P(\mathrm{success})
=
\left(p_{\mathrm{active}} + p_{\mathrm{gather}} - p_{\mathrm{active}}p_{\mathrm{gather}}\right)
\left(p_{\mathrm{build}} + p_{\mathrm{acquire}} - p_{\mathrm{build}}p_{\mathrm{acquire}}\right)
\left(p_{\mathrm{port}}p_{\mathrm{masquerading}}\right)
p_{\mathrm{c2}}
\left(p_{\mathrm{direct}} + p_{\mathrm{reflection}} - p_{\mathrm{direct}}p_{\mathrm{reflection}}\right).
$$

The notebook should implement this formula directly. The output distribution is obtained by sampling the nine leaf probabilities from their current nodewise mixtures and propagating those samples through this formula.

---

## Basic fitted object

The basic fitted object is an elicitation distribution:

$$
D_{j,d},
$$

where:

- \(j\) indexes one of the nine OC3 DoS MITRE-step inputs;
- \(d\) indexes one valid SOTA elicitation draw, identified by `draw_uid` and associated with an LLM forecaster model.

Each usable row gives three quartiles \((q_{25}, q_{50}, q_{75})\) for a probability input.

Fit a distribution on \([0,1]\) to each usable row. The main method should use a Beta distribution fitted by quantile matching:

$$
(\hat\alpha,\hat\beta)
=
\arg\min_{\alpha,\beta}
\sum_{\tau \in \{0.25,0.50,0.75\}}
\left[
Q_{\mathrm{Beta}}(\tau;\alpha,\beta)-q_{\tau}
\right]^2.
$$

Implementation details:

- constrain \(\alpha,\beta > 0\);
- handle invalid or inconsistent quartiles explicitly before fitting;
- when sampling by inverse CDF, avoid exact endpoints by clipping uniforms away from 0 and 1;

---

## Pooling model: exchangeable nodewise mixtures

For the final-report experiment, use an **exchangeable nodewise mixture** over revealed draws.

For MITRE-step input \(j\), let \(S_j\) be the set of currently revealed valid SOTA draws for that step. Define the current node mixture as:

$$
M_j(S_j)
=
\frac{1}{|S_j|}
\sum_{d\in S_j} D_{j,d}.
$$

The current model is:

$$
M = (M_1,\ldots,M_9).
$$

This is a nodewise approximation. It represents the spread of elicited views separately for each MITRE-step input. For example, the mixture for Active Scanning contains the revealed draws for Active Scanning, and the mixture for Direct Network Flood contains the revealed draws for Direct Network Flood. However, it does not preserve the fact that a single LLM model or repeat may have given a coherent set of estimates across all nine MITRE-step inputs. In SaferAI's full expert-coherent procedure, one can sample an internally consistent set of estimates from a single expert across the whole model. Here, because we mix independently at each node, one Monte Carlo draw may combine an Active Scanning value derived from one LLM/repeat with a Direct Network Flood value derived from another. This is acceptable for the budget-recovery experiment, but it is a modeling approximation.

### Interpretation of exchangeable pooling

Under the exchangeable formulation, the unit of budget is an additional **LLM elicitation draw**, not an additional independent expert. This should be interpreted as a pilot for elicitation-allocation methods rather than a direct estimate of optimal human-expert allocation.

Earlier versions of this plan used a model-balanced mixture, giving equal weight to each of the five LLM forecaster models. That remains conceptually attractive if we treat the five LLMs as the main diversity units.

For the final-report experiment, however, the primary goal is budget recovery from the actual repeated-elicitation dataset. The exchangeable nodewise mixture has two advantages:

1. It is simple and directly tied to the available row-level elicitation draws.
2. If the budget eventually reveals all valid SOTA draws, the current model exactly becomes the full-data reference model.

We will still use a model-aware reveal protocol so that the early observed draws are diverse across LLM forecaster models. But the pooled node distribution itself will be exchangeable over the revealed draws.

Model-balanced and model-coherent alternatives should be discussed as future robustness checks, not implemented as part of the final-report experiment.

---

## Sampling from a nodewise mixture

To sample from the current node mixture \(M_j(S_j)\):

1. choose one revealed elicitation draw \(d\in S_j\) uniformly;
2. sample a probability value from that draw's fitted distribution \(D_{j,d}\).

Repeat this independently for each MITRE-step input \(j=1,\ldots,9\), then propagate the nine sampled probabilities through the OC3 DoS success-probability formula.

For common-random-number comparisons, the implementation should keep the same underlying random streams where possible across current, full-data, and perturbed models. Conceptually, mixture sampling needs randomness both for selecting a mixture component and for sampling within that component's fitted distribution.

---

## Full-data reference model

The primary full-data reference is the **full-data exchangeable nodewise mixture**.

For each MITRE-step input \(j\), let \(S_j^{full}\) be the set of all valid SOTA draws for that step. Define:

$$
M_j^{full}
=
\frac{1}{|S_j^{full}|}
\sum_{d\in S_j^{full}}D_{j,d}.
$$

The full-data reference model is:

$$
M^{full}=(M_1^{full},\ldots,M_9^{full}).
$$

The full-data reference output is:

$$
T(M^{full})
=
\mathcal{L}\left(P_{M^{full}}(\mathrm{success})\right).
$$

This is **not truth**. It is an internal retrospective reference: the output distribution that would be obtained if all valid SOTA elicitation draws in the dataset were used under the exchangeable nodewise mixture approximation.

---

## Output distance

Use squared Wasserstein-2 distance between one-dimensional empirical output distributions as the main error metric:

$$
\mathrm{Err}(M)
=
W_2^2\left(T(M),T(M^{full})\right).
$$

For one-dimensional distributions with quantile functions \(Q_A(u)\) and \(Q_B(u)\),

$$
W_2^2(A,B)
=
\int_0^1 \left(Q_A(u)-Q_B(u)\right)^2\,du.
$$

So in this setting, \(W_2^2\) is interpretable as a **mean squared quantile discrepancy** between the current and full-data output distributions.

This is the main reason to prefer \(W_2^2\) over \(W_1\) for the final-report experiment: the task is framed as a squared-error-style budget-recovery problem.

\(W_1\) would be a natural future robustness check. It corresponds to mean absolute quantile discrepancy and is less sensitive to large tail differences.

---

## Monte Carlo design

Use empirical Monte Carlo output distributions rather than fitting a parametric family to the output.

Recommended starting settings:

```text
Initial development particles: 2,000
Final rerun particles:         5,000 or more if runtime allows
Outer reveal seeds:            10-20, not 100 by default
```

Use common random numbers where possible:

- reuse the same random streams across current/full/perturbed models when comparing output distributions;
- clip inverse-CDF uniforms away from exact 0 and 1;
- do not rely on Monte Carlo differences from independent random seeds if common-random-number comparisons are available.

Before running the full experiment, profile a small version using a few reveal seeds and a smaller number of Monte Carlo particles. Use this to choose the final number of reveal seeds, particles, and batch size.

---

## Hidden reveal protocol

The retrospective experiment simulates a budget-limited elicitation process.

For each outer random seed:

1. For every MITRE-step input \(j\), pre-randomize hidden reveal orders for the valid SOTA draws.
2. The same hidden reveal order is shared by all allocation policies in that seed.
3. All policies start from the same initial seed set.
4. When a policy selects MITRE-step input \(j\), it receives the next unused draw for \(j\) according to that seed's hidden reveal order.
5. Policy comparisons are therefore paired within each random seed.

### Model-aware reveal order

Although the pooling model is exchangeable over revealed draws, the reveal protocol should be model-aware.

Reason: the dataset was deliberately generated from five different LLM forecaster models. Early samples should not accidentally come from only one or two models.

For each MITRE-step input \(j\):

1. split valid SOTA draws by LLM forecaster model;
2. shuffle repeats within each model;
3. choose a seed-specific random order of the five models;
4. reveal draws by cycling through models, using the next available repeat from each model.

This gives a model-diverse reveal order while keeping the pooled estimator simple.

### Initial seed allocation

Use the term **initial seed allocation** rather than burn-in. Here it means the small, fixed set of elicitation draws that every policy receives before adaptive allocation begins. It is not MCMC burn-in.

Use an initial seed allocation of **five draws per MITRE-step input**, one from each LLM forecaster model.

For nine MITRE-step inputs, this uses:

$$
9 \times 5 = 45
$$

initial SOTA elicitation draws, which is about 2.5% of the available SOTA dataset.

After the initial seed allocation, allocation policies begin selecting which MITRE-step input receives additional draws.

---

## Budget schedule

The main figure should plot recovery performance as a function of budget.

Use total revealed SOTA elicitation draws as the x-axis. Equivalently, report average draws per MITRE-step input.

Suggested budget checkpoints:

```text
45    = initial seed set only = 5 draws per step
90    = 10 draws per step on average
180   = 20 draws per step on average
360   = 40 draws per step on average
720   = 80 draws per step on average
1080  = 120 draws per step on average
1440  = 160 draws per step on average
all valid SOTA draws
```

The exact final checkpoint should be the number of valid SOTA rows after preprocessing.

These checkpoints are suggestions, not a hard methodological requirement. After profiling, it is fine to adjust them, try a denser grid at low budgets, or use different integer checkpoints that make implementation cleaner.

To reduce computation, recompute LOO fragility scores after fixed-size batches rather than after every single revealed draw. A natural default batch size is **nine allocations**, corresponding to one additional draw per MITRE-step input on average. This default can be tuned after profiling:

- smaller batches make the policies more adaptive but more computationally expensive;
- larger batches reduce computation but risk using stale fragility scores.

---

## Leave-one-out fragility score

The primary deployable score is **leave-one-out output fragility**.

Given current revealed draw sets \(S_1,\ldots,S_9\), let \(M\) be the current exchangeable nodewise mixture model.

For MITRE-step input \(j\), and draw \(d\in S_j\), define the leave-one-out node mixture:

$$
M_j^{(-d)}
=
\frac{1}{|S_j|-1}
\sum_{d'\in S_j,\ d'\neq d}D_{j,d'}.
$$

Define the corresponding perturbed full model as:

$$
M^{(j,-d)}
=
(M_1,\ldots,M_{j-1},M_j^{(-d)},M_{j+1},\ldots,M_9).
$$

That is, \(M^{(j,-d)}\) is identical to the current model \(M\) except that node \(j\)'s mixture has been replaced by its leave-one-out version.

Define:

$$
F_j^{\mathrm{LOO}}
=
\frac{1}{|S_j|}
\sum_{d\in S_j}
W_2^2\left(
T(M^{(j,-d)}),
T(M)
\right).
$$

Interpretation:

> A MITRE-step input has high leave-one-out fragility if the current output distribution is sensitive to removing one currently observed elicitation draw from that step's mixture.

This is a finite-sample instability score. It is not value of information and does not prove that the next draw at that node will reduce error. It is a deployable heuristic for identifying where the current pooled node estimate is unstable in a downstream-relevant way.

### Why LOO rather than bootstrap for the final report

A bootstrap version is natural:

- resample the currently observed draws for node \(j\);
- rebuild the node mixture;
- measure the output movement;
- average over bootstrap replicates.

However, bootstrap fragility adds another Monte Carlo layer and complicates the final-report implementation.

For this experiment, use LOO fragility because it is:

- deterministic;
- cheaper;
- easier to explain;
- sufficient for a first test of whether fragility-guided allocation has signal.

Mention bootstrap fragility as a future extension in the report, but do not implement it for the final-report experiment.

---

## Allocation policies

Compare four policies.

### 1. Uniform allocation

Allocate additional draws as evenly as possible across MITRE-step inputs.

Implementation:

- choose among currently eligible nodes with the smallest revealed count \(|S_j|\);
- break ties using the seed-specific node order.

This is the main baseline.

### 2. Greedy LOO-fragility allocation

At each allocation decision, choose the eligible MITRE-step input with the largest current LOO fragility score:

$$
j^* = \arg\max_j F_j^{\mathrm{LOO}}.
$$

If scores are recomputed only in batches, use the most recently computed \(F_j^{\mathrm{LOO}}\) values within the batch.

### 3. Epsilon-greedy LOO-fragility allocation

At each allocation decision:

- with probability \(1-\varepsilon\), choose the eligible node with largest \(F_j^{\mathrm{LOO}}\);
- with probability \(\varepsilon\), explore by choosing among under-sampled eligible nodes.

Exploration should not be uniform over all nodes. Prefer nodes with low revealed count \(|S_j|\).

A simple default is:

```text
epsilon = 0.2
exploration choice = uniformly among eligible nodes with the smallest current |S_j|
```

This value can be varied in a small sensitivity check if implementation time allows.

### 4. Exploration-bonus LOO-fragility allocation

Use an acquisition score:

$$
A_j = F_j^{\mathrm{LOO}} + \lambda b(n_j),
$$

where:

- \(n_j = |S_j|\);
- \(b(n_j)=1/\sqrt{n_j}\);
- \(\lambda\) controls exploration strength.

To set a sensible scale, define:

$$
\lambda = c \times \mathrm{median}\left\{F_j^{\mathrm{LOO}}: j \text{ eligible and } F_j^{\mathrm{LOO}}>0\right\}.
$$

Use a small fixed set of \(c\) values if time allows, for example:

```text
c ∈ {0.25, 0.5, 1.0}
```

Do not overstate retrospectively tuned \(c\) values as prospective performance. In the report, either choose one value in advance or show a small sensitivity plot across these values.

If all current fragility scores are zero or undefined, the exploration-bonus policy should fall back to uniform allocation for that batch.

---

## Evaluation metrics

Use only a small number of metrics for the final report.

### Main metric

Squared Wasserstein-2 error to the full-data reference:

$$
\mathrm{Err}(M)=W_2^2\left(T(M),T(M^{full})\right).
$$

### Main figure

Plot error-vs-budget curves:

```text
x-axis: total SOTA elicitation draws revealed
        or average draws per MITRE-step input

y-axis: W2^2 distance to full-data reference

line: allocation policy

ribbon/error bars: variability across outer reveal seeds
```

Lower is better.

### Main scalar summary

Compute area under the error-vs-budget curve for each policy.

Lower area means better budget efficiency.

Do not include additional Shapley, threshold-crossing, top-k, raw-risk, or oracle next-add-one metrics in the final-report experiment. Those are useful future extensions but unnecessary here.

---

## Computational plan

The main computational layers are:

1. data cleaning;
2. fitting Beta distributions to quartile triples;
3. constructing hidden reveal orders;
4. propagating current node mixtures through the OC3 DoS forward model;
5. computing LOO fragility scores;
6. running allocation policies;
7. evaluating error to the full-data reference;
8. repeating over outer reveal seeds.

To keep runtime feasible:

- use SOTA only;
- use \(P(\mathrm{success})\), not log-risk;
- use LOO, not bootstrap;
- use common random numbers;
- record performance at budget checkpoints;
- recompute fragility after batches rather than after every single draw;
- start with 10-20 outer reveal seeds;
- use 2,000 particles during development and 5,000+ particles for final figures if runtime allows;
- profile a small run before committing to the final seed/particle/batch settings.

---

## Implementation order

Implement in this order:

1. Load and clean the CSV.
2. Construct a globally unique draw identifier from `(model, run_id, repeat_index)`.
3. Add `capability_level` and keep SOTA rows only.
4. Fit Beta distributions to valid 25/50/75 triples.
5. Implement the OC3 DoS \(P(\mathrm{success})\) forward model exactly as specified above.
6. Build the full-data exchangeable reference distribution \(T(M^{full})\).
7. Implement model-aware hidden reveal orders.
8. Implement initial seed allocation of 5 draws per MITRE-step input, one from each LLM model.
9. Implement current exchangeable node mixtures.
10. Implement the output distance \(W_2^2\).
11. Implement uniform allocation.
12. Implement LOO fragility.
13. Implement greedy LOO-fragility allocation.
14. Implement epsilon-greedy LOO-fragility allocation.
15. Implement exploration-bonus LOO-fragility allocation.
16. Generate error-vs-budget curves.
17. Compute area under the error-vs-budget curves.
18. Write up results and limitations.

---

## Expected final-report claims

Claims the report can safely make if the results support them:

> We propose a retrospective fragility-guided budget-recovery experiment for the OC3 DoS \(P(\mathrm{success})\) submodel. In this setting, allocation policies that prioritise MITRE-step inputs with high leave-one-out output fragility can be compared against uniform allocation by how quickly they recover the full-data reference output distribution.

> The method is active-learning-inspired, but it is not a true value-of-information calculation. It uses only currently observed elicitation draws and measures downstream instability of nodewise mixtures.

> The experiment tests whether non-uniform allocation can recover the full-data distribution of \(P(\mathrm{success})\) more efficiently than uniform allocation under repeated LLM-based elicitation.

Claims to avoid:

- This is optimal active learning.
- This estimates true value of information.
- This recovers full total-risk uncertainty.
- This proves where human experts should always be allocated.
- The full-data reference is truth.

---

## Limitations to state explicitly

1. **Scoped output**

   The experiment targets \(P(\mathrm{success})\), not full total risk.

2. **Retrospective reference, not truth**

   The full-data reference is the output using all valid SOTA LLM-elicited rows under the exchangeable nodewise mixture approximation. It is not ground truth.

3. **LLM elicitation draws, not human experts**

   The dataset consists of repeated LLM forecaster elicitations. Under the exchangeable formulation, the unit of budget is an additional LLM elicitation draw, not an additional independent expert. Results should be interpreted as a pilot for elicitation-allocation methods rather than as a direct estimate of optimal human-expert allocation.

4. **Nodewise mixture approximation**

   The method pools draws independently at each MITRE-step input. This means it captures the spread of elicited views at each individual node, but not the correlation between a single LLM model's estimates across different nodes. A Monte Carlo draw can therefore combine node values originating from different LLM models or repeats. This differs from an expert-coherent aggregation procedure, where one samples a whole internally consistent set of estimates from the same expert across the full model.

5. **LOO fragility is not value of information**

   Leave-one-out fragility measures finite-sample instability of the current node mixture, not the expected value of collecting another elicitation.

6. **Exchangeable pooling is a modeling choice**

   The exchangeable mixture is chosen to make the budget-recovery experiment clean and directly tied to the row-level repeated-elicitation dataset. A model-balanced mixture would answer a somewhat different question about the five LLM forecaster models as diversity units.

7. **Monte Carlo error**

   Output distributions are estimated by Monte Carlo. Use common random numbers and sufficient particles to keep simulation noise below the policy differences of interest.

---

## Future work after the final report

The final-report experiment should stay SOTA-only and focused. Natural extensions include:

1. repeat the experiment for the saturated capability level;
2. run a joint SOTA+saturated allocation experiment;
3. implement bootstrap fragility and compare it to LOO fragility;
4. compare \(W_2^2\) with \(W_1\);
5. compare exchangeable full-data references with model-balanced references;
6. investigate model-coherent references that preserve cross-node LLM-model coherence;
7. compute oracle next-add-one diagnostics to test whether LOO fragility predicts retrospective add-one improvement;
8. evaluate a log-scale output \(\mathcal{L}(\log P(\mathrm{success}))\);
9. extend from \(P(\mathrm{success})\) to full total-risk once actors, attempts, and damage factors are elicited;
10. repeat the method on other SaferAI risk models;
11. apply the method to human expert elicitation data if available.
