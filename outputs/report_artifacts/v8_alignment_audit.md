# V8 Alignment Audit

Date: 2026-05-30

This audit compares `saferai_mixture_fragility_allocation_plan_v8.md`, `report/main.tex`, `outputs/report_artifacts/results_summary_draft.md`, and the current experiment implementation. It is an audit pass only. No experiments were rerun and `report/main.tex` was not rewritten.

## Executive Verdict

Most of the implementation matches the v8 plan on scope, data cleaning, Beta fitting, forward-model mapping, exchangeable nodewise reference construction, LOO fragility definition, policy families, and evaluation metrics.

The unresolved issue in report Section 5.5 is a real implementation-vs-v8 mismatch, not merely cautious wording. The v8 plan specifies a pre-materialized hidden reveal order for each MITRE-step input, shared by all policies within an outer reveal seed, with model-aware cycling after the initial seed allocation. The current implementation does not materialize or share such per-step reveal orders. Instead, after the initial seed allocation, policies choose concrete unrevealed fitted-row IDs using seed-controlled random selection at decision time.

This difference is material for final empirical claims. Under v8, two policies that reveal the same number of draws from a given step under the same seed receive the same first `k` hidden draws for that step. In the current implementation, the identity of within-step rows can differ by policy because each policy makes its own random row selections from the unrevealed pool. This can confound policy comparisons with within-step draw identity and can reduce the intended model diversity from model-aware cycling, especially for concentrated fragility-guided policies.

Recommended action: implement v8 model-aware hidden reveal orders and rerun one final locked experiment before treating the results as final empirical results. Until then, the current outputs should be described as development diagnostics.

## Checklist

| V8 item | Implementation status | Report status | Severity | Recommended action |
|---|---|---|---|---|
| OC3 Denial of Service only | Implementation restricts fitted experiment inputs to OC3 DoS dataset and OC3 DoS forward model. | Report states OC3 Denial of Service. | OK | None. |
| SOTA condition only | Cleaning maps capability levels and fitted policy experiments use `outputs/fitted_distributions/sota_beta_fits.csv`. | Report states SOTA only and gives usable SOTA count. | OK | None. |
| `P(success)` submodel only | `forward_model.py` implements only OC3 DoS `P(success)`; sampling propagates only this output. | Report states `P(success)` submodel only and excludes total-risk factors. | OK | None. |
| Output distribution of `P(success)`, not log-output | Sampling and distance code evaluate `p_success`; no log-output policy results are used. | Report states primary output is `P(success)`, not `log P(success)`. | OK | None. |
| No full total-risk modelling | No total-risk factors are implemented in the experiment path. | Report explicitly excludes full total-risk modelling. | OK | None. |
| No saturated or joint SOTA+saturated final results | Repeated experiment scripts consume SOTA Beta fits only. | Report results are SOTA-only; saturated analysis is future work. | OK | None. |
| Budget unit is one LLM elicitation draw | Revealed units are fitted rows with stable row IDs. | Report uses “LLM elicitation draw.” | OK | None. |
| Raw row count | Cleaning report records 3600 raw rows. | Report states 3600 raw rows. | OK | None. |
| Usable row count | Cleaning report records 3587 usable rows. | Report states 3587 usable rows. | OK | None. |
| Usable SOTA row count | Cleaning report records 1798 usable SOTA rows. | Report states 1798 usable SOTA rows. | OK | None. |
| Invalid-row handling | `data.py` excludes error rows, missing quartiles, bad ordering, and out-of-range quartiles; invalid accounting is reported. | Report describes these cleaning rules and invalid count. | OK | None. |
| Capability mapping | `data.py` maps task names to SOTA/saturated and audits task names by MITRE step. | Report says capability mapping was audited. | OK | None. |
| Use of quartiles | `beta_fit.py` fits to q25/q50/q75; `estimate` is not used for fitting. | Report states fitting uses quartiles. | OK | None. |
| Unique draw identifiers | `data.py` constructs `draw_uid` from `(model, run_id, repeat_index)`; `reveal.py` uses `(step_name, draw_uid)` as fitted-row UID. | Report says `repeat_index` alone is not used. | OK | None. |
| Beta fitted to q25/q50/q75 | `fit_beta_to_quartiles` minimizes quantile residuals at 0.25/0.50/0.75. | Report gives the quantile-matching objective. | OK | None. |
| Endpoint handling | `beta_fit.py` clips exact endpoint quartiles for fitting; audit outputs say no SOTA rows required endpoint clipping. | Report states no SOTA rows required endpoint clipping. | OK | None. |
| Fit-quality flags | `ok`, `warn`, and `fail` are generated; usable mixture rows include `ok` and `warn`. | Report states 1790 ok, 8 warn, 0 failed and warns retained. | OK | None. |
| Failed fits handled | `usable_fit_rows` and `build_nodewise_mixtures` require finite positive alpha/beta and `ok`/`warn`; failed rows would be excluded. | Report says no failed SOTA fits. | OK | None. |
| OC3 DoS formula | `forward_model.py` implements Recon OR, Resource OR, Defense AND, direct C2, Impact OR, and product. | Report formula matches v8. | OK | None. |
| Step-label mapping | `STEP_LABEL_BY_VARIABLE` matches the v8 mapping and was written to smoke-test JSON. | Report lists the mapping. | OK | None. |
| Output range checks | `forward_model.py` validates input probabilities and asserts output in `[0,1]`. | Not discussed in detail, but not necessary for report. | OK | None. |
| Full-data reference is exchangeable nodewise mixture | `mixtures.py` and `sampling.py` build independent nodewise mixtures over all usable SOTA rows. | Report describes exchangeable nodewise mixture and internal reference. | OK | None. |
| Model identity not preserved inside nodewise mixtures | Mixture sampling chooses rows independently per node. | Report describes this as a modelling approximation. | OK | None. |
| All valid SOTA rows used in full reference | Repeated experiment uses `usable_fit_rows(fit_df)` from SOTA Beta fits for full reference. | Report describes all valid SOTA rows. | OK | None. |
| Reference not described as ground truth | Code constructs an internal retrospective reference. | Report states it is not ground truth. | OK | None. |
| Initial seed allocation size and composition | `make_initial_seed_reveal` selects one row for each expected `(step, model)` group; strict mode produces 45 rows on this dataset. | Report states five rows per step, one per model, total 45. | OK | None. |
| Initial seed reproducibility | `make_initial_seed_reveal` is seed-controlled. | Report states initial seed is per reveal seed. | OK | None. |
| “Initial seed allocation” terminology | Implementation and report use “initial seed allocation.” | Report does not use “burn-in.” | OK | None. |
| Pre-materialized hidden reveal order after initial seed | Current main experiment does not materialize per-step hidden reveal orders. `make_uniform_reveal_order` exists but is a global random order and is not used by `run_policy_recovery` for v8 policies. | Section 5.5 explicitly flags this as unresolved. | Needs code change | Implement v8 hidden per-step reveal orders and rerun locked experiment. |
| Shared reveal order across policies within seed | Current policies select concrete row IDs via each policy’s own RNG path; there is no shared per-step order consumed by all policies. | Report says comparisons are paired by seed but flags the reveal-order issue. | Needs code change | Build reveal orders once per seed or deterministically from seed independent of policy; consume next row for selected step. |
| Model-aware cycling after initial seed | Current post-seed row selection is random within selected step, not model-cycled. | Report says v8 model-aware cycling should be verified. | Needs code change | Split each step by model, shuffle repeats within model, choose seed-specific model order, and cycle through models. |
| LOO removes one draw, not a whole node | `loo_perturbed_revealed_df` removes exactly one fitted row for a step; current model retains all other rows. | Report definition matches v8. | OK | None. |
| LOO is finite-sample instability, not VOI | Code exposes fragility scores only; report explicitly says not expected information gain. | Report wording is aligned. | OK | None. |
| Approximate LOO subsampling labelled | `compute_loo_fragility_scores` defaults to exact mode; optional cap is explicit and diagnostics are recorded. | Report clearly states development runs used cap 20 and exact remains the definition. | OK | None, but final run must label approximation if used. |
| `uniform_step_balanced` policy | Implemented; chooses among eligible steps with smallest revealed count, then a row within that step. | Report describes step-balanced uniform baseline. | OK for step choice; row reveal mechanism affected by reveal-order issue. | After reveal-order patch, policy should choose a step and consume hidden next row. |
| `greedy_loo_fragility` policy | Implemented; chooses eligible step with largest finite LOO fragility, fallback to under-sampled uniform. | Report describes greedy LOO. | OK for step choice; row reveal mechanism affected by reveal-order issue. | After reveal-order patch, policy should choose a step and consume hidden next row. |
| `epsilon_greedy_eps0.2` policy | Implemented with epsilon configurable and exploration among under-sampled eligible steps. | Report describes epsilon 0.2. | OK for step choice; row reveal mechanism affected by reveal-order issue. | After reveal-order patch, policy should choose a step and consume hidden next row. |
| `exploration_bonus_loo_fragility` c sensitivity | Implemented with `c`; sensitivity run includes c=0.25, 0.5, 1.0. | Report labels c=1.0 as coming from separate sensitivity run. | OK for step choice; row reveal mechanism affected by reveal-order issue. | After reveal-order patch, include c sensitivity or predeclare selected c. |
| Source-run differences | Reporting helpers preserve `source_run`; report says c=1.0 comes from separate sensitivity run. | Source-run differences are explicit. | Needs final-run decision | Prefer one final locked run for the primary table to avoid combining source runs. |
| Squared Wasserstein-2 error | `distances.py`, `experiment.py`, and `analysis.py` use squared W2 via quantile grids. | Report defines squared W2 as main error. | OK | None. |
| AUC under error-vs-budget curve | `analysis.py` computes trapezoidal AUC by seed/policy. | Report uses AUC as scalar diagnostic. | OK | None. |
| Concentration diagnostics secondary | Analysis computes concentration diagnostics; report presents them as diagnostics. | Report does not treat concentration as primary metric. | OK | None. |
| No extra unplanned core metrics | Shapley, threshold, top-k, raw-risk, oracle metrics are absent from implementation and report. | Report does not introduce these as core results. | OK | None. |
| Report overclaim check | Report uses “active-learning-inspired” but caveats not optimal, not BED, not VOI; claims are framed as development diagnostics. | Wording is mostly cautious. | Minor wording issue | If final results are locked, replace “development run” language with “locked experiment” where appropriate; keep caveats. |
| Current empirical outputs as final results | Current outputs use non-v8 reveal mechanics and combine source runs for key policies. | Report already calls them development outputs. | Needs final-run decision | Do not treat current outputs as final empirical results. |

## Scope

Implementation and report scope align with v8. The experiment path is restricted to OC3 Denial of Service, SOTA rows, and the `P(success)` submodel. The output is the distribution of `P(success)`, not log-output. There is no saturated analysis, joint SOTA+saturated analysis, or full total-risk model in the reported results. The budget unit is one fitted SOTA LLM elicitation row.

## Data and Cleaning

`data.py` implements the v8 cleaning rules: exclude rows with errors, missing quartiles, invalid quartile ordering, or out-of-range quartiles. The cleaning report records 3600 raw rows, 3587 usable rows, 1798 usable SOTA rows, and 13 invalid rows. The capability-level mapping is task-name based and audited by MITRE-step label. `draw_uid` is constructed from `(model, run_id, repeat_index)`, and fitted-row identifiers are constructed from `(step_name, draw_uid)`.

One nuance: `number_of_unique_draw_uids_in_sota` is 200 because each `draw_uid` spans the nine MITRE-step rows from the same model/run/repeat. This is expected; a fitted row is uniquely identified by `(step_name, draw_uid)`.

## Beta Fitting

`beta_fit.py` matches v8: one Beta distribution is fitted to q25/q50/q75 for each valid row using positive alpha/beta parameters. Endpoint clipping is implemented for fitting, but audit outputs report that no SOTA quartiles required clipping. Fit-quality flags are recorded; `warn` fits are retained and `fail` fits are excluded from usable mixture rows. The report correctly summarizes 1798 fitted SOTA rows, 1790 ok fits, 8 warn fits, and 0 failed fits.

## Forward Model

`forward_model.py` implements the v8 OC3 DoS `P(success)` formula exactly:

- Reconnaissance: OR of active scanning and gather victim network information.
- Resource Development: OR of acquire botnet and build/compromise botnet.
- Defense Evasion: AND of non-standard port and masquerading.
- Command and Control: direct elicited probability.
- Impact: OR of direct flood and reflection/amplification.
- Final success probability: product of the five tactic-level probabilities.

The variable-to-step mapping matches v8 and output range checks are present.

## Reference Model

The implementation builds the full-data reference as an exchangeable nodewise mixture over all usable fitted SOTA rows. Mixture sampling does not preserve LLM model identity or cross-node coherence; it samples independently at each node. The report describes this as an internal retrospective reference, not truth, which matches v8.

## Initial Seed Allocation

`make_initial_seed_reveal` selects one usable fitted row per expected MITRE-step input and LLM model, with strict failure if a `(step, model)` group is missing. This gives the nominal 45-row initial seed allocation on the current dataset. Selection is reproducible by reveal seed. Terminology is aligned: the report uses “initial seed allocation,” not “burn-in.”

## Reveal Protocol After Initial Seed

### V8 Requirement

The v8 plan requires, for each outer reveal seed:

1. Pre-randomized hidden reveal orders for every MITRE-step input.
2. The same hidden reveal order shared by all allocation policies in that seed.
3. A common initial seed set.
4. When a policy selects input `j`, reveal the next unused draw for `j` according to that seed’s hidden order.
5. A model-aware reveal order: split by LLM model, shuffle repeats within model, choose a seed-specific model order, and reveal by cycling through models.

### Current Implementation

The current implementation does not do this in the main experiment loop:

- `reveal.py` has `make_uniform_reveal_order`, but it is a global random shuffle over unrevealed rows and is not a model-aware per-step hidden order.
- `run_policy_recovery` calls policy functions that return concrete fitted-row IDs.
- `choose_next_uniform_step_balanced`, `choose_next_greedy_fragility`, `choose_next_epsilon_greedy_fragility`, and `choose_next_exploration_bonus_fragility` select rows from the current unrevealed pool using a policy-specific RNG path.
- There is no shared per-step hidden order that all policies consume.
- There is no model-aware cycling after the initial seed allocation.

### Equivalence Assessment

This is not equivalent to v8 for final paired policy comparisons.

It is reproducible and paired by reveal seed in a broad sense, but it does not ensure that different policies see the same first `k` draws from a step when they allocate `k` draws to that step. Under the v8 hidden-order protocol, draw identity within a step is controlled by the seed-level hidden order and is independent of policy. Under the current implementation, within-step draw identity is selected by the policy run itself. Policies can therefore differ both in which steps they select and in which row identities they happen to receive within a selected step.

This can materially affect policy comparison because the fitted Beta rows vary by LLM model and repeat. It is especially relevant for concentrated policies: random within-step selection can accidentally over- or under-sample particular LLM models, while v8 model-aware cycling is designed to keep early revealed rows model-diverse.

### Required Change

This should be treated as a code change, followed by a final locked run. Report wording alone is not enough if the current outputs are to be used as final empirical results.

Precise patch plan:

1. Add a reveal-order primitive to `reveal.py`, for example `make_model_aware_hidden_reveal_orders(fit_df, seed)`.
2. For each MITRE-step input:
   - split usable rows by LLM model;
   - shuffle rows within each model using the reveal seed;
   - choose a seed-specific model order;
   - use one row per model for the initial seed allocation;
   - order remaining rows by cycling through models and taking the next available row for each model.
3. Return both:
   - `initial_revealed_df`, with one row per `(step, model)`;
   - `hidden_reveal_order_df`, with per-step ranks for all post-seed rows.
4. Modify policy functions or `run_policy_recovery` so policies choose a MITRE-step input, not a concrete row ID. After a policy selects a step, reveal the next unused row for that step from `hidden_reveal_order_df`.
5. Ensure the same hidden orders are shared across all policies for a given reveal seed. This can be achieved by constructing them in the repeated-experiment script and passing them to `run_policy_recovery`, or by deterministically reconstructing them inside each run without consuming policy RNG.
6. Add tests that verify:
   - hidden orders contain every usable fitted row exactly once across initial plus post-seed order;
   - initial seed has one row per `(step, model)`;
   - two policies that request the same `k` draws from a step under the same reveal seed receive the same first `k` row IDs;
   - post-seed reveal order cycles through models where model rows remain available;
   - row selection no longer depends on policy-specific RNG once a step is selected.
7. Rerun a final locked policy experiment and regenerate report artifacts from that locked output.

## LOO Fragility

The LOO implementation matches the v8 mathematical definition. It removes one currently revealed row within a node and recomputes the output movement; it does not leave out a whole node. Nodes with too few rows return NaN fragility rather than crashing. Exact mode remains available by default (`max_loo_terms_per_step=None`), and capped LOO-term subsampling is explicit and recorded.

The report correctly describes LOO fragility as finite-sample instability, not value of information or expected improvement.

## Policies

The step-choice logic for the v8 policy families is implemented:

- `uniform_step_balanced`: chooses among eligible under-sampled steps.
- `greedy_loo_fragility`: chooses the eligible step with largest finite LOO fragility.
- `epsilon_greedy_loo_fragility`: explores among under-sampled eligible steps with epsilon probability.
- `exploration_bonus_loo_fragility`: uses `F_j^LOO + lambda / sqrt(n_j)` with configurable `c`.

The c-values 0.25, 0.5, and 1.0 were evaluated in the sensitivity run. The report clearly labels that `exploration_bonus_c1.0` comes from a separate sensitivity run, not the reduced all-policy run.

However, all these policies currently select concrete row IDs within selected steps. After the reveal-order patch, the policies should choose steps and the reveal protocol should provide the next hidden row.

## Evaluation Metrics

The main metric is squared Wasserstein-2 distance to the full-data reference distribution. AUC under the error-vs-budget curve is computed as a scalar diagnostic. Concentration metrics are reported as diagnostics and not presented as the primary recovery-error metric. No unplanned core metrics such as Shapley, threshold-crossing, top-k, raw-risk, or oracle next-add-one metrics are introduced.

## Report Claims

`report/main.tex` is generally cautious. It explicitly rejects claims of optimal active learning, Bayesian experimental design, and value of information. It describes the results as development outputs and labels source-run differences.

Two report issues remain:

1. Section 5.5 cannot remain as “should be verified” in a final report. It should either describe the patched v8 reveal protocol after code changes, or explicitly state that the current outputs are development diagnostics because the reveal protocol differs from v8.
2. If a final locked run is performed, “development run” language should be replaced where appropriate with “locked experiment,” while retaining the caveats about approximate LOO if approximate LOO is used.

## Recommended Replacement for Report Section 5.5

Because the implementation currently differs from v8, there are two possible replacement texts.

### Replacement Before Code Changes

Use this only if the report continues to discuss the current development outputs:

```latex
\subsection{Budgeted reveal protocol}

All policies begin with the same initial seed allocation for each reveal seed: five rows per MITRE-step input, one from each LLM forecaster model, for a total initial budget of 45 SOTA elicitation rows. In the current development runs, additional allocation decisions reveal one fitted row at the selected MITRE-step input by seed-controlled random selection from that input's remaining unrevealed rows.

This differs from the locked v8 reveal protocol, which pre-materializes a model-aware hidden reveal order for each MITRE-step input and shares that order across all policies within a reveal seed. Under the v8 protocol, when a policy selects input \(j\), it receives the next unused row for \(j\) from the seed-specific hidden order. The current outputs should therefore be treated as development diagnostics, not final locked empirical results.
```

### Replacement After Code Changes and Locked Rerun

Use this after implementing model-aware hidden reveal orders and rerunning the locked experiment:

```latex
\subsection{Budgeted reveal protocol}

For each outer reveal seed, the experiment pre-materializes a hidden reveal order for every MITRE-step input. Within each input, valid SOTA rows are split by LLM forecaster model, repeats are shuffled within model, a seed-specific model order is chosen, and post-seed rows are ordered by cycling through models while rows remain available. The same hidden orders are shared by all allocation policies under that reveal seed.

All policies begin with the same initial seed allocation: five rows per MITRE-step input, one from each LLM forecaster model, for a total initial budget of 45 SOTA elicitation rows. After this initial seed allocation, a policy selects only the next MITRE-step input to reveal. When it selects input \(j\), it receives the next unused fitted row for \(j\) from the seed-specific hidden reveal order. This keeps policy comparisons paired by reveal seed and prevents within-step row identity from being chosen by the policy-specific random path.
```

## Final Locked-Run Readiness

The current outputs are not adequate to be treated as final empirical results. They are useful development diagnostics, but a final locked experiment should be run after resolving the reveal protocol mismatch.

Recommended final locked configuration:

- Policies:
  - `uniform_step_balanced`
  - `greedy_loo_fragility`
  - `epsilon_greedy_eps0.2`
  - `exploration_bonus_c0.25`
  - `exploration_bonus_c0.5`
  - `exploration_bonus_c1.0`
- Baseline: `uniform_step_balanced`.
- Budget checkpoints: `[45, 90, 180, 360, 720, 1200, 1798]`, where 1798 is the current all-valid-SOTA endpoint. If runtime forces a smaller schedule, keep `[45, 90, 180, 360, 720, 1200]` and explicitly state that the all-valid endpoint was not evaluated.
- Reveal seeds: fixed set of at least 10 seeds, e.g. `[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]`.
- Reference samples: at least 40,000, sampled once globally or once per locked design and reused across policies.
- Budget samples: at least 12,000 per evaluation budget; 20,000 if runtime allows.
- Distance grid: 401 or 501 quantile points.
- Fragility settings: approximate LOO with `max_loo_terms_per_step=20`, `n_samples=600` or higher, `n_grid=151` or higher, unless exact LOO is feasible on a reduced budget schedule.
- Fragility recomputation: keep `fragility_recompute_every=90` if needed for runtime, but label it as batched fragility recomputation. A smaller batch size such as 45 would be more adaptive if runtime permits.
- Expected runtime category: medium-high to high. With all six policies, 10 seeds, capped LOO, and the 1200 or 1798 budget endpoint, expect a multi-hour run on the current development machine. Exact LOO for all budgets and policies is likely impractical without a smaller budget schedule.
- Outputs should use a new locked prefix, for example `v8_locked_hidden_order_*`, and report artifacts should be regenerated from that single locked source run rather than combining development runs.

Using approximate LOO in the final locked run is acceptable if exact LOO is too expensive, because the approximation is explicit, audited, and exact mode remains available. The report must continue to label it as an approximation to the v8 mathematical definition.

## Urgent TODOs Before Treating the Report as Final

1. Implement v8 model-aware hidden reveal orders.
2. Refactor policies or experiment loop so policies choose steps and the reveal protocol supplies row IDs.
3. Add reveal-order tests as described above.
4. Run one final locked experiment with predeclared settings.
5. Regenerate report artifacts from the locked output.
6. Replace Section 5.5 with the post-patch wording.
7. Replace or qualify remaining “development run” language depending on whether the final locked run is used as the empirical result.

