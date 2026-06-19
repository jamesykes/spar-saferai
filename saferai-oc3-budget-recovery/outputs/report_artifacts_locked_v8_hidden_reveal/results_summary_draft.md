# Locked V8 Hidden-Reveal Results Summary Draft

## Source Run

All report artifacts in this directory use only `locked_v8_hidden_reveal` outputs.

The report JSON says:

- `mode = LOCKED_V8_HIDDEN_REVEAL`
- `uses_hidden_reveal_orders = True`
- `hidden_reveal_order_protocol = shared_per_step_model_aware_orders`
- `settings_reduced_from_requested = False`

The locked run used the shared hidden reveal-order protocol. For each reveal seed, per-step hidden reveal orders were shared across policies, and policies selected only the MITRE-step input to reveal next.

## Locked Settings

- Policies run: 6
- Reveal seeds: 10 ([101, 202, 303, 404, 505, 606, 707, 808, 909, 1010])
- Budgets: [45, 90, 180, 360, 720, 1200, 1798]
- Approximate LOO cap for fragility-guided policies: `max_loo_terms_per_step=20`
- Requested settings reduced: False

## Main Recovery-Error Result

The main metric is squared Wasserstein-2 error to the full-data exchangeable reference distribution for `P(success)`. Lower AUC under the error-vs-budget curve indicates better aggregate recovery over the selected budget grid.

The lowest locked average AUC is `exploration_bonus_c1.0` with average AUC `0.00940892088118`. The uniform baseline average AUC is `0.0110745610315`.

Under the corrected hidden reveal-order protocol, the best average-AUC policy was therefore an exploration-bonus fragility policy. However, all fragility-guided policies had paired win fractions below 0.5 against the uniform baseline, so the AUC advantage is aggregation-dependent and should be interpreted cautiously.

## Concentration Result

At budget 1200, the lowest concentration among exploration-bonus settings was `c=1.0` with mean L1 imbalance `516`. Uniform remains balanced by construction, with mean L1 imbalance 0 at budget 1200.

At budget 1798, all policies have mean, median, and max L1 imbalance 0. This is expected because the full available fitted SOTA dataset has been revealed.

## Exploration-Bonus Sensitivity

The locked run included exploration-bonus settings `c=0.25`, `c=0.5`, and `c=1.0`. Among these, `c=1.0` had the lowest average AUC. Larger `c` also reduced concentration at budget 1200 in this locked run.

This suggests that stronger exploration pressure may help control over-concentration in LOO-fragility allocation. It does not show that `c=1.0` is prospectively optimal.

## Caveats

The locked run provides weak-to-moderate evidence that fragility-guided policies can improve aggregate recovery error, especially with a stronger exploration bonus. It does not show majority paired wins over uniform.

The full-data reference is not ground truth. It is the output distribution from all valid SOTA LLM elicitation rows under the exchangeable nodewise mixture approximation.

The budget unit is an LLM elicitation draw, not a human expert.

The experiment targets the OC3 DoS `P(success)` submodel, not full total-risk uncertainty.

LOO fragility measures finite-sample instability of the current nodewise mixture. It is not a value-of-information estimate, not optimal active learning, and not Bayesian experimental design.

Fragility-guided policies used approximate LOO fragility with `max_loo_terms_per_step=20`.
