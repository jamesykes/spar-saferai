# Fragility-Guided Elicitation Budget Allocation for AI-Cyber Risk Modeling with Bayesian Networks

SaferAI’s quantitative AI-cyber risk models are parametrized with probability distributions elicited from experts, and this expert elicitation process often proves to be an expensive bottleneck. We investigate whether elicitation effort can be allocated more efficiently by prioritizing inputs that appear to have the greatest effect on the output probability distribution. In order to investigate this, we perform a retrospective budget-recovery experiment on SaferAI’s OC3 Denial-of-Service risk model, restricted to the SOTA capability condition and the $P(\text{success})$ submodel. The experiment compares a uniform baseline with several fragility-guided allocation heuristics.

Each elicitation supplies three probability quartiles, which are fitted with a Beta distribution. At any point in the simulated elicitation process, the revealed distributions for each node form an exchangeable mixture and are propagated through the risk model. Allocation policies are evaluated by how quickly they recover the output distribution obtained from the full dataset.

The main heuristic is a leave-one-out (LOO) fragility score: for each node, we measure how much the current output distribution moves when individual revealed elicitations are removed. The experiments compare several ways of turning this signal into an allocation policy.

## Main results

The policy results come from three experiments with 30 shared reveal seeds and 40 budgets between 45 and 1,798 revealed elicitations. The main findings are:

- Deterministic fragility policies perform poorly. Greedy and exploration-bonus variants concentrate the budget too heavily on a small number of nodes and often produce substantially more error than uniform allocation.
- Stochastic policies perform much more strongly. The stochastic epsilon-greedy policy, for example, has lower median error than uniform at the majority of the budgets.
- The softmax ablation also shows that for reasonable choices of the temperature parameter, we again perform better than the uniform baseline. However, no clear improvement over the best performing standard stochastic policies is exhibited.

## Repository structure

- `data/` contains the source elicitation data and reproducible cleaned datasets.
- `scripts/` contains the numbered data-processing, validation, experiment, and report-artifact workflows.
- `outputs/` contains validation results and the three retained 30-seed experiment families.
- `report/` contains tables, figures, planning and reference material.
- `tests/` contains the automated test suite.

## Running the project

The project requires Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/00_check_environment.py
.venv/bin/python scripts/01_clean_data.py
.venv/bin/python scripts/02_fit_beta_distributions.py
.venv/bin/python -m pytest
```

The later numbered scripts run the forward-model and fragility checks, repeated allocation experiments, and report generation. The dense policy experiments are computationally expensive; the checked-in outputs allow the reported analysis and figures to be inspected without rerunning them.