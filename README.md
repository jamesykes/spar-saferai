# Fragility-guided elicitation for AI-cyber risk models

Expert elicitation is one of the expensive parts of building quantitative risk models. This project asks whether a limited elicitation budget can be used more efficiently by directing new elicitations towards the inputs that currently have the greatest effect on the model's output distribution.

We study this retrospectively using SaferAI's OC3 Denial-of-Service model. The analysis is restricted to the state-of-the-art (SOTA) capability condition and the nine-node $P(\mathrm{success})$ submodel. Each elicitation supplies three probability quartiles, which are fitted with a Beta distribution. At any point in the simulated elicitation process, the revealed distributions for each node form an exchangeable mixture and are propagated through the risk model. Allocation policies are evaluated by how quickly they recover the output distribution obtained from the full dataset.

The main heuristic is a leave-one-out (LOO) fragility score: for each node, we measure how much the current output distribution moves when individual revealed elicitations are removed. The experiments compare several ways of turning this signal into an allocation policy against a step-balanced uniform baseline.

## Main results

The raw dataset contains 3,600 elicitation rows. Cleaning leaves 3,587 valid rows, including 1,798 in the SOTA condition. Of the 1,798 fitted SOTA distributions, 1,790 pass the main fit-quality threshold, eight receive warnings, and none fail. The full-data Monte Carlo reference has mean $P(\mathrm{success}) \approx 0.224$ and median $\approx 0.220$.

The policy results come from three experiments with 30 shared reveal seeds and 40 budgets between 45 and 1,798 revealed elicitations. The main findings are:

- Step-balanced uniform allocation is a strong baseline and has the lowest all-budget error AUC in each dense experiment.
- Deterministic fragility policies perform poorly. Greedy and exploration-bonus variants concentrate the budget too heavily on a small number of nodes and often produce substantially more error than uniform allocation.
- Stochastic policies are much more competitive. Stochastic epsilon-greedy has lower median error than uniform at 22 of the 38 non-boundary budgets and a strict paired win rate above 50% at 21. Uniform sampling among nodes with positive fragility does so at 20 budgets on both measures.
- The softmax ablation tells a similar story: the flattest tested policy ($T=4$) has lower median error than uniform at 20 of 38 budgets, whereas the sharpest ($T=0.25$) does so at only three.

These results suggest that LOO fragility contains some useful information, but its magnitude is not calibrated well enough to use aggressively. In this setting, preserving substantial randomisation matters more than selecting the node with the largest estimated score. The evidence supports further work on output-sensitive elicitation, not a claim that fragility guidance reliably beats uniform allocation.

## Repository structure

- `data/` contains the source elicitation data and reproducible cleaned datasets.
- `src/saferai_budget_recovery/` contains the data, fitting, sampling, forward-model, fragility, policy, and reporting code.
- `scripts/` contains the numbered data-processing, validation, experiment, and report-artifact workflows.
- `outputs/` contains validation results and the three retained 30-seed experiment families.
- `report/` contains the LaTeX manuscript, bibliography, tables, figures, and reference material.
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

To rebuild the report, run `latexmk -pdf main.tex` from `report/`.
