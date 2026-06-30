# SaferAI OC3 Budget Recovery

Reproducible Python-script workflow for the first pass of the SaferAI OC3 Denial of Service budget-recovery experiment.

This pass is deliberately narrow:

- load the repeated LLM elicitation CSV;
- validate and clean quartile rows;
- derive globally unique draw IDs;
- map task names to capability levels;
- write processed CSVs and sanity-check outputs;
- test the cleaning logic with synthetic data.

It does not implement Beta fitting, Monte Carlo propagation, Wasserstein distances, reveal orders, fragility scores, policies, or allocation experiments.

## Scope

The final-report experiment is restricted to:

- risk model: OC3 Denial of Service;
- component: `P(success)` submodel only;
- capability level: SOTA only;
- elicitation budget unit: one LLM elicitation draw.

See `report/planning_docs/saferai_mixture_fragility_allocation_plan_v8.md` for the implementation plan and terminology.

## Run

From the repository root, using Python 3.12:

```bash
python scripts/00_check_environment.py
python scripts/01_clean_data.py
pytest
```

If `python` is not on `PATH`, use the repository virtual environment:

```bash
.venv/bin/python scripts/00_check_environment.py
.venv/bin/python scripts/01_clean_data.py
.venv/bin/python -m pytest
```
