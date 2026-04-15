# QA Experiment Protocol (Machine Tract)

Defines **QA_EXPERIMENT_PROTOCOL.v1**: an enforceable design contract
for any empirical QA study — hypothesis, null design, source mapping,
ablation, reproducibility, decision rules, observer projection, and
pre-registration.

Authority: `EXPERIMENT_AXIOMS_BLOCK.md` (Part A, E1–E6; Part C, N1–N3).

## What it is

Every empirical experiment must produce a concrete object:

`X = (H, N, P, D, O, R, S, A, M)` where

- `H`: hypothesis (falsifiable)
- `N`: null model (generating process, held-fixed fields, permuted fields,
  independence argument)
- `P`: pre-registration (seed, date, n_trials)
- `D`: decision rules (accept, reject, on_unsupportive)
- `O`: observer projection (how continuous → (b,e))
- `R`: real-data status (path | "pending" | "synthetic_only")
- `S`: source mapping cross-reference (`primary_source` must occur in `theory_doc`)
- `A`: ablation contract
- `M`: reproducibility manifest

## Files

- `schema.json` — JSON Schema for `QA_EXPERIMENT_PROTOCOL.v1`
- `validator.py` — Gate 1–9 validator
- `canonical_experiment_protocol.json` — canonical shared object
- `fixtures/valid_min.json` — minimal passing fixture
- `fixtures/invalid_missing_null_independence.json` — intentionally failing fixture

## How to run

```bash
python qa_experiment_protocol/validator.py --self-test
python qa_experiment_protocol/validator.py qa_experiment_protocol/fixtures/valid_min.json
```

## How empirical scripts declare compliance

Inline form (experiment JSON lives next to the script):

```python
EXPERIMENT_PROTOCOL_REF = "path/to/experiment_protocol.json"
```

Or place `experiment_protocol.json` in the same directory as the script.
`qa_axiom_linter.py` rule `EXP-1` gates both forms.
