# Lean 4 Blind Eval Harness

This directory extends the blind-eval pattern to a second domain: Lean 4
proofs. The purpose is to test whether the anti-slop legitimacy controls built
for TLA+ generalize to proof-assistant work, where formal-looking structure can
still hide theorem mismatch, weak source fidelity, or low external admissibility.

The suite is parallel to `evals/tla_blind/` and keeps the same basic shape:

- blind generation with hidden references kept out of prompts
- blind review of mixed-quality proof bundles
- blind repair where some artifacts should still remain rejected
- a deterministic current-system executor for starter-corpus scoring
- hidden labels/reference material separated from model-visible prompts

## Layout

```text
evals/lean4_blind/
├── README.md
├── runner.py
├── execute_current_system.py
├── test_execute_current_system.py
├── rubrics/
│   ├── rubric.md
│   └── scorecard_schema.json
├── tasks/
│   └── generation/
│       ├── known_good_add_zero/
│       └── sparse_double_even/
├── review_corpus/
│   ├── good_add_zero_example/
│   ├── polished_bad_group_proof/
│   └── sparse_legit_even_double/
├── repair_cases/
│   ├── reject_fake_theorem_scope/
│   └── revise_induction_explanation/
└── results/
    └── README.md
```

## Layers

### 1. Blind generation

The system sees a natural-language Lean task but not the hidden reference
solution. It must produce:

- Lean 4 proof artifact(s)
- theorem statement explanation
- source grounding
- short outsider-facing note on why the theorem/proof is worth reviewing

### 2. Blind review

The system reviews proof bundles without knowing authorship or hidden labels.
It must return:

- `accept`, `revise`, or `reject`
- a rationale
- a scorecard using the Lean 4 rubric

### 3. Blind repair

The system receives a flawed proof bundle and must:

- diagnose the failure mode
- decide whether repair is sufficient
- distinguish `revise` cases from `reject even after repair` cases

## Required Score Axes

Every Lean 4 scorecard must include:

- `formal_validity_score`
- `external_admissibility_score`
- `proof_correctness_score`
- `theorem_statement_fidelity_score`
- `source_fidelity_score`
- `math_explanation_quality_score`
- `reviewer_rejection_risk_score`

## Runner

Examples:

```bash
python3 evals/lean4_blind/runner.py list
python3 evals/lean4_blind/runner.py show generation known_good_add_zero
python3 evals/lean4_blind/runner.py init-scorecard review polished_bad_group_proof
python3 evals/lean4_blind/runner.py --self-test --json
python3 evals/lean4_blind/execute_current_system.py
```

## CI / Workflow Fit

This suite does not replace the TLA+ gate or the TLA+ blind suite. It checks
whether the same general harness philosophy carries over to proof-assistant
work:

- formal validity is not enough
- theorem statement fidelity matters
- source fidelity matters
- polished proof-shaped artifacts can still be externally inadmissible

