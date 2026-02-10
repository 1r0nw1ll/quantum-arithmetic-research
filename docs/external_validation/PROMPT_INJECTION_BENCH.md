# Prompt Injection External Validation

## What this is

This external-validation check runs `qa_guardrail` against a labeled
benchmark-style corpus of prompt-injection and benign prompts. The harness
reports classification metrics (precision/recall/F1/accuracy) and verifies
that blocked attacks produce a typed obstruction:
`POLICY_CONSTRAINT_VIOLATION`.

It is wired into `qa_meta_validator.py` as test `[30]` (after Level-3
generalization recompute test `[29]`).

## Dataset

Frozen local corpus:

`qa_alphageometry_ptolemy/external_validation_data/prompt_injection_benchmark_subset.jsonl`

Each row contains:

- `case_id`
- `benchmark_family` (`TensorTrust` | `Lakera` | `Gandalf`)
- `label` (`attack` | `benign`)
- `prompt`
- `source_note`

The local corpus is intentionally small and deterministic so CI remains fast
and reproducible.

## How to run

```bash
# Full report
python3 qa_alphageometry_ptolemy/external_validation_prompt_injection.py

# CI mode (single line)
python3 qa_alphageometry_ptolemy/external_validation_prompt_injection.py --ci
```

Optional throttle:

```bash
QA_PI_MAX_CASES=12 python3 qa_alphageometry_ptolemy/external_validation_prompt_injection.py --ci
```

## Pass criteria

The gate passes only when all are true:

- accuracy = 1.0
- recall = 1.0
- false positives = 0
- typed obstruction mismatches = 0

This is strict by design: regressions are surfaced immediately in CI.

## Outputs

Written to:

- `qa_alphageometry_ptolemy/external_validation_certs/prompt_injection_summary.json`
- `qa_alphageometry_ptolemy/external_validation_certs/prompt_injection_case_results.json`

## Scope and limits

This validates scanner behavior on a frozen benchmark-style subset, not full
coverage of all real-world prompt-injection techniques. It is intended as a
deterministic external check and should be complemented by larger benchmark
runs when available.
