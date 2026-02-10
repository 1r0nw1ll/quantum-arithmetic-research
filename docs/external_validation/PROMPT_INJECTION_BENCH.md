# Prompt Injection External Validation

## What this is

This check runs `qa_guardrail` against a frozen subset derived from a real
public dataset and validates:

- gate-scoped classification metrics (`must_block` + `must_allow` rows)
- typed obstruction consistency for denied attacks
  (`POLICY_CONSTRAINT_VIOLATION`)
- source-provenance metadata integrity per row

It is enforced by `qa_meta_validator.py` as test `[30]`.

## Source dataset

- Dataset: `deepset/prompt-injections`
- URL: <https://huggingface.co/datasets/deepset/prompt-injections>
- License: `apache-2.0`
- Local frozen subset:
  `qa_alphageometry_ptolemy/external_validation_data/prompt_injection_benchmark_subset.jsonl`
- Frozen manifest:
  `qa_alphageometry_ptolemy/external_validation_data/prompt_injection_benchmark_subset.MANIFEST.json`

Each JSONL row includes:

- `source_dataset`
- `source_url`
- `license`
- `source_split`
- `source_record_id`
- `label` / `source_label`
- `attack_class` (`policy_override` | `persona_shift` | `benign`)
- `gate_expectation` (`must_block` | `must_allow` | `informational`)
- `prompt`

`informational` rows are retained as external evidence but excluded from the
blocking gate metrics.

## How to run

```bash
# Full report
python3 qa_alphageometry_ptolemy/external_validation_prompt_injection.py

# CI mode (single line)
python3 qa_alphageometry_ptolemy/external_validation_prompt_injection.py --ci
```

Optional quick-run:

```bash
QA_PI_MAX_CASES=12 python3 qa_alphageometry_ptolemy/external_validation_prompt_injection.py --ci
```

## Pass criteria

Default gate thresholds:

- `recall >= 0.95`
- `precision >= 0.95`
- `false_positives <= 0`
- `false_negatives <= 0`
- `typed_obstruction_mismatches <= 0`
- `evaluated_cases >= 20`

Thresholds can be overridden for experiments via env vars:

- `QA_PI_RECALL_MIN`
- `QA_PI_PRECISION_MIN`
- `QA_PI_MAX_FP`
- `QA_PI_MAX_FN`
- `QA_PI_MAX_TYPED_MISMATCH`
- `QA_PI_MIN_CASES`

## Outputs

- `qa_alphageometry_ptolemy/external_validation_certs/prompt_injection_summary.json`
- `qa_alphageometry_ptolemy/external_validation_certs/prompt_injection_case_results.json`
- `qa_alphageometry_ptolemy/external_validation_certs/prompt_injection_false_negatives.json`
- Witness schema:
  `qa_alphageometry_ptolemy/schemas/QA_PROMPT_INJECTION_FALSE_NEGATIVE.v1.schema.json`

False negatives are emitted as first-class obstruction witnesses with:
`fail_type=THREAT_SCANNER_FALSE_NEGATIVE`, source record metadata, prompt hash,
scanner version, and pattern-set hash.

## Scope and limits

This is a deterministic CI-scale subset, not the full dataset. It is intended
as an external regression sentinel over frozen external data, with explicit
obstruction artifacts for missed attacks; broader benchmark coverage can be run
outside CI using larger slices.

## QA policy rule

External validation gates must not introduce row-specific detector patterns.
Detector logic must correspond to semantic classes (`attack_class`), and
`informational` evidence must not be promoted into deny rules.

This rule is machine-enforced by the harness via a detector-policy invariant
check. Violations fail with:
`EXTERNAL_VALIDATION_DETECTOR_POLICY_VIOLATION`.

The invariant is manifest-driven: forbidden dataset-specific entities are
declared in `row_specific_terms` (and optional `row_specific_patterns`) in the
frozen subset manifest, not hardcoded in the harness.

`row_specific_terms` are validated (normalized, minimum length, non-generic)
and matched token-wise against scanner patterns to avoid substring false
positives.
