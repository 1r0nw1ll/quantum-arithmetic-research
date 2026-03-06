# Family [30]: QA Agent Trace Competency Certificate

## Purpose

Defines and validates `QA_AGENT_TRACE_COMPETENCY_CERT_SCHEMA.v1` — a
certificate that binds an agent trace to a task reference, derives
competency metrics deterministically, and emits a dominance-lattice
verdict against named baselines.

This is the evaluation layer built on top of Family [29] (trace container).

## Machine tract

| Artifact | Path |
|----------|------|
| JSON Schema | `qa_alphageometry_ptolemy/qa_agent_traces/schemas/QA_AGENT_TRACE_COMPETENCY_CERT_SCHEMA.v1.json` |
| Validator | `qa_alphageometry_ptolemy/qa_agent_traces/qa_agent_trace_competency_cert_validator.py` |
| Valid fixture | `qa_alphageometry_ptolemy/qa_agent_traces/fixtures/competency_cert_valid.json` |
| Neg fixture (bad trace hash) | `qa_alphageometry_ptolemy/qa_agent_traces/fixtures/competency_cert_invalid_bad_trace_hash.json` |
| Neg fixture (missing invariant_diff) | `qa_alphageometry_ptolemy/qa_agent_traces/fixtures/competency_cert_invalid_missing_invariant_diff.json` |
| Neg fixture (bad dominance) | `qa_alphageometry_ptolemy/qa_agent_traces/fixtures/competency_cert_invalid_bad_dominance.json` |

## Certificate structure

A competency cert binds:

- **trace_ref**: trace_id + trace_sha256 + trace_schema_id (cryptographic binding to a specific trace)
- **task_ref**: task_id + source_dataset + dataset_slice_sha256 + license (feature-only task reference)
- **derivation**: versioned spec for how metrics are computed from trace events (event types, action vocab, timebase, canonicalization params)
- **metrics**: four Levin-aligned competency dimensions (agency_index, plasticity_index, goal_density, control_entropy)
- **baselines**: baseline_set_id + baseline_bundle_sha256 + list of baselines used (each with kind + optional metrics for deterministic dominance recomputation)
- **dominance**: rule (strong_4of4 / weak_3of4 / existential_1of4) + min_margin + per-dimension win/loss
- **verdict**: passed + fail_type (typed, deterministic)
- **invariant_diff**: always present, explains pass/fail for CI

## Typed failure algebra

| Fail type | Trigger |
|-----------|---------|
| `SCHEMA_INVALID` | Missing or malformed required fields |
| `MISSING_INVARIANT_DIFF` | Certificate without `invariant_diff` object |
| `TRACE_REF_HASH_MISMATCH` | trace_sha256 doesn't match provided trace |
| `METRIC_OUT_OF_BOUNDS` | Metric value outside declared bounds |
| `NONDETERMINISTIC_DERIVATION` | dimensions_won count doesn't match bools, dominance recomputation mismatch, or verdict inconsistency |
| `DOMINANCE_FAILURE` | Dominance rule not satisfied (e.g., weak_3of4 with only 1 win) |

## Validation order

1. `invariant_diff` presence (hard gate)
2. Schema structural validation
3. Trace ref hash verification (if trace provided)
4. Metric bounds check
5. Dominance consistency (dims count + rule satisfaction)
5b. Dominance recomputation (if baseline metrics present: verifies `by_dimension` booleans match `agent_metric - baseline_metric >= min_margin`)
6. Verdict internal consistency

## Dominance rules

| Rule | Requirement |
|------|-------------|
| `strong_dominance_4of4` | Agent wins all 4 dimensions |
| `weak_dominance_3of4` | Agent wins >= 3 of 4 dimensions |
| `existential_1of4` | Agent wins >= 1 dimension |

Current default: `existential_1of4` (matches SWE-bench competency gate).
Future tightening to `weak_dominance_3of4` once real agent traces are available.

## Scope and limits

- This cert validates **shape + consistency + derivation-correctness**, not real-world correctness
- When baseline metrics are present, dominance wins are **recomputed deterministically** and verified against declared values
- Baseline binding is by hash, not by running baselines (that's the baseline family, future)
- Metrics in fixtures are synthetic — real derivation requires real traces
- The `derivation.derivation_id` is locked to `QA_COMPETENCY_FROM_AGENT_TRACE.v1`

## What comes next

- `QA_BASELINE_PROFILE_SCHEMA.v1` — formalize baseline bundles so dominance lattice references are fully hash-bound
- Real agent traces from live runs → real competency certs
- Dominance lattice upgrade from existential to weak/strong as data accumulates
