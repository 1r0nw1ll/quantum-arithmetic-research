# Family [29]: QA Agent Trace Schema

## Purpose

Defines and validates `QA_AGENT_TRACE_SCHEMA.v1` — a hash-chained event
trace container for recording agent actions during code-change tasks.
This is the structural layer on which competency certification is built:
traces first, metrics on top.

## Machine tract

| Artifact | Path |
|----------|------|
| JSON Schema | `qa_alphageometry_ptolemy/qa_agent_traces/schemas/QA_AGENT_TRACE_SCHEMA.v1.json` |
| Validator | `qa_alphageometry_ptolemy/qa_agent_traces/qa_agent_trace_validator.py` |
| Valid fixture | `qa_alphageometry_ptolemy/qa_agent_traces/fixtures/valid_trace.json` |
| Neg fixture (missing invariant_diff) | `qa_alphageometry_ptolemy/qa_agent_traces/fixtures/invalid_trace_missing_invariant_diff.json` |
| Neg fixture (nondeterministic order) | `qa_alphageometry_ptolemy/qa_agent_traces/fixtures/invalid_trace_nondeterministic_order.json` |

## Schema structure

A trace contains:

- **provenance**: collector, method, source dataset, license, artifact policy, privacy redactions
- **task**: benchmark type, instance ID, repo, base commit, problem ref (hash + length)
- **agent**: name, version, model, policy (tooling + guardrails)
- **environment**: OS, Python version, runner type
- **events**: ordered array of typed actions (read, search, plan, edit, run, test, error, final)
- **trace_hashes**: SHA-256 of events array, full trace hash, per-event hash chain
- **summary**: derived counters (event_count, tool_call_count, edit_count, test_run_count, outcome)

## Determinism invariants

1. `events` must be strictly increasing by `event_index`, gap-free, starting at 0
2. `hash_chain[i].event_index == events[i].event_index`
3. Each chain link: `prev_sha256` = prior link's `event_sha256` (genesis = 64 zeros)
4. `summary.event_count == len(events)` and derived counts match event types

## Typed failure algebra

| Fail type | Trigger |
|-----------|---------|
| `SCHEMA_INVALID` | Missing or malformed required fields |
| `NONDETERMINISTIC_EVENT_ORDER` | Event indices not strictly 0,1,2,... |
| `MISSING_INVARIANT_DIFF` | Event without `invariant_diff` object |
| `HASH_MISMATCH` | Recomputed hash differs from declared |
| `HASH_CHAIN_BROKEN` | `prev_sha256` doesn't match prior event hash |
| `SUMMARY_MISMATCH` | Summary counters don't match events |
| `PROVENANCE_INVALID` | Missing license or source dataset |
| `REDACTION_POLICY_VIOLATION` | Raw content in hash_only trace |

## Validation order

Gates run in this order; first failure stops:

1. Schema structural validation
2. Event ordering (determinism)
3. `invariant_diff` presence
4. Hash integrity (events hash + chain)
5. Summary consistency
6. Provenance policy checks

## Meta-validator integration

Test `[29]` in `qa_meta_validator.py` runs:
- Validator self-test (8 built-in checks)
- Valid fixture must PASS
- Missing invariant_diff fixture must FAIL with `SCHEMA_INVALID`
- Nondeterministic order fixture must FAIL with `NONDETERMINISTIC_EVENT_ORDER`

## Privacy model

The `raw_artifacts_policy` field controls what content appears in event payloads:
- `hash_only`: payloads contain only content hashes (validator enforces no long raw strings)
- `redacted`: payloads may contain redacted text
- `full`: raw content included (not recommended for redistribution)

The `privacy_redactions` array declares what categories were stripped (api_keys, tokens, paths, etc.).

## What comes next

This schema is the container layer. The next cert on top of it:
`QA_AGENT_TRACE_COMPETENCY_CERT.v1` — consumes a trace + task record,
emits competency metrics, failure algebra, and dominance lattice.
