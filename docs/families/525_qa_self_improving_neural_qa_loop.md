<!-- PRIMARY-SOURCE-EXEMPT: reason=project-internal cert family doc; this cert registers a local loop transcript contract rather than a literature claim. -->
# [525] QA Self-Improving Neural QA Loop Cert

**Family ID**: 525
**Slug**: `qa_self_improving_neural_qa_loop_cert_v1`
**Status**: Active
**Registered**: 2026-07-12

## Purpose

This family certifies multi-round continual learning for the self-improving
neural QA runner introduced by [524].

The runner remains bounded: neural artifacts propose candidate updates, but
deterministic QA replay and ledger validation decide each commit.

The live runner also enforces a candidate-priority rule so general learning is
not starved by the older HSI replay backlog: configuration proposals first,
then `general_ml` replay artifacts, then other non-HSI replay artifacts, with
HSI retained as fallback.

## Transcript Schema

Each transcript row is canonical JSON:

```json
{"record_hash":"64 hex chars","record":{}}
```

The `record` includes:

| Field | Meaning |
|---|---|
| `schema_version` | Must equal `QA_SELF_IMPROVING_NEURAL_QA_LOOP_ROUND.v0`. |
| `round` | Absolute transcript round number. |
| `previous_record_hash` | Hash-chain pointer to the prior row, or 64 zeroes for the first row. |
| `ledger_hash_before` / `ledger_hash_after` | Domain-separated ledger file hashes. |
| `ledger_summary_before` / `ledger_summary_after` | Validated ledger counts before and after the round. |
| `packet_hash` | Candidate packet hash committed in the round. |
| `decision` | `accepted` or `rejected`. |
| `fixed`, `harmed`, `protected` | Replay-gate counters. |
| `source_replay_hash` | Evidence hash of the replay artifact. |
| `promoted_state_mutated` | `true` only for accepted rounds. |

## Validator Checks

| Check | Meaning |
|---|---|
| `SINQAL_ROW` | Every row is canonical JSON. |
| `SINQAL_HASH` | `record_hash` matches the domain-separated hash of `record`. |
| `SINQAL_CHAIN` | `previous_record_hash` links rows in order. |
| `SINQAL_DELTA` | Ledger row count increases by exactly one per round. |
| `SINQAL_ACCEPT` | Accepted rounds fix positive failures, harm zero protected rows, and increment accepted count. |
| `SINQAL_REJECT` | Rejected rounds increment rejected count and do not mutate promoted state. |
| `SINQAL_UNIQ` | Packet hashes and replay evidence hashes are unique. |

## Live Transcript

The live transcript fixture records a bounded nine-round run:

| Metric | Value |
|---|---:|
| Rounds | 9 |
| Accepted | 5 |
| Rejected | 4 |

The rejected rounds are preserved as safety evidence and do not mutate promoted
state.

## Verification

```bash
python3 tools/qa_self_improving_neural_qa_transcript_validate.py results/self_improving_neural_qa/loop_transcript.jsonl
python3 qa_alphageometry_ptolemy/qa_self_improving_neural_qa_loop_cert_v1/qa_self_improving_neural_qa_loop_cert_validate.py --self-test
```

Builds on cert [524].
