# QA_SELF_IMPROVING_NEURAL_QA_LOOP_CERT.v1

## Scope

Validate multi-round transcript evidence for the self-improving neural QA loop.
The implementation prioritizes configuration proposals and `general_ml` replay
artifacts ahead of legacy HSI replay artifacts, while preserving HSI as a
fallback lane.

## Checks

| Check | Meaning |
|---|---|
| SINQAL_ROW | Every transcript row is canonical JSON. |
| SINQAL_HASH | `record_hash` equals the domain-separated hash of `record`. |
| SINQAL_CHAIN | `previous_record_hash` links each row to the prior row hash. |
| SINQAL_DELTA | Ledger row count increases by exactly one per round. |
| SINQAL_ACCEPT | Accepted rounds fix positive failures, harm zero protected rows, and increment accepted count. |
| SINQAL_REJECT | Rejected rounds increment rejected count and must not mutate promoted state. |
| SINQAL_UNIQ | Packet hashes and source replay hashes are unique in the transcript. |

## Live Fixture

`pass_live_transcript.jsonl` records nine bounded live rounds:

| Metric | Value |
|---|---:|
| Rounds | 9 |
| Accepted | 5 |
| Rejected | 4 |

The rejected rounds are safety evidence and do not mutate promoted state.
