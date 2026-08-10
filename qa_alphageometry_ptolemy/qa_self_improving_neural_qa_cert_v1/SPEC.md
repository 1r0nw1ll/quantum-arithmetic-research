# QA_SELF_IMPROVING_NEURAL_QA_CERT.v1

## Scope

Validate append-only ledgers for `QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0`
packets.

## Checks

| Check | Meaning |
|---|---|
| SINQA_ROW | Every ledger row is canonical JSON. |
| SINQA_HASH | `packet_hash` equals the domain-separated hash of the embedded packet. |
| SINQA_UNIQ | `update_id`, `packet_hash`, and present `evidence.source_replay_hash` values are unique within the ledger. |
| SINQA_PACKET | Every embedded packet validates under the v0 packet validator. |
| SINQA_ACCEPT | Accepted packets require positive fixes, zero protected harm, deterministic replay, and all invariant checks passing. |
| SINQA_CAPACITY | Accepted configuration/capacity patches require bounded diffs, hard resource caps, rollback metadata, and manual activation after cert. |
| SINQA_REJECT | Rejected packets require `rejection_reason` and at least one failed gate. |
| SINQA_PASS_MIX | PASS fixtures include at least one accepted and one rejected packet. |

## Fixture Intent

`pass_ledger.jsonl` is compact and synthetic. `pass_capacity_patch_ledger.jsonl`
proves that a bounded capacity proposal with rollback and manual activation can
be certified. `pass_live_ledger.jsonl` is copied from
`results/self_improving_neural_qa/ledger.jsonl` and proves the validator accepts
the current live packet shape.

`fail_duplicate_update_id.jsonl` has two valid packets with the same
`update_id`. `fail_accepted_harm.jsonl` declares an accepted packet with
protected harm, which must fail. `fail_duplicate_source_replay_hash.jsonl`
contains two otherwise valid packets emitted from the same replay evidence,
which must fail to prevent duplicate evidence from inflating the ledger.
`fail_unbounded_capacity_patch.jsonl` exceeds the hard parameter cap and must
fail even if the packet otherwise claims acceptance.

## Hash Domains

- Packet row hash: `qa_self_improving_neural_qa_packet_v0`
- Packet ledger hash inside promotion packet:
  `qa_self_improving_neural_qa_ledger_v0`

## Registration Boundary

This family is registered in `qa_meta_validator.py`.
