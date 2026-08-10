<!-- PRIMARY-SOURCE-EXEMPT: reason=project-internal cert family doc; this cert registers a local replay/ledger contract rather than a literature claim. -->
# [524] QA Self-Improving Neural QA Cert

**Family ID**: 524
**Slug**: `qa_self_improving_neural_qa_cert_v1`
**Status**: Active
**Registered**: 2026-07-12

## Purpose

This family certifies a bounded form of self-improving neural QA:

```text
neural model -> candidate update -> deterministic replay gate -> append-only ledger
```

The neural system may propose an update, but deterministic QA replay decides
whether that update is accepted. Rejections are recorded in the same ledger so
failed gates remain visible.

This is not a claim of unconstrained autonomous weight mutation. v0 certifies a
replay-gated continual-improvement layer over neural QA candidates, including
bounded configuration/capacity proposals that require manual activation after
certification.

## Schema

Packets use `QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0`, scoped by
`docs/specs/QA_SELF_IMPROVING_NEURAL_QA_V0.md`.

| Field | Meaning |
|---|---|
| `schema_version` | Must equal `QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0`. |
| `update_id` | Stable unique update identifier. |
| `base_model` | Neural base model metadata; `base_model.neural` must be `true`. |
| `candidate` | Proposed update kind and artifact reference; `configuration_patch` and `capacity_patch` include bounded `config_patch` metadata. |
| `evidence` | Optional replay artifact reference and source replay hash; new emitters include it. |
| `replay_gate` | Fixed/harmed/protected counts plus deterministic trace hashes. |
| `invariant_checks` | Named boolean gates. |
| `promotion` | `accepted` or `rejected`; rejected packets require `rejection_reason`. |

## Validator Checks

| Check | Meaning |
|---|---|
| `SINQA_ROW` | Every ledger row is canonical JSON. |
| `SINQA_HASH` | `packet_hash` equals the domain-separated hash of the embedded packet. |
| `SINQA_UNIQ` | `update_id`, `packet_hash`, and present `evidence.source_replay_hash` values are unique. |
| `SINQA_PACKET` | Every embedded packet validates under the v0 packet validator. |
| `SINQA_ACCEPT` | Accepted packets require positive fixes, zero protected harm, deterministic replay, and all invariant checks passing. |
| `SINQA_CAPACITY` | Configuration/capacity patches require approved diff keys, resource hard caps, rollback metadata, and manual activation after cert. |
| `SINQA_REJECT` | Rejected packets require `rejection_reason` and at least one failed gate. |
| `SINQA_PASS_MIX` | PASS fixtures include at least one accepted and one rejected packet. |

## Fixtures

| Fixture | Expected | Purpose |
|---|---|---|
| `pass_ledger.jsonl` | PASS | Compact synthetic ledger with one accepted and one rejected packet. |
| `pass_capacity_patch_ledger.jsonl` | PASS | Bounded accepted capacity proposal with rollback and manual activation. |
| `pass_live_ledger.jsonl` | PASS | Current HSI-backed live ledger copied from `results/self_improving_neural_qa/ledger.jsonl`. |
| `fail_duplicate_update_id.jsonl` | FAIL | Two otherwise valid packets reuse one `update_id`. |
| `fail_accepted_harm.jsonl` | FAIL | Packet declares `accepted` while protected harm is nonzero. |
| `fail_duplicate_source_replay_hash.jsonl` | FAIL | Two otherwise valid packets reuse one replay evidence hash. |
| `fail_unbounded_capacity_patch.jsonl` | FAIL | Capacity packet exceeds the hard parameter cap. |

## Live Ledger Snapshot

The current live ledger contains:

| Metric | Value |
|---|---:|
| Rows | 24 |
| Accepted | 13 |
| Rejected | 11 |
| Accepted failures fixed | 581 |
| Accepted protected harm | 0 |
| Total protected replays | 365,574 |
| Unique evidence source hashes | 14 |

The accepted rows are the only promoted improvements. Rejected rows remain in
the ledger as safety evidence, including harm/no-fix cases.

## Family Relationships

- Builds on the HSI corrected-model/replay pipeline used to emit the first live
  packets.
- Separates neural proposal generation from QA promotion authority.
- Allows model-size/configuration proposals only through bounded replay-gated
  packets with manual activation.
- Complements the existing unregistered `qa_self_improvement_cert_v1` and
  `SelfImprovementAgentV2` work by registering a ledger-level promotion contract.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_self_improving_neural_qa_cert_v1/qa_self_improving_neural_qa_cert_validate.py --self-test
python3 tools/qa_self_improving_neural_qa_ledger_validate.py results/self_improving_neural_qa/ledger.jsonl
```
