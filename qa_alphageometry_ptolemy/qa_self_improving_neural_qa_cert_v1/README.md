# QA Self-Improving Neural QA Cert v1

Status: active registered cert family.

## Purpose

This family validates the ledger discipline for Self-Improving Neural QA v0:
a neural model may propose an update, but deterministic QA replay decides
whether the update is promoted.

The cert validates both outcomes:

- `accepted`: positive incremental fix, zero protected harm, deterministic
  replay, and all invariant checks pass.
- `rejected`: failed gate evidence is preserved with an explicit
  `rejection_reason`.

Rejected packets are valid ledger entries because they prevent survivorship
bias in the self-improvement history.

Configuration and capacity changes are allowed only as bounded proposals. They
must carry a `config_patch` diff, hard resource bounds, rollback metadata, and
`activation_policy: manual_after_cert`. The unattended learner does not apply
these changes to launchd, scheduler caps, or model shape by itself.

## Artifacts

| Artifact | Path |
|---|---|
| Validator | `qa_self_improving_neural_qa_cert_validate.py` |
| Mapping ref | `mapping_protocol_ref.json` |
| Spec | `SPEC.md` |
| Synthetic pass fixture | `fixtures/pass_ledger.jsonl` |
| Capacity pass fixture | `fixtures/pass_capacity_patch_ledger.jsonl` |
| Live pass fixture | `fixtures/pass_live_ledger.jsonl` |
| Duplicate-id fail fixture | `fixtures/fail_duplicate_update_id.jsonl` |
| Accepted-harm fail fixture | `fixtures/fail_accepted_harm.jsonl` |
| Unbounded-capacity fail fixture | `fixtures/fail_unbounded_capacity_patch.jsonl` |

## How to Run

```bash
python3 qa_alphageometry_ptolemy/qa_self_improving_neural_qa_cert_v1/qa_self_improving_neural_qa_cert_validate.py --self-test
```

The validator intentionally reuses the v0 packet semantics from
`tools/qa_self_improving_neural_qa_v0.py`.

## Non-Claims

This cert does not claim autonomous safe self-modification, model-quality
improvement in general, or LLM-scale continual learning. It certifies that
update packets and their append-only ledger obey the declared replay-gate
semantics, including bounded capacity proposals that still require manual
activation after certification.
