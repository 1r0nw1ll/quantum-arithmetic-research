# QA Self-Improving Neural QA Activation Cert [527]

## Purpose

Cert [527] validates activation-plan evidence for accepted self-improving neural
QA configuration and capacity patches. It closes the gap between "proposal was
accepted into the ledger" and "runtime config may change" by making activation a
separate, auditable artifact.

## Schema

| Field | Meaning |
|---|---|
| `schema_version` | Must be `QA_SELF_IMPROVING_NEURAL_QA_ACTIVATION_PLAN.v0`. |
| `update_id`, `packet_hash` | The accepted config/capacity packet being planned. |
| `activation_policy` | Must be `manual_after_cert`. |
| `runtime_mutated` | Whether the plan claims runtime state changed. |
| `config_diff` | Proposed config/resource diff. |
| `resource_bounds` | Hard limits for parameters, memory, and runtime. |
| `rollback` | Rollback artifact ref/hash and computed hash. |
| `replay_gate` | Deterministic replay counts from the accepted packet. |
| `plan_hash` | Domain-separated hash of the canonical plan body. |

## Validator Checks

| Check | Requirement |
|---|---|
| Schema/manual gate | Correct schema, manual approval, `manual_after_cert`. |
| Resource bounds | Parameters, memory, and runtime are below hard caps. |
| Rollback | Artifact exists and computed hash matches the recorded hash. |
| Replay gate | Deterministic replay, positive fixes, zero protected harm. |
| Mutation proof | `runtime_mutated=true` requires post-activation replay and rollback proof. |
| Plan hash | `plan_hash` matches the canonical plan body. |

## Fixtures

| Fixture | Expected | Purpose |
|---|---:|---|
| `pass_activation_plan.json` | PASS | Real non-mutating plan for the accepted general-ML capacity packet. |
| `fail_runtime_mutated_without_proof.json` | FAIL | Rejects a plan that claims mutation without post-activation evidence. |
| `fail_bad_rollback_hash.json` | FAIL | Rejects rollback hash drift. |

## Family Relationships

Builds on [524] packet/ledger promotion, [525] loop transcript validation, and
[526] scheduled-run validation. Cert [527] is the manual activation boundary:
accepted capacity proposals remain proposals until this activation-plan evidence
passes.
