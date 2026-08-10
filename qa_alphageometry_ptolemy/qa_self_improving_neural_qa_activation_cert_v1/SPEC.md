# QA Self-Improving Neural QA Activation Cert v1

## Claim

An accepted self-improving neural QA `capacity_patch` or `configuration_patch`
may produce an activation plan only if the plan is manually activated, bounded by
hard resource caps, rollback-verifiable, and replay-backed with zero protected
harm. Plans with `runtime_mutated=false` are allowed as planning evidence. Plans
with `runtime_mutated=true` are rejected unless they include post-activation
replay evidence and rollback proof.

## Checks

| Check | Meaning |
|---|---|
| `SINQAA_SCHEMA` | Plan schema is `QA_SELF_IMPROVING_NEURAL_QA_ACTIVATION_PLAN.v0`. |
| `SINQAA_MANUAL` | `activation_policy=manual_after_cert` and manual approval is required. |
| `SINQAA_RESOURCE` | Parameters, memory, and runtime remain under hard caps. |
| `SINQAA_ROLLBACK` | Rollback artifact exists and hashes to the recorded value. |
| `SINQAA_REPLAY` | Source replay gate is deterministic, fixes positive failures, and harms zero protected cases. |
| `SINQAA_MUTATION` | Runtime mutation requires post-activation replay and rollback proof. |
| `SINQAA_HASH` | `plan_hash` matches the canonical plan body. |

## Boundary

This cert is not an automatic deployment mechanism. It validates activation-plan
evidence and explicitly rejects silent runtime mutation.
