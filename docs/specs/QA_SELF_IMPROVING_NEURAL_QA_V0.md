# Self-Improving Neural QA v0

Status: scaffold contract, not a live cert.

## Goal

Build toward "self-improving neural QA" without allowing silent neural drift.
The v0 system treats a neural model as a proposal generator and a QA replay
gate as the promotion authority.

```text
neural model -> candidate update -> deterministic replay gate -> append-only ledger
```

An update is promoted only when it fixes new failures, harms zero protected
cases, preserves declared invariants, and records deterministic trace hashes.

## Boundary

This is not yet an autonomous self-modifying neural network. v0 permits:

- correction rules mined from neural residuals,
- small adapters or correction heads proposed by a neural model,
- routing changes suggested by embeddings or uncertainty,
- generator-pattern candidates proposed by neural diagnostics.
- bounded configuration/capacity proposals with manual activation after cert.

v0 does not permit direct mutation of base neural weights without a replay
artifact. Weight updates may be trained, but promotion is gated by QA replay.
Configuration and capacity changes may be proposed only as bounded
`configuration_patch` or `capacity_patch` packets. They require rollback
metadata, resource bounds, replay evidence, and `activation_policy:
manual_after_cert`; the unattended learner does not rewrite its own launchd
job, runtime caps, or model size directly.

## Update Packet

Each candidate promotion is a JSON packet with schema. Accepted packets use
`promotion.decision == "accepted"`; rejected packets use
`promotion.decision == "rejected"` plus `promotion.rejection_reason`.

```json
{
  "schema_version": "QA_SELF_IMPROVING_NEURAL_QA_UPDATE.v0",
  "update_id": "sinqa.v0.example",
  "base_model": {
    "name": "hsi_ensemble",
    "kind": "classifier",
    "neural": true
  },
  "candidate": {
    "kind": "correction_rule",
    "description": "fix residual confusion class 7 -> 5",
    "artifact_ref": "results/example_rules.json"
  },
  "evidence": {
    "source_replay_ref": "results/example_replay.json",
    "source_replay_hash": "64 hex chars"
  },
  "replay_gate": {
    "new_failures_fixed": 3,
    "protected_cases_replayed": 100,
    "protected_cases_harmed": 0,
    "deterministic_replay": true,
    "trace_hash_before": "64 hex chars",
    "trace_hash_after": "64 hex chars"
  },
  "invariant_checks": [
    {"name": "zero_harm", "passed": true},
    {"name": "integer_state", "passed": true}
  ],
  "promotion": {
    "decision": "accepted",
    "ledger_hash": "64 hex chars"
  }
}
```

## Acceptance Rule

The validator accepts a packet only if:

- `base_model.neural == true`;
- `candidate.kind` is one of `correction_rule`, `adapter`, `routing_patch`,
  `generator_pattern`, `calibration_patch`, `configuration_patch`, or
  `capacity_patch`;
- `new_failures_fixed > 0`;
- `protected_cases_replayed > 0`;
- `protected_cases_harmed == 0`;
- `deterministic_replay == true`;
- every invariant check has `passed == true`;
- trace and ledger hashes are 64-character lowercase hex strings;
- `promotion.decision == "accepted"`.

For `configuration_patch` and `capacity_patch`, `candidate.config_patch` must
also include:

- `activation_policy == "manual_after_cert"`;
- a non-empty bounded `diff` over approved keys such as `adapter.rank`,
  `model.hidden_dim`, `training.max_steps`, or `runtime.max_parameters`;
- `resource_bounds` with hard-capped parameter, memory, and runtime ceilings;
- `rollback.artifact_ref` and `rollback.artifact_hash`;
- passed checks named `bounded_resource_delta`, `rollback_available`, and
  `manual_activation_required`.

When present, `evidence.source_replay_hash` is the domain-separated hash of
the replay artifact used to emit the packet. New emitters include this block.

Any failed gate leaves the candidate as a rejected packet. Rejection is a
successful safety outcome, not a system failure. Rejected packets must still
carry deterministic replay hashes, protected replay counts, and the failed
invariant/gate evidence. This prevents survivorship bias in the improvement
ledger.

## Ledger Validation

The append-only ledger stores one canonical JSON row per packet:

```json
{"packet_hash":"64 hex chars","packet":{ "...": "..." }}
```

`tools/qa_self_improving_neural_qa_ledger_validate.py` checks:

- row canonicalization,
- packet hash integrity,
- packet validity,
- unique `update_id`,
- unique packet hash,
- unique `evidence.source_replay_hash` for packets that include evidence,
- accepted and rejected counts.

## Candidate Selection

The unattended loop may scan broad replay globs, but selection is not purely
largest-fix-first. To keep the system from being monopolized by the legacy HSI
backlog, candidate priority is:

1. bounded configuration/capacity proposals,
2. `general_ml` replay artifacts, especially
   `results/self_improving_neural_qa/general_ml/`,
3. other non-HSI replay artifacts,
4. HSI replay artifacts as fallback.

Within each priority lane, accepted zero-harm candidates are preferred before
rejected safety records, then higher fixed-count candidates.

## General-ML Neural Worker

`tools/qa_general_ml_neural_worker.py` is the bounded neural training step for
general QA benchmarks. It selects enabled tasks from
`results/self_improving_neural_qa/curriculum_registry.json`, trains a small
deterministic numpy MLP for the selected task, honors the activated
runtime-config rank and step caps, writes benchmark summaries under
`experiments/qa_ml/`, and persists only worker bookkeeping state. The worker
does not append to the SINQA ledger and does not promote weights directly.

The initial curriculum contains two enabled tasks:

- `qa_residue_mod3`: synthetic integer QA residue classification over held-out
  moduli.
- `qa_ml_benchmark_selector`: neural selector over existing project QA-ML
  paired-control benchmark summaries.

The replay producer treats neural-worker result files as typed evidence:
`baseline_*` controls may promote to `neural_adapter_*` or
`neural_benchmark_*` controls, but the reverse direction is not emitted as an
improvement artifact.

## Curriculum Discovery

`tools/qa_curriculum_discovery_worker.py` scans
`experiments/qa_ml/results_*.json` for non-HSI paired-control benchmark
summaries and emits disabled-by-default curriculum task proposals under
`results/self_improving_neural_qa/curriculum_proposals/`. It never edits the
active registry. `tools/qa_curriculum_proposal_validate.py` gates these
proposals by source hash, non-HSI status, paired-case count, controls per case,
metric declaration, duplicate task ID checks, and duplicate active-source
checks. A newly discovered task may not reuse the `source_glob` or
`source_hash` of an already-active curriculum task under a different name.

`tools/qa_curriculum_activation_gate.py` turns a validated proposal into a
rollback-bound activation plan. By default it is non-mutating and records the
registry diff, rollback artifact, plan hash, and manual approval requirement.
It mutates `curriculum_registry.json` only when explicitly invoked with
`--apply`.

`tools/qa_curriculum_lifecycle.py` keeps the proposal queue actionable. Active
proposals live in `results/self_improving_neural_qa/curriculum_proposals/` and
must all validate. Rejected or stale proposals move to
`results/self_improving_neural_qa/curriculum_archive/` with a lifecycle record
that hashes the archived proposal, stale activation plan, and rollback artifact.
The scheduled runner validates both the active queue and lifecycle archive as
focused checks.

## Capacity Proposal Worker

`tools/qa_general_ml_capacity_proposal_worker.py` scans accepted SINQA ledger
packets for neural `general_ml` replay evidence and emits at most one bounded
`capacity_patch` proposal for manual activation. It uses rollback artifacts and
hard resource caps, and it never edits the active runtime config itself.

## General-ML Producer

`tools/qa_general_ml_replay_worker.py` is the bounded producer for fresh
non-HSI replay evidence. Each scheduled run scans
`experiments/qa_ml/results_*.json` for paired-control benchmark summaries,
emits at most one unseen `general_ml` replay/rules pair, and records the
activated runtime config it used. The producer never appends to the SINQA
ledger; `tools/qa_self_improving_neural_qa_scheduled_run.py` runs neural
training, then this producer, then the supervisor. The existing replay gate
decides whether each new artifact is accepted or rejected.

## Artifact Pruning

Repetitive generated artifacts may be planned for archive, but archive is not
the same as deletion. `tools/qa_sinqa_artifact_prune_plan.py` emits a
hash-bound plan that groups equivalent neural result, replay, rule, rollback,
and config-proposal artifacts by stable signature and keeps the latest artifact
in each group.

Before any archive operation, `tools/qa_sinqa_artifact_prune_plan_validate.py`
must pass. It recomputes the plan hash, verifies every keep/candidate file hash,
checks counts and duplicate paths, and fails closed when a prune candidate is
still referenced by SINQA provenance such as the ledger, loop transcript,
supervisor state, or packet files. Referenced artifacts require an archive
resolver or explicit reference-preserving migration before they can move.

## Existing Project Anchors

- `qa_lab/agents/self_improvement_agent_v2.py`: level-tagged, Lyapunov-gated
  kernel improvement agent.
- `qa_alphageometry_ptolemy/qa_self_improvement_cert_v1/`: unregistered cert
  stub for v2 cycle traces.
- `tools/qa_hsi_residual_pair_scheduler.py`: mines incremental correction
  candidates.
- `tools/qa_hsi_greedy_rule_merger.py`: greedily accepts only positive
  incremental corrections under replay.
- `tools/qa_hsi_corrected_model_validator.py`: validates deployable corrected
  model artifacts and zero-harm replay.

## First Milestone

Use the HSI corrected-model pipeline as the first neural substrate:

1. Treat the HSI ensemble/classifier as the neural base model.
2. Treat mined correction rules as candidate updates.
3. Emit one v0 update packet per accepted correction batch.
4. Validate the packet with `tools/qa_self_improving_neural_qa_v0.py`.
5. Store accepted packets in an append-only ledger.
6. Store rejected packets in the same ledger with explicit rejection reasons.

That gives the project a defensible claim:

> certified continual improvement layer over a neural model.
