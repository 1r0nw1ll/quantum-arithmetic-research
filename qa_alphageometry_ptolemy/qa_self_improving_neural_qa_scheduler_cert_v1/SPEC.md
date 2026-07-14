# QA Self-Improving Neural QA Scheduler Cert v1

## Purpose

Validate scheduled-run records for the self-improving neural QA scheduler,
including the optional capped general-ML neural worker and replay producer
phases.

## Checks

| Check | Meaning |
|---|---|
| SINQAS_ROW | Every scheduler log row is canonical JSON. |
| SINQAS_SCHEMA | `schema_version` is `QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_RUN.v0`. |
| SINQAS_SUPERVISOR | Supervisor subprocess exits successfully and returns JSON with `ok=true`. |
| SINQAS_NEURAL_WORKER | Optional neural-worker subprocess exits successfully and returns JSON with `ok=true`. |
| SINQAS_PRODUCER | Optional producer subprocess exits successfully and returns JSON with `ok=true`. |
| SINQAS_PATHS | Supervisor argv forwards `--glob`, `--config-glob`, `--runtime-config`, `--ledger`, `--out-dir`, `--transcript`, `--state`, `--heartbeat`, and `--lock`. |
| SINQAS_FOCUSED | Ledger, transcript, active curriculum proposal queue, curriculum archive, [524], [525], prune-plan, and anti-forgetting focused checks all pass. Historical rows before scheduler run 13 may contain only the original ledger/transcript/[524]/[525] focused checks; runs 13-35 may contain the six-check lifecycle-aware set or seven-check prune-aware set; run 36 and later must contain the eight-check anti-forgetting-aware set. |
| SINQAS_META | If checkpoint is due, meta result exists and passes; otherwise it is absent. |
| SINQAS_STOP | Stop reason is one of the known bounded-run stop reasons. |
