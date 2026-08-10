<!-- PRIMARY-SOURCE-EXEMPT: reason=project-internal cert family doc; this cert registers a local scheduled-run contract rather than a literature claim. -->
# [526] QA Self-Improving Neural QA Scheduler Cert

**Family ID**: 526
**Slug**: `qa_self_improving_neural_qa_scheduler_cert_v1`
**Status**: Active
**Registered**: 2026-07-12

## Purpose

This family certifies the scheduled execution layer for the self-improving
neural QA runner introduced by [524] and [525].

The scheduler is allowed to invoke bounded learning batches, but every run must
first run the optional capped general-ML neural worker, then the capped
general-ML replay producer, forward the replay discovery glob, config proposal
discovery glob, activated runtime-config path, and isolated mutable paths into
the supervisor, run focused validators, and record checkpoint status. This cert
specifically guards against accidental live ledger mutation during tests or
unattended invocation, and against silently narrowing discovery back to one
domain or one proposal type.

## Schema

Each scheduler log row is canonical JSON with:

| Field | Meaning |
|---|---|
| `schema_version` | Must equal `QA_SELF_IMPROVING_NEURAL_QA_SCHEDULER_RUN.v0`. |
| `run` | Positive scheduled-run number. |
| `ok` | Overall scheduled-run verdict. |
| `neural_worker` | Optional subprocess result for capped general-ML neural training. |
| `producer` | Optional subprocess result for the capped general-ML replay producer. |
| `supervisor` | Subprocess result for the capped supervisor batch. |
| `focused_checks` | Ledger, transcript, [524], and [525] validator subprocess results. |
| `checkpoint_due` | Whether a meta-validator checkpoint was required. |
| `meta` | Meta-validator subprocess result when `checkpoint_due=true`; otherwise `null`. |
| `stop_reason` | Bounded-run stop reason. |

## Validator Checks

| Check | Meaning |
|---|---|
| `SINQAS_ROW` | Every scheduler row is canonical JSON. |
| `SINQAS_SCHEMA` | Scheduler schema/version fields are valid. |
| `SINQAS_SUPERVISOR` | Supervisor subprocess exits 0 and reports JSON `ok=true`. |
| `SINQAS_NEURAL_WORKER` | When present, neural-worker subprocess exits 0 and reports JSON `ok=true`. |
| `SINQAS_PRODUCER` | When present, producer subprocess exits 0 and reports JSON `ok=true`. |
| `SINQAS_PATHS` | Supervisor argv includes `--glob`, `--config-glob`, `--runtime-config`, `--ledger`, `--out-dir`, `--transcript`, `--state`, `--heartbeat`, and `--lock`. |
| `SINQAS_FOCUSED` | All four focused checks pass. |
| `SINQAS_META` | Meta checkpoint is present and passing when due, absent otherwise. |
| `SINQAS_STOP` | Stop reason is from the bounded scheduler vocabulary. |

## Fixtures

| Fixture | Expected | Purpose |
|---|---|---|
| `pass_scheduler_run.jsonl` | PASS | Canonical scheduled-run row with forwarded paths, focused checks, and meta checkpoint. |
| `fail_missing_supervisor_path_forwarding.jsonl` | FAIL | Reproduces the path-forwarding class of scheduler bug. |
| `fail_failed_focused_check.jsonl` | FAIL | Focused validator failure must fail the scheduler cert. |

## Family Relationships

- Builds on [524] neural QA promotion ledger.
- Builds on [525] loop transcript hash chain.
- Certifies the operational scheduler boundary above the producer and supervisor tools.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_self_improving_neural_qa_scheduler_cert_v1/qa_self_improving_neural_qa_scheduler_cert_validate.py --self-test
python3 tools/qa_self_improving_neural_qa_scheduled_run.py --self-test
```
