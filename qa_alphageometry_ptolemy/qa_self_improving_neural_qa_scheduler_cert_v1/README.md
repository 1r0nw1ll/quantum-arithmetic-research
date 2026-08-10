# QA Self-Improving Neural QA Scheduler Cert v1

This cert validates scheduled-run evidence for the self-improving neural QA
runner. It certifies that a scheduled invocation can run the capped general-ML
neural worker and replay producer, forwards replay/config discovery globs, the
activated runtime-config path, and isolated mutable paths into the supervisor,
runs focused validators, records checkpoint status, and exposes meta-validator
checkpoint results when due.

## Files

| Path | Purpose |
|---|---|
| `qa_self_improving_neural_qa_scheduler_cert_validate.py` | Cert-family validator. |
| `fixtures/pass_scheduler_run.jsonl` | Synthetic PASS scheduled-run record. |
| `fixtures/fail_missing_supervisor_path_forwarding.jsonl` | FAIL record missing isolated supervisor path args. |
| `fixtures/fail_failed_focused_check.jsonl` | FAIL record with a failed focused validator. |

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_self_improving_neural_qa_scheduler_cert_v1/qa_self_improving_neural_qa_scheduler_cert_validate.py --self-test
```
