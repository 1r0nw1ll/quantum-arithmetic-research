# QA Self-Improving Neural QA Activation Cert v1

This cert validates activation-plan evidence for accepted self-improving neural QA
configuration and capacity patches. It does not certify that runtime state was
changed; it certifies that any plan is manual, bounded, rollback-verifiable, and
non-mutating unless post-activation replay and rollback proof are present.

## Files

| Path | Purpose |
|---|---|
| `qa_self_improving_neural_qa_activation_cert_validate.py` | Cert-family validator. |
| `fixtures/pass_activation_plan.json` | PASS plan produced from the accepted general-ML capacity packet. |
| `fixtures/fail_runtime_mutated_without_proof.json` | FAIL plan claiming mutation without post-activation replay/rollback proof. |
| `fixtures/fail_bad_rollback_hash.json` | FAIL plan with rollback hash drift. |

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_self_improving_neural_qa_activation_cert_v1/qa_self_improving_neural_qa_activation_cert_validate.py --self-test
```
