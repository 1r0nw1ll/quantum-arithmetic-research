# QA Safety — Prompt Injection Refusal Certificate v1

Machine-checkable certificate for test-suite-based prompt injection robustness.

Core claim:

> The model’s refusal/pass rate on a prompt-injection test suite is at least `pass_rate_min_required`.

This certificate is **exact arithmetic**:
- counts are integers
- rates are rationals serialized as `"p/q"` (no floats)

Important: **payloads are not embedded**. Certificates carry only `test_case_id` and verdicts, plus suite identifiers/hashes.

## Files

- `schema.json` — `QA_SAFETY_PROMPT_INJECTION_REFUSAL_CERT.v1`
- `validator.py` — 5-gate validator + CLI + self-test fixtures
- `fixtures/valid_min.json` — passing example
- `fixtures/invalid_rate.json` — failing example (pass rate below threshold)
- `mapping_protocol_ref.json` — intake protocol reference

## Run

```bash
python qa_safety_prompt_injection_refusal_cert_v1/validator.py --self-test
python qa_safety_prompt_injection_refusal_cert_v1/validator.py qa_safety_prompt_injection_refusal_cert_v1/fixtures/valid_min.json
```

