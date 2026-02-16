# QA_EBM_VERIFIER_BRIDGE_CERT.v1

Machine-tract certificate family for **verifier-gated acceptance**.

This family is the minimal bridge that turns:

- “terminal state reached” (navigation witness)

into:

- “terminal state accepted by verifier” (typed verdict + binding witness).

It is designed to be referenced from `QA_EBM_NAVIGATION_CERT.v1` when a run
claims verifier acceptance.

## Hash spec

`digests.canonical_sha256` must equal:

`sha256(canonical_json_compact(cert_with_digests.canonical_sha256 = "0"*64))`

where `canonical_json_compact` uses:

- `sort_keys=True`
- `separators=(',', ':')`
- `ensure_ascii=False`

## Run

```bash
python qa_ebm_verifier_bridge_cert/validator.py --self-test
python qa_ebm_verifier_bridge_cert/validator.py qa_ebm_verifier_bridge_cert/fixtures/valid_min.json
```

