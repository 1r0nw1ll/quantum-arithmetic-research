# QA_EBM_NAVIGATION_CERT.v1

Machine-tract certificate family that hardens “energy-based reasoning” into a QA-native **navigation witness**:

- **Exact energy** scalars only (`int` or exact rational `n/d`), no floats
- **Deterministic policy**: select the minimum-energy **legal** successor, with a total tie-break
- **invariant_diff required** per step: `delta_energy`, `delta_violations`, and a non-empty witness string
- **Failure completeness**: typed failures must include obstruction witness (failure-as-theorem)
- **Verifier coupling (optional)**: if `trace.outcome.accepted_by_verifier=true`, require `verifier_bridge_ref`
  and digest-link it via `digests.refs`

This family is designed to sit between:

- `QA_MAPPING_PROTOCOL.v1` (mapping intake: `M=(S,G,I,F,R,C)`)
- Higher-level certificate families (agentic orchestration, verifier bridges, etc.), including:
  `QA_EBM_VERIFIER_BRIDGE_CERT.v1` (verifier-gated acceptance witness)

## Hash spec

`digests.canonical_sha256` must equal:

`sha256(canonical_json_compact(cert_with_digests.canonical_sha256 = "0"*64))`

where `canonical_json_compact` uses:

- `sort_keys=True`
- `separators=(',', ':')`
- `ensure_ascii=False`

## Run

```bash
python qa_ebm_navigation_cert/validator.py --self-test
python qa_ebm_navigation_cert/validator.py qa_ebm_navigation_cert/fixtures/valid_min.json
```
