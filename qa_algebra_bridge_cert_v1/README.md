# QA Algebra Bridge Cert v1

Machine-tract family that exports a semantic anchor for QA structural algebra.

## Schema

`QA_ALGEBRA_BRIDGE_CERT.v1`

## Gates

1. Schema shape / types
2. Deterministic generator probe recomputation
3. Word convention / order probes
4. Component-bridge probes (word + scale)
5. Semantics hash binding + invariant diff claim verification

## Semantics Anchor

- `semantics_id = QA_ALGEBRA_SEMANTICS.v1`
- `generator_semantics_ref = QA_GENERATORS_SIGMA_SHEAR.v1`
- `word_application_order = left_to_right`

## Encoding Convention

- Word token `L` means apply `sigma`.
- Word token `R` means apply generator `R := mu sigma mu`.
- Words are applied left-to-right to the seed state.

## Fixtures

- `fixtures/valid_min.json` (PASS)
- `fixtures/invalid_bad_word_probe.json` (FAIL)

`result.invariant_diff_map` is treated as a claim. The validator recomputes failures from gates, checks claim equality (`entries` + rollup), and emits authoritative output from recomputed failures.

## Run

```bash
python qa_algebra_bridge_cert_v1/validator.py --self-test
python qa_algebra_bridge_cert_v1/validator.py qa_algebra_bridge_cert_v1/fixtures/valid_min.json
```
