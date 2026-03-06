# QA Failure Algebra Structure Cert v1

Machine-tract family for finite failure algebra structure verification.

## Schema

`QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1`

## Anchor Module

`failure_algebra_anchor.py` exports canonical failure algebra constants and helpers:
- `FAILURE_ALGEBRA_ANCHOR_REF`
- `FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256`
- `FAILURE_TYPES`, `LEQ_ROWS`, `JOIN_ROWS`
- `join(f, g)` and `compose(f, g)` (v1 uses `compose = join`)

## Gates

1. Schema shape / types
2. Poset laws for `<=` (reflexive-transitive closure + antisymmetry)
3. Join-semilattice laws + LUB (`join`)
4. Composition laws (`compose`): associativity + monotonicity + propagation + unit
5. Invariant diff claim binding + rollup verification

## Fixtures

- `fixtures/PASS_tiny.json` (PASS)
- `fixtures/invalid_compose_associativity_violation.json` (FAIL, one intended Gate 4 obstruction)

`result.invariant_diff_map` is treated as a claim. The validator recomputes violations from gates, checks claim equality (`entries` + rollup), and emits authoritative output from recomputed failures.

## Run

```bash
python -m py_compile qa_failure_algebra_structure_cert_v1/validator.py
python qa_failure_algebra_structure_cert_v1/validator.py --self-test
python qa_failure_algebra_structure_cert_v1/validator.py qa_failure_algebra_structure_cert_v1/fixtures/PASS_tiny.json
python qa_failure_algebra_structure_cert_v1/validator.py qa_failure_algebra_structure_cert_v1/fixtures/invalid_compose_associativity_violation.json
```
