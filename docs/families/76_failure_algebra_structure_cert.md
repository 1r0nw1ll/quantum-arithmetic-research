# Family [76]: QA Failure Algebra Structure Cert

## Purpose

Family [76] certifies a finite algebraic model of QA failure tags as:

- a finite carrier,
- a refinement order (`<=`) as a poset,
- a join-semilattice (`join`) with LUB law,
- a witness composition operator (`compose`) that is associative and monotone,
- a propagation law for this family: `compose(a,b) = join(a,b)`.

## Location

`qa_failure_algebra_structure_cert_v1/`

## Schema

`QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1`

## Gates

1. **Gate 1 — Schema shape / type checks**
2. **Gate 2 — Poset laws**
3. **Gate 3 — Join-semilattice + LUB laws**
4. **Gate 4 — Composition laws (assoc + monotone + propagation + unit)**
5. **Gate 5 — Invariant diff claim binding + rollup verification**

## Artifacts

| File | Purpose |
|------|---------|
| `schema.json` | Schema for `QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1` |
| `validator.py` | Five-gate validator |
| `failure_algebra_anchor.py` | Canonical failure-algebra anchor (types + order + join + compose + rollup) |
| `mapping_protocol_ref.json` | Gate 0 mapping protocol reference |
| `fixtures/PASS_tiny.json` | PASS fixture |
| `fixtures/invalid_compose_associativity_violation.json` | FAIL fixture (single intended Gate 4 associativity obstruction) |

## CLI

```bash
python -m py_compile qa_failure_algebra_structure_cert_v1/validator.py
python qa_failure_algebra_structure_cert_v1/validator.py --self-test
python qa_failure_algebra_structure_cert_v1/validator.py qa_failure_algebra_structure_cert_v1/fixtures/PASS_tiny.json
python qa_failure_algebra_structure_cert_v1/validator.py qa_failure_algebra_structure_cert_v1/fixtures/invalid_compose_associativity_violation.json
```
