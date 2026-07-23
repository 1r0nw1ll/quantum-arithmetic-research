# Family [532]: QA h_integer Square-Part Reduction Cert

**Schema:** `QA_H_INTEGER_SQUARE_PART_REDUCTION_CERT.v1`  
**Root:** `qa_alphageometry_ptolemy/qa_h_integer_square_part_reduction_cert_v1/`  
**Validator:** `qa_h_integer_square_part_reduction_cert_validate.py --self-test`  
**Status:** Active; self-test passing; added to `FAMILY_SWEEPS`

## Purpose

This cert family retires `h_integer` from empirical-open status. Since
`h=sqrt(F)*d` and `d=b+e` is an integer, the target reduces exactly to asking
whether `F=a*b` is a square, where `a=b+2e`.

The cert validates the square-part reduction: for `g=gcd(a,b)`, write
`a=g*A` and `b=g*B` with `gcd(A,B)=1`. Then `a*b` is a square iff `A` and `B`
are each perfect squares.

## Schema

| Field | Meaning |
|---|---|
| `schema_version` | Must be `QA_H_INTEGER_SQUARE_PART_REDUCTION_CERT.v1` |
| `cert_type` | Must be `qa_h_integer_square_part_reduction_cert` |
| `theorem_status` | Must be `PROVEN_STRUCTURAL_SQUARE_PART_REDUCTION` |
| `theorem_statement` | Declares the square-part reduction and forbids complete-parametrization overclaims |
| `proof_obligations` | Required proof gates: h-to-F reduction, gcd decomposition, coprime product square, bounded audit, strength boundary |
| `witnesses` | Positive and negative `(b,e,d,a,F,gcd_a_b,a_reduced,b_reduced,h_integer)` rows |
| `bounded_audit` | Recomputed bounded mismatch audit |
| `non_claims` | Explicit exclusions such as complete geometry parametrization |
| `result` | `PASS` or `FAIL` |
| `fail_ledger` | Declared failure types for failing fixtures |

## Validator Checks

| Gate | Check |
|---|---|
| Schema | Fixed schema/type/status values and required fields |
| Proof obligations | Requires all five proof obligation labels |
| Strength boundary | Rejects claiming a complete geometry parametrization theorem |
| Witnesses | Recomputes `d`, `a`, `F`, `gcd(a,b)`, and reduced square parts |
| Square-part identity | Checks `h_integer` iff `a/gcd(a,b)` and `b/gcd(a,b)` are both squares |
| Generated witness | Optionally checks derived rows `b=g*r*r`, `a=g*s*s`, `e=(a-b)/2` |
| Bounded audit | Recomputes support and mismatch counts for the declared window |
| Non-claims | Requires `complete_geometry_parametrization` exclusion |

## Fixtures

| Fixture | Expected | Purpose |
|---|---|---|
| `hint_pass_reduction.json` | PASS | Valid theorem certificate with positive and negative witnesses plus a `40x40` zero-mismatch audit |
| `hint_fail_bad_reduced_parts.json` | FAIL | Detects incorrect reduced square-part witness data |
| `hint_fail_parametrization_overclaim.json` | FAIL | Rejects upgrading the structural reduction into a complete geometry parametrization claim |

## Family Relationships

- Derived from `experiments/qa_quantum_arithmetic_mining/h_integer_reduction_closure_stage29.py`.
- Source artifact: `results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_stage29_h_integer_reduction_closure.json`.
- Companion to [530], because both are structural reduction certs.
- Separate from [531], which certifies a complete Pythagorean parametrization theorem.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_h_integer_square_part_reduction_cert_v1/qa_h_integer_square_part_reduction_cert_validate.py --self-test
```

Expected result: `{"ok":true,...}` with one passing fixture and two correctly rejected fail fixtures.
