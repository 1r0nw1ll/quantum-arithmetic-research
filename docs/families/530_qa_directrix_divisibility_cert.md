# Family [530]: QA Directrix Divisibility Cert

**Schema:** `QA_DIRECTRIX_DIVISIBILITY_CERT.v1`  
**Root:** `qa_alphageometry_ptolemy/qa_directrix_divisibility_cert_v1/`  
**Validator:** `qa_directrix_divisibility_cert_validate.py --self-test`  
**Status:** Active; self-test passing; added to `FAMILY_SWEEPS`

## Purpose

This cert family retires `directrix_distance_integer` from empirical-open
status. It validates the exact reduction from the conic-labeled directrix target
to a generator divisibility theorem:

`e | d*d*d` iff `e | b*b*b`, where `d=b+e`.

It also validates the sharper prime-exponent classifier
`kernel3(e)=product p^ceil(v_p(e)/3)`, where the condition holds exactly when
`kernel3(e) | b`.

## Schema

| Field | Meaning |
|---|---|
| `schema_version` | Must be `QA_DIRECTRIX_DIVISIBILITY_CERT.v1` |
| `cert_type` | Must be `qa_directrix_divisibility_cert` |
| `theorem_status` | Must be `PROVEN_STRUCTURAL_DIVISIBILITY_REDUCTION` |
| `theorem_statement` | Declares the divisibility reduction and forbids orbit-lift theorem overclaims |
| `proof_obligations` | Required proof gates: modular reduction, cube congruence, kernel3 classifier, bounded audit, orbit context boundary |
| `witnesses` | Positive and negative `(b,e,d,kernel3_e,directrix_integer)` rows |
| `bounded_audit` | Recomputed bounded mismatch audit |
| `orbit_context` | Stage 21 orbit-lift numbers, explicitly marked non-theorem |
| `non_claims` | Explicit exclusions such as conic-geometry explanation and orbit-lift theorem |
| `result` | `PASS` or `FAIL` |
| `fail_ledger` | Declared failure types for failing fixtures |

## Validator Checks

| Gate | Check |
|---|---|
| Schema | Fixed schema/type/status values and required fields |
| Proof obligations | Requires all five proof obligation labels |
| Modular reduction | Checks `e | d*d*d` iff `e | b*b*b` |
| Kernel classifier | Recomputes `kernel3(e)` prime-by-prime and checks `kernel3(e) | b` |
| Witnesses | Recomputes `d`, `kernel3_e`, and directrix truth values |
| Bounded audit | Recomputes support and mismatch counts for the declared window |
| Orbit boundary | Keeps Stage 21 orbit lift as empirical context only |
| Non-claims | Requires `conic_geometry_explanation` exclusion |

## Fixtures

| Fixture | Expected | Purpose |
|---|---|---|
| `directrix_pass_divisibility.json` | PASS | Valid theorem certificate with positive/negative witnesses, Stage 21 orbit context, and `40x40` audit |
| `directrix_fail_bad_kernel.json` | FAIL | Detects an incorrect `kernel3(e)` witness |
| `directrix_fail_orbit_overclaim.json` | FAIL | Rejects treating empirical orbit lift as part of the theorem |

## Family Relationships

- Derived from `experiments/qa_quantum_arithmetic_mining/directrix_divisibility_closure_stage25.py`.
- Source artifact: `results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_stage25_directrix_divisibility_closure.json`.
- Stage 21 empirical context: `qa_orbit_family9` / `qa_orbit_id9` lift `3.928`, beating `e_only` lift `2.428` and `b_only` lift `2.615`.
- Separate from [529], which certifies a rational-conic parametrization theorem.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_directrix_divisibility_cert_v1/qa_directrix_divisibility_cert_validate.py --self-test
```

Expected result: `{"ok":true,...}` with one passing fixture and two correctly rejected fail fixtures.
