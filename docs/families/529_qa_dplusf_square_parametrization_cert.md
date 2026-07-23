# Family [529]: QA D_plus_F Square Parametrization Cert

**Schema:** `QA_DPLUSF_SQUARE_PARAMETRIZATION_CERT.v1`  
**Root:** `qa_alphageometry_ptolemy/qa_dplusf_square_parametrization_cert_v1/`  
**Validator:** `qa_dplusf_square_parametrization_cert_validate.py --self-test`  
**Status:** Active; self-test passing; added to `FAMILY_SWEEPS`

## Purpose

This cert family promotes the mined `D_plus_F_square` / `director_radius_sq_square`
target into a structural theorem. It validates the exact integer classification:
for `b,e >= 1`, with `d=b+e`, `a=b+2e`, `D=d*d`, and `F=a*b`, `D+F` is a square
exactly on the rational-conic parametrized family.

## Schema

| Field | Meaning |
|---|---|
| `schema_version` | Must be `QA_DPLUSF_SQUARE_PARAMETRIZATION_CERT.v1` |
| `cert_type` | Must be `qa_dplusf_square_parametrization_cert` |
| `theorem_status` | Must be `PROVEN_BY_RATIONAL_CONIC_PARAMETRIZATION` |
| `theorem_statement` | Declares the parametrization and forbids orbit-lift theorem overclaims |
| `proof_obligations` | Required proof gates: QA reduction, conic parametrization, positive branch, forward identity, bounded audit |
| `witnesses` | Positive integer `(b,e,t,m,n,sqrt_D_plus_F)` rows |
| `bounded_audit` | Recomputed bounded exhaustion sanity check |
| `non_claims` | Explicit exclusions such as factorization shortcut and physical conic measurement |
| `result` | `PASS` or `FAIL` |
| `fail_ledger` | Declared failure types for failing fixtures |

## Validator Checks

| Gate | Check |
|---|---|
| Schema | Fixed schema/type/status values and required fields |
| Proof obligations | Requires all five proof obligation labels |
| Orbit boundary | Rejects empirical orbit lift claimed as theorem |
| Witnesses | Recomputes `b=t*2*m*n`, `e=t*(m*m - 4*m*n + 2*n*n)>0`, and `sqrt(D+F)` |
| D+F identity | Recomputes `D+F` using `d*d` and checks it equals the declared square |
| Bounded audit | Recomputes brute solutions and parametrized hits for the declared window |
| Non-claims | Requires `prime_prediction_or_factorization_shortcut` exclusion |

## Fixtures

| Fixture | Expected | Purpose |
|---|---|---|
| `dpf_pass_parametrization.json` | PASS | Valid theorem certificate with four positive branch witnesses and a `40x40` zero-miss audit |
| `dpf_fail_bad_witness.json` | FAIL | Detects a witness that does not match the parametrization and square identity |
| `dpf_fail_orbit_overclaim.json` | FAIL | Rejects folding empirical orbit lift into the theorem |

## Family Relationships

- Derived from `experiments/qa_quantum_arithmetic_mining/dplusf_square_proof_closure_stage24.py`.
- Source artifact: `results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_stage24_dplusf_square_proof_closure.json`.
- Companion handoff: `experiments/qa_quantum_arithmetic_mining/cert_handoff_stage26.md`.
- Separate from [530], which certifies a divisibility reduction rather than a conic parametrization.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_dplusf_square_parametrization_cert_v1/qa_dplusf_square_parametrization_cert_validate.py --self-test
```

Expected result: `{"ok":true,...}` with one passing fixture and two correctly rejected fail fixtures.
