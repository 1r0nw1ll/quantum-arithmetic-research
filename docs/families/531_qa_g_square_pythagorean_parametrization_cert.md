# Family [531]: QA G Square Pythagorean Parametrization Cert

**Schema:** `QA_G_SQUARE_PYTHAGOREAN_PARAMETRIZATION_CERT.v1`  
**Root:** `qa_alphageometry_ptolemy/qa_g_square_pythagorean_parametrization_cert_v1/`  
**Validator:** `qa_g_square_pythagorean_parametrization_cert_validate.py --self-test`  
**Status:** Active; self-test passing; added to `FAMILY_SWEEPS`

## Purpose

This cert family promotes the mined `G_square` target into a complete
Pythagorean-parametrization theorem. For `b,e >= 1`, with `d=b+e` and
`G=d*d+e*e`, the target is exactly the assertion that `(d,e,sqrt(G))` is an
integer right triangle.

The cert validates the classical Euclid parametrization, filtered through the
QA coordinate constraint `d>e` and `b=d-e`.

## Schema

| Field | Meaning |
|---|---|
| `schema_version` | Must be `QA_G_SQUARE_PYTHAGOREAN_PARAMETRIZATION_CERT.v1` |
| `cert_type` | Must be `qa_g_square_pythagorean_parametrization_cert` |
| `theorem_status` | Must be `PROVEN_BY_EUCLID_PYTHAGOREAN_PARAMETRIZATION` |
| `theorem_statement` | Declares the Euclid parametrization and forbids empirical orbit-lift theorem overclaims |
| `proof_obligations` | Required proof gates: QA reduction, Euclid parametrization, branch filter, forward identity, bounded audit |
| `witnesses` | Positive integer `(b,e,d,t,m,n,branch,sqrt_G)` rows |
| `bounded_audit` | Recomputed bounded exhaustiveness sanity check |
| `non_claims` | Explicit exclusions such as factorization shortcut and new triple classification |
| `result` | `PASS` or `FAIL` |
| `fail_ledger` | Declared failure types for failing fixtures |

## Validator Checks

| Gate | Check |
|---|---|
| Schema | Fixed schema/type/status values and required fields |
| Proof obligations | Requires all five proof obligation labels |
| Orbit boundary | Rejects empirical orbit lift claimed as theorem |
| Witnesses | Recomputes `{d,e}` from `t*(m*m-n*n)` and `t*2*m*n` with branch selection |
| Branch filter | Checks `d>e` and `b=d-e` |
| G identity | Recomputes `G=d*d+e*e` and checks it equals the declared square |
| Bounded audit | Recomputes brute solutions and parametrized hits for the declared window |
| Non-claims | Requires `prime_prediction_or_factorization_shortcut` exclusion |

## Fixtures

| Fixture | Expected | Purpose |
|---|---|---|
| `gsq_pass_parametrization.json` | PASS | Valid theorem certificate with both Euclid leg-order branches and a `40x40` zero-miss audit |
| `gsq_fail_bad_witness.json` | FAIL | Detects a witness that does not match the parametrization or square identity |
| `gsq_fail_orbit_overclaim.json` | FAIL | Rejects folding empirical orbit lift into the theorem |

## Family Relationships

- Derived from `experiments/qa_quantum_arithmetic_mining/g_square_proof_closure_stage28.py`.
- Source artifact: `results/qa_quantum_arithmetic_mining_001/qa_quantum_arithmetic_stage28_g_square_proof_closure.json`.
- Companion to [529], because both are complete parametrization certs.
- Separate from [532], which certifies a structural square-part reduction rather than a Pythagorean generator theorem.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_g_square_pythagorean_parametrization_cert_v1/qa_g_square_pythagorean_parametrization_cert_validate.py --self-test
```

Expected result: `{"ok":true,...}` with one passing fixture and two correctly rejected fail fixtures.
