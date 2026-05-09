# [277] QA Orbit Pisano 5-Factor Boundary Cert

## What this is

Certifies a bounded falsifiable theorem about the divisor shortcut for QA
orbit classification: for `m = 15k` with `k ∈ K_verified = {1, 2, 3, 4,
5, 6, 7, 8, 9, 10, 11, 12, 15, 20}` (14 verified values), the shortcut
under-counts the canonical period-based satellite class by exactly 32
pairs and never over-claims, with the missed pairs partitioning by
`(gcd(b, m), gcd(e, m))` into three signatures.

## Primary source

- Wall, D. D. (1960). *Fibonacci series modulo m*. American Mathematical
  Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

## Artifacts

| Artifact | Path |
|---|---|
| Mapping protocol ref | `qa_orbit_pisano_5_factor_boundary_cert_v1/mapping_protocol_ref.json` |
| Schema | `qa_orbit_pisano_5_factor_boundary_cert_v1/schema.json` |
| Validator | `qa_orbit_pisano_5_factor_boundary_cert_v1/qa_orbit_pisano_5_factor_boundary_cert_validate.py` |
| Fixtures | `qa_orbit_pisano_5_factor_boundary_cert_v1/fixtures/{pass_m15,pass_m30,pass_m45,pass_m60,pass_m75,pass_m150,pass_m300, fail_wrong_undercount,fail_overclaim,fail_wrong_signatures,fail_treats_shortcut_as_canonical}.json` |
| Spec | `qa_orbit_pisano_5_factor_boundary_cert_v1/SPEC.md` |
| Design draft | `docs/specs/QA_ORBIT_PISANO_5_FACTOR_BOUNDARY_CERT_DRAFT.md` |
| Code-level fix | `qa_orbit_rules.py` (commit `e7b2af0`) |
| Surfacing sweep | `experiments/qa_ml/03_gnn_modulus_sweep.py` (cert [276]) |

## How to run

```bash
cd qa_alphageometry_ptolemy

# Standalone validation
python qa_orbit_pisano_5_factor_boundary_cert_v1/qa_orbit_pisano_5_factor_boundary_cert_validate.py

# Or via meta-validator
python qa_meta_validator.py
```

## Semantics

- **PISANO_1**: re-running both classifiers on the fixture's modulus
  yields a missed count equal to the declared `expected_shortcut_undercount`.
- **PISANO_2**: shortcut never over-claims (`expected_overclaims = 0` for
  PASS; FAIL fixtures may declare > 0 to test rejection).
- **PISANO_3**: the missed pairs partition by `(gcd(b, m), gcd(e, m))` into
  the declared `expected_signatures` distribution.
- **SRC**: `mapping_protocol_ref.json` present with required fields.
- **F**: every FAIL fixture declares `expected_fail_type` and the validator
  observes the declared failure mode actually fires.

## Failure modes

| `fail_type` | Meaning | Fix |
|---|---|---|
| `WRONG_UNDERCOUNT` | Declared undercount mismatches reality | Update fixture or theorem |
| `OVERCLAIM_DECLARED` | Declared overclaim > 0 in regime where always 0 | Update fixture |
| `WRONG_SIGNATURES` | Declared gcd-signature distribution mismatches reality | Update fixture |
| `TREATS_SHORTCUT_AS_CANONICAL` | Fixture asserts undercount = 0 on a `5\|m, 3\|m` modulus | Use the canonical orbit_family, not the shortcut |

## Out of scope (not certified)

- `m = 15k` with `k ∉ K_verified` — empirically suggestive but not
  certified. Promotion requires a denser sweep or a structural proof.
- `5 | m ∧ 3 ∤ m` regime (verified `m ∈ {10, 20, 25, 35, 50, 100}`):
  canonical satellites = 0, shortcut over-claims by 9. Different theorem;
  candidate for a separate cert family
  `qa_orbit_5_factor_no_3_overclaim_cert_v1`.

## Example fixtures

**Passing** (`fixtures/pass_m15.json`):

```json
{
  "schema_version": "QA_ORBIT_PISANO_5_FACTOR_BOUNDARY_CERT.v1",
  "fixture_kind": "pass",
  "modulus": 15,
  "k": 1,
  "expected_shortcut_undercount": 32,
  "expected_overclaims": 0,
  "expected_signatures": {"1,3": 8, "1,1": 16, "3,1": 8}
}
```

**Failing** (`WRONG_UNDERCOUNT` in `fixtures/fail_wrong_undercount.json`):

```json
{
  "fixture_kind": "fail",
  "expected_fail_type": "WRONG_UNDERCOUNT",
  "modulus": 15,
  "expected_shortcut_undercount": 16,
  "expected_overclaims": 0
}
```

## References

- Wall (1960) — arxiv-equivalent DOI: 10.1080/00029890.1960.11989541.
- `qa_orbit_rules.py` (post-`e7b2af0`) — canonical `orbit_family` +
  `orbit_family_divisor_shortcut`.
- `experiments/qa_ml/03_gnn_modulus_sweep.py` — generating sweep that
  surfaced the boundary as a [276] follow-up observation.
- `docs/specs/QA_ORBIT_PISANO_5_FACTOR_BOUNDARY_CERT_DRAFT.md` — design
  draft with the full investigation.

## Changelog

- **v1.0.0** (2026-05-09): Initial release; bounded `K_verified` claim;
  14 PASS data points across `k ∈ {1..12, 15, 20}`; standalone validator
  PASS (7 PASS + 4 FAIL fixtures).
