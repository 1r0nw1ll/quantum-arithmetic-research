# [278] QA Orbit No-3-Divisor Overclaim Cert

## What this is

Certifies a bounded falsifiable theorem about the divisor shortcut's
**overclaim** failure mode: for every tested modulus `m` with `3 ∤ m` and
`m ≥ 7` (excluding `m = 8` in v1), the canonical period-based
classifier finds zero period-8 satellite states while the algebraic
divisor shortcut produces a 3×3 grid of 9 false satellites.

## Primary source

- Wall, D. D. (1960). *Fibonacci series modulo m*. American Mathematical
  Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

## Pairing with [277]

[277] and [278] together close the divisor shortcut's failure surface:

```text
[277] m = 15k:  shortcut UNDER-COUNTS real period-8 satellites by 32
[278] 3 ∤ m:    shortcut OVER-CLAIMS 9 fake satellites
```

## Artifacts

| Artifact | Path |
|---|---|
| Mapping protocol ref | `qa_orbit_no_3_divisor_overclaim_cert_v1/mapping_protocol_ref.json` |
| Schema | `qa_orbit_no_3_divisor_overclaim_cert_v1/schema.json` |
| Validator | `qa_orbit_no_3_divisor_overclaim_cert_v1/qa_orbit_no_3_divisor_overclaim_cert_validate.py` |
| Fixtures | `qa_orbit_no_3_divisor_overclaim_cert_v1/fixtures/{pass_m7,pass_m10,pass_m11,pass_m20,pass_m25,pass_m35,pass_m50,pass_m100, fail_m8_boundary_exception,fail_wrong_overclaim,fail_claims_5_factor_causal,fail_shortcut_as_canonical}.json` |
| Spec | `qa_orbit_no_3_divisor_overclaim_cert_v1/SPEC.md` |
| Design draft | `docs/specs/QA_ORBIT_5_FACTOR_NO_3_OVERCLAIM_CERT_DRAFT.md` |
| Sister cert | `qa_orbit_pisano_5_factor_boundary_cert_v1/` (cert [277]) |

## How to run

```bash
cd qa_alphageometry_ptolemy

# Standalone validation
python qa_orbit_no_3_divisor_overclaim_cert_v1/qa_orbit_no_3_divisor_overclaim_cert_validate.py

# Or via meta-validator
python qa_meta_validator.py
```

## Semantics

- **NO3_1**: canonical satellite count = expected (always 0 for PASS).
- **NO3_2**: shortcut satellite count = expected (always 9 for PASS).
- **NO3_3**: overclaim = expected (always 9), missed = expected (always 0).
- **NO3_4**: `m = 8` boundary exception is rejected from v1 (FAIL fixture
  `fail_m8_boundary_exception.json` exposes this — validator observes
  actual 15 vs declared 9 and reports the mismatch).
- **SRC**: `mapping_protocol_ref.json` present with required fields.
- **F**: every FAIL fixture declares `expected_fail_type` and the
  declared mode actually fires.

## Causal scope = `no_3_divisor`

The cert is framed around the structural cause (`m // 3` not being a
divisor of `m`), not the discovery slice (`5 | m`). Fixtures declare
`causal_scope = "no_3_divisor"`; a `causal_scope = "5_factor"` declaration
is rejected by the `FIVE_FACTOR_CAUSAL` FAIL fixture.

## Failure modes

| `fail_type` | Meaning | Fix |
|---|---|---|
| `M8_BOUNDARY` | `m = 8` fixture asserts the v1 contract (overclaim = 9) but validator computes 15 | Either drop m=8 from fixtures or add an m=8 sub-claim with expected_overclaims = 15 |
| `WRONG_OVERCLAIM` | Declared overclaim mismatches reality | Update fixture or theorem |
| `FIVE_FACTOR_CAUSAL` | Fixture declares `causal_scope = "5_factor"` | Use `"no_3_divisor"` |
| `SHORTCUT_AS_CANONICAL` | Fixture treats shortcut count as canonical | Use the canonical orbit_family, not the shortcut |
| `MISSING_FIELD` | Required schema field absent | Add the field |

## Out of scope (not certified)

- `m ∈ {1, 2, 3, 4, 5}` — degenerate moduli (`m // 3 ∈ {0, 0, 1, 1, 1}`).
- `m = 8` — boundary exception (`m // 3 = 2`, 4×4 grid, 15 overclaims).
  Candidate for v2 sub-claim.
- `3 | m` — covered by [277] (under-count) or unaffected (no overclaim).

## Example fixtures

**Passing** (`fixtures/pass_m10.json`):

```json
{
  "schema_version": "QA_ORBIT_NO_3_DIVISOR_OVERCLAIM_CERT.v1",
  "fixture_kind": "pass",
  "modulus": 10,
  "expected_canonical_satellites": 0,
  "expected_shortcut_satellites": 9,
  "expected_overclaims": 9,
  "expected_missed": 0,
  "expected_overclaim_grid": [[3,3],[3,6],[3,9],[6,3],[6,6],[6,9],[9,3],[9,6],[9,9]],
  "causal_scope": "no_3_divisor"
}
```

**Failing** (`M8_BOUNDARY` in `fixtures/fail_m8_boundary_exception.json`):

```json
{
  "fixture_kind": "fail",
  "expected_fail_type": "M8_BOUNDARY",
  "modulus": 8,
  "expected_canonical_satellites": 0,
  "expected_shortcut_satellites": 9,
  "expected_overclaims": 9,
  "expected_missed": 0
}
```

## References

- Wall (1960) — DOI: 10.1080/00029890.1960.11989541.
- `qa_orbit_rules.py` (post-`e7b2af0`) — canonical orbit_family +
  orbit_family_divisor_shortcut.
- `docs/specs/QA_ORBIT_5_FACTOR_NO_3_OVERCLAIM_CERT_DRAFT.md` — design
  draft with both Scope A and Scope B options.
- Cert family [277] — sister cert for the under-count regime.

## Changelog

- **v1.0.0** (2026-05-09): Initial release; bounded `3 ∤ m, m ≥ 7, m ≠ 8`
  claim; 8 PASS data points (`m ∈ {7, 10, 11, 20, 25, 35, 50, 100}`)
  spanning 5-multiple and non-5-multiple cases; 4 FAIL fixtures including
  the m=8 boundary exposure; standalone validator PASS.
