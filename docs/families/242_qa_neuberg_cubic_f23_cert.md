# [242] QA Neuberg Cubic F23 Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-14
**Source:** Will Dale + Claude; Wildberger, *Neuberg cubics over finite fields*, arXiv:0806.2495, 2008

## Claim

Wildberger's Neuberg-cubic construction extends universal geometry to finite
fields. This cert packages the QA-native finite-field slice over `F_23`, using
integer arithmetic modulo 23 throughout.

For the Weierstrass curve `E: y^2=x^3+x+1`, the validator exhaustively counts
`27` affine points over `F_23`, hence `28` projective points after adding the
point at infinity. It also checks selected Weierstrass tangent-conic witnesses
by enumerating the conic point sets over `F_23` and requiring the relation to be
either identical or disjoint, never partial.

The fixture includes an integer-polynomial spread witness for a triangle in
`F_23` and a QA compatibility marker: `p=23`, characteristic not `2` or `3`,
with all checked arithmetic staying finite-field integer arithmetic.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_neuberg_cubic_f23_cert_v1/qa_neuberg_cubic_f23_cert_validate.py`
- Fixtures: `ncf23_pass_neuberg_f23.json` (PASS), `ncf23_fail_wrong_point_count.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_NEUBERG_CUBIC_F23_CERT.v1` |
| `elliptic_curve` | `F_23` Weierstrass curve parameters, point counts, and affine point list |
| `tangent_conic_witnesses` | Two finite-field conic set relations, one identical and one disjoint |
| `orthic_spread_polynomial_witness` | Integer-polynomial spread numerator and denominator pair |
| `qa_compatibility` | Characteristic and integer-arithmetic compatibility marker |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| NCF23_1 | schema version matches |
| NCF23_POINT_COUNT | `y^2=x^3+x+1` recomputes to 27 affine points and 28 projective points |
| NCF23_TANGENT_CONIC_DICHOTOMY | conic witnesses recompute as identical or disjoint, not partial |
| NCF23_SPREAD_POLYNOMIAL | spread pair recomputes from integer coordinate polynomials over `F_23` |
| NCF23_QA_COMPAT | `p=23`, characteristic not `2` or `3`, integer arithmetic only |
| NCF23_SRC | source attribution includes Wildberger, Neuberg cubics, and arXiv:0806.2495 |
| NCF23_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `ncf23_pass_neuberg_f23.json` | PASS | Packages point enumeration, conic dichotomy witnesses, spread polynomial data, and QA compatibility |
| `ncf23_fail_wrong_point_count.json` | FAIL | Changes the projective point count from 28 to 27 to prove the point-count gate is active |

## Family Relationships

- Adds the finite-field elliptic-curve Wildberger bridge staged in the Tier-3 follow-up theory note.
- Complements [234] and [236] by moving rational-trig style integer formulas into `F_23`.
- Source theory note: `docs/theory/QA_WILDBERGER_TIER3_FOLLOWUPS.md` section 1.
