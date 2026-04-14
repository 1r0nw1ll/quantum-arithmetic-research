# [241] QA Quadruple Coplanarity Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-14
**Source:** Will Dale + Claude; Notowidigdo + Wildberger, *Generalised vector products and metrical trigonometry of a tetrahedron*, arXiv:1909.08814; Wildberger, *Chromogeometry*, 2008

## Claim

Every QA point `(b,e,d)` with `d=b+e` lies in the 2-plane
`d-b-e=0` in `R3`. Hence every four QA points are coplanar.

Equivalently, the 4-point Cayley-Menger determinant vanishes for the same
quadruples under all three chromogeometric quadrances:

- blue: `Q_b = dx*dx + dy*dy`
- red: `Q_r = dx*dx - dy*dy`
- green: `Q_g = 2*dx*dy`

The validator uses SymPy for exact integer determinants and never uses floats.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_quadruple_coplanarity_cert_v1/qa_quadruple_coplanarity_cert_validate.py`
- Fixtures: `qco_pass_coplanarity.json` (PASS), `qco_fail_wrong_plane.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_QUADRUPLE_COPLANARITY_CERT.v1` |
| `plane_identity` | Exhaustive range for `d-b-e=0` plus optional explicit points |
| `parallelepiped_volume` | Thirty fixed sampled triples from `{1..9}^2`, each with determinant zero |
| `cayley_menger` | Thirty fixed sampled quadruples plus Satellite #1, with blue/red/green CM determinants |
| `chromo_coplanarity_preserved` | Marker that all three chromogeometric forms preserve the zero determinant witness |
| `witnesses` | Required witness kinds |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| QCO_1 | schema version matches |
| QCO_PLANE_IDENTITY | `d-b-e=0` recomputes on `[-9..9]^2` and for explicit fixture points |
| QCO_PARALLELEPIPED_VOL | 30 sampled QA triples have zero `3x3` determinant |
| QCO_CM_4POINT_BLUE_RED_GREEN | 30 sampled quadruples plus Satellite #1 have zero Cayley-Menger determinant under blue, red, and green quadrances |
| QCO_CHROMO_COPLANARITY_PRESERVED | all three forms are recorded as preserving coplanarity |
| QCO_SRC | Notowidigdo, Wildberger, and Chromogeometry source markers are present |
| QCO_WITNESS | all six required witness kinds are present |
| QCO_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `qco_pass_coplanarity.json` | PASS | Packages the plane identity, 30 triple witnesses, 30 quadruple witnesses, Satellite #1, and all three chromogeometric CM zeros |
| `qco_fail_wrong_plane.json` | FAIL | Adds synthetic non-QA point `(1,2,5)` with `d-b-e=2` to prove the plane gate is active |

## Family Relationships

- Extends [237] from the QA 2-plane in `R4` to the projected QA 2-plane in `R3`.
- Complements [234] by using the same blue/red/green chromogeometric forms in a coplanarity determinant witness.
- Source theory note: `docs/theory/QA_WILDBERGER_TIER2_FOLLOWUPS.md` section 4.
