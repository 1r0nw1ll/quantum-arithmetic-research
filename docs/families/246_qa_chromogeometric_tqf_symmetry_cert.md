# [246] QA Chromogeometric TQF Symmetry Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-14
**Source:** Will Dale + Claude; Wildberger, *Chromogeometry*, Math. Intelligencer 30:26-38, 2008; Wildberger, *Divine Proportions*, 2005

## Claim

For any three integer-coordinate points `P_i=(x_i,y_i)`, define

- blue quadrance: `Q_b = dx*dx + dy*dy`
- red quadrance: `Q_r = dx*dx - dy*dy`
- green quadrance: `Q_g = 2*dx*dy`

and the Triple Quad Formula residue

`TQF(Q1,Q2,Q3) = (Q1+Q2+Q3)*(Q1+Q2+Q3) - 2*(Q1*Q1+Q2*Q2+Q3*Q3)`.

Then, as polynomial identities over `Z`,

`TQF_r = TQF_g = -TQF_b`.

The blue residue factors as

`TQF_b = 4*area2*area2 = 16*A*A`,

where `area2` is twice the signed triangle area and `A=area2/2`. Hence
`TQF_b >= 0`, and all three chromogeometries agree on collinearity.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_chromogeometric_tqf_symmetry_cert_v1/qa_chromogeometric_tqf_symmetry_cert_validate.py`
- Fixtures: `ctqf_pass_sample_identity.json` (PASS), `ctqf_fail_wrong_sign.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_CHROMOGEOMETRIC_TQF_SYMMETRY_CERT.v1` |
| `symbolic_rb` | SymPy witness that `TQF_r + TQF_b` simplifies to zero |
| `symbolic_gb` | SymPy witness that `TQF_g + TQF_b` simplifies to zero |
| `factored_blue` | Exact factorization `TQF_b=4*area2*area2=16*A*A` |
| `sample_identity` | Deterministic 3000-triangle sample from `[1..9]^2` with all three TQF values |
| `collinearity_invariant` | Exhaustive `C(81,3)` collinearity check across blue/red/green TQF zeros |
| `witnesses` | Required witness kinds |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| CTQF_1 | schema version matches |
| CTQF_SYMBOLIC_RB | `TQF_r + TQF_b` simplifies to zero symbolically |
| CTQF_SYMBOLIC_GB | `TQF_g + TQF_b` simplifies to zero symbolically |
| CTQF_FACTORED_BLUE | `TQF_b` factors as `4*area2*area2=16*A*A` |
| CTQF_SAMPLE_EXHAUSTIVE | 3000 deterministic sample triangles satisfy `TQF_r=TQF_g=-TQF_b` |
| CTQF_COLLINEARITY_INVARIANT | all `C(81,3)` triples have zero TQF iff collinear, for all three forms |
| CTQF_SRC | Wildberger, Chromogeometry, and Divine Proportions source markers are present |
| CTQF_WITNESS | all five required witness kinds are present |
| CTQF_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `ctqf_pass_sample_identity.json` | PASS | Packages symbolic sign identities, blue factorization, 3000 triangle rows, and exhaustive collinearity counts |
| `ctqf_fail_wrong_sign.json` | FAIL | Changes one sampled row to assert the wrong sign `TQF_r=TQF_b` |

## Family Relationships

- Structural follow-up to [241], moving from four-point coplanarity determinants to three-point TQF residues.
- Extends [234] from the segment-level chromogeometric Pythagorean identity to a triangle-level sign symmetry.
- Source theory note: `docs/theory/QA_WILDBERGER_STRUCTURAL_FOLLOWUPS.md` section 2.
