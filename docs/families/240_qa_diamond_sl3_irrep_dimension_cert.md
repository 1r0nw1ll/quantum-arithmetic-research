# [240] QA Diamond sl3 Irrep Dimension Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-14
**Source:** Will Dale + Claude; Wildberger, *Quarks, diamonds, and representations of sl_3*, UNSW preprint, Oct 2003

## Claim

Under the QA identification `(qa_b, qa_e) = (sl3_a, sl3_b)` with
`qa_d = qa_b + qa_e`, the sl(3) irreducible representation dimension formula

`dim pi[a,b] = (a+1)(b+1)(a+b+2)/2`

becomes the QA integer polynomial

`dim = (qa_b+1)*(qa_e+1)*(qa_d+2)//2`.

The validator checks the formula against 22 standard Fulton-Harris / LiE table
entries, verifies the adjoint representation `pi[1,1]` has dimension `8`, and
packages the quark / anti-quark fundamental triples and integer Heisenberg
commutator witness.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_diamond_sl3_irrep_dimension_cert_v1/qa_diamond_sl3_irrep_dimension_cert_validate.py`
- Fixtures: `dsi_pass_sl3_dimensions.json` (PASS), `dsi_fail_wrong_adjoint.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_DIAMOND_SL3_IRREP_DIMENSION_CERT.v1` |
| `dim_formula` | QA coordinate identification, dimension formula, 22 table rows, and polynomial-equivalence marker |
| `adjoint` | `pi[1,1]` dimension witness and `dim sl(3)=8` equality |
| `triangular_column` | `D(a,0)` dimensions for `a=0..5` as triangular numbers |
| `quark_antiquark` | Fundamental `D(1,0)` and `D(0,1)` dimension and height witnesses |
| `heisenberg_commutators` | Integer structure constants for the three listed root-vector brackets |
| `witnesses` | Required witness kinds |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| DSI_1 | schema version matches |
| DSI_DIM_FORMULA | 22 standard entries match `(qa_b+1)*(qa_e+1)*(qa_d+2)//2`; equivalence is checked on grid `0..6` |
| DSI_ADJOINT | `pi[1,1]` recomputes to `8=dim sl(3)` |
| DSI_TRIANGULAR_COLUMN | `D(a,0)` dimensions are `1,3,6,10,15,21` for `a=0..5` |
| DSI_QUARK_ANTIQUARK | `D(1,0)` and `D(0,1)` both have dimension `3` with the listed heights |
| DSI_HEISENBERG | listed brackets have integer coefficients `1,0,0` |
| DSI_SRC | Wildberger, Quarks/diamonds, and `sl_3` source markers are present |
| DSI_WITNESS | all five required witness kinds are present |
| DSI_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `dsi_pass_sl3_dimensions.json` | PASS | Packages dimension table, adjoint, triangular column, quark triples, and Heisenberg witnesses |
| `dsi_fail_wrong_adjoint.json` | FAIL | Changes the adjoint dimension from `8` to `7` to prove the adjoint gate is active |

## Family Relationships

- Extends the Wildberger Tier-2 batch with the integer diamond model for `sl_3=A_2`.
- Complements [231]-[239] by adding the representation-theoretic integer-lattice branch of the Wildberger corpus.
- Source theory note: `docs/theory/QA_WILDBERGER_TIER2_FOLLOWUPS.md` section 3.
