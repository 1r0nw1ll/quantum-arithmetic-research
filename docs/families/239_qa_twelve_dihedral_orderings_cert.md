# [239] QA Twelve Dihedral Orderings Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-14
**Source:** Will Dale + Claude; Le + Wildberger, *Pentagrammum Mysticum, Twelve Special Conics and the Twisted Icosahedron*, J. Geom. Graphics 24(2):175-191, 2020

## Claim

Five objects `{0,1,2,3,4}` admit exactly `5!/(2*5)=12` distinct dihedral
orderings under the order-10 dihedral group `D_5`.

The validator enumerates all `120` permutations, canonicalizes each ordering by
taking the lexicographic minimum over five rotations and five reflected
rotations, and verifies that the quotient contains exactly 12 canonical classes.
Each class contains exactly 10 permutations.

The fixture records the QA connection `12 = G_2` non-identity root count =
cuboctahedral shell `S_1` = icosahedral vertex count.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_twelve_dihedral_orderings_cert_v1/qa_twelve_dihedral_orderings_cert_validate.py`
- Fixtures: `tdo_pass_dihedral_orderings.json` (PASS), `tdo_fail_wrong_class_count.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_TWELVE_DIHEDRAL_ORDERINGS_CERT.v1` |
| `object_set` | Five labeled objects being ordered |
| `dihedral_group` | Rotation count, reflection count, and group order |
| `permutation_count` | Exhaustive `5!` permutation count |
| `dihedral_class_count` | Number of canonical dihedral classes |
| `canonical_reps` | The 12 canonical representatives |
| `class_sizes` | Size of each dihedral equivalence class |
| `quotient_formula` | `5!/(2*5)=12` witness |
| `qa_connections` | Three recorded QA-adjacent 12-counts |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| TDO_1 | schema version matches |
| TDO_GROUP | `D_5` has five rotations, five reflections, and order 10 |
| TDO_PERMUTATIONS | all `120` permutations are counted |
| TDO_CANONICAL_REPS | canonical reps match the pre-verified list |
| TDO_CLASS_COUNT | exactly 12 dihedral classes are present |
| TDO_CLASS_SIZE | every class has size 10 |
| TDO_FORMULA | quotient formula recomputes to 12 |
| TDO_QA_CONNECTION | G_2, cuboctahedral, and icosahedral 12-counts are recorded |
| TDO_SRC | source attribution includes Le, Wildberger, and Twelve Special Conics |
| TDO_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `tdo_pass_dihedral_orderings.json` | PASS | Packages the exhaustive canonical reps, class sizes, quotient formula, and QA connection |
| `tdo_fail_wrong_class_count.json` | FAIL | Changes the class count from 12 to 11 to prove the count gate is active |

## Family Relationships

- Extends the Wildberger cert batch beyond [235]-[237] into the Pentagrammum Mysticum paper.
- Shares the `12` witness with [217] Fuller VE shell `S_1` and the icosahedral vertex count noted in the Tier-2 follow-up theory file.
- Source theory note: `docs/theory/QA_WILDBERGER_TIER2_FOLLOWUPS.md` section 2.
