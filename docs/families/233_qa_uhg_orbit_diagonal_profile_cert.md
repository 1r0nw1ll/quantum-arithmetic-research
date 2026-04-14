# [233] QA UHG Orbit Diagonal Profile Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-13
**Source:** Will Dale + Claude

## Claim

At modulus \(m=9\), with
\(T(b,e)=((b+e-1)\bmod 9)+1\) and orbit step
\((b,e)\mapsto(T(b,e),T(e,T(b,e)))\), the 81 points of
\(\{1,\ldots,9\}^2\) partition into:

- 1 singularity orbit of length 1
- 2 satellite orbits of length 4
- 6 cosmos orbits of length 12

Every non-singular orbit containing \(D_1\) has exactly two \(D_1\) points,
and those two points sum to \((9,9)\):

- Sat #1: \((3,3)+(6,6)=(9,9)\)
- Cos #1: \((1,1)+(8,8)=(9,9)\)
- Cos #3: \((2,2)+(7,7)=(9,9)\)
- Cos #4: \((5,5)+(4,4)=(9,9)\)

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_uhg_orbit_diagonal_profile_cert_v1/qa_uhg_orbit_diagonal_profile_cert_validate.py`
- Fixtures: `uodp_pass_m9_full_partition.json` (PASS), `uodp_fail_wrong_complement_pair.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Checks

| Check | Meaning |
|-------|---------|
| UODP_1 | schema version matches |
| UODP_M | modulus is 9 |
| UODP_PARTITION | generated orbit lengths match 1 + 2 + 6 partition |
| UODP_ORBIT_DATA | fixture orbit point lists and diagonal-class multisets recompute |
| UODP_D1_PROFILE | \(D_1\) multiplicities recompute |
| UODP_COMPLEMENT | non-singular \(D_1\) pairs sum to \((9,9)\) |
| UODP_SRC | source attribution present |
| UODP_F | fail ledger well-formed |

## Cross-references

- Spec: `docs/specs/WILDBERGER_CERT_BATCH_SPEC.md`
- Theory: `docs/theory/QA_UHG_PROJECTIVE_COORDINATES.md`
- Template family: `docs/families/217_qa_fuller_ve_diagonal_decomposition_cert.md`

## Open questions (not in this cert)

- This cert records the \(m=9\) orbit profile only; higher-modulus orbit profiles remain outside scope.
