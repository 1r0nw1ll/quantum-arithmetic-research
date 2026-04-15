# [249] QA E8 Embedding Orbit Classifier Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-15
**Source:** Will Dale + Claude; (Wildberger, 2020) Mutation Game; cert [244].
**Theory:** `docs/theory/QA_E8_EMBEDDING_AND_ADE.md` §A.

## Claim

The QA tuple `(b, e, d, a)` at `m = 9` lifts to `ℤ^8` via the canonical
**diagonal embedding** `E_diag(b,e) = (b, e, d, a, 0, 0, 0, 0)`. Under the
E_8 Cartan quadratic form `Q(v) = v^T · G · v` (with `G` from cert [244]),
the per-T-orbit minimum of `Q` is a complete classifier of the five m=9
T-orbits.

The five values `min Q | O = (8, 16, 28, 72, 162)` correspond to the three
size-24 Cosmos orbits, the size-8 Satellite, and the size-1 Singularity
respectively, and are pairwise distinct.

The closed form `Q(E_diag(b,e)) = 2(b² + e² + d² + a²) − 2(b·d + e·a + d·a)`
is verified symbolically (sympy) and exhaustively on `[1..9]²`.

`E_tri(b,e) = (b, e, b, e, d, d, a, a)` is also recorded: it likewise yields
5/5 distinct Q-multisets but `min Q` collides at the Singularity (162) with
`E_diag`. `E_diag` ships as canonical for the simpler closed form.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_e8_embedding_orbit_classifier_cert_v1/qa_e8_embedding_orbit_classifier_cert_validate.py`
- Fixtures: `e8e_pass_e_diag.json` (PASS), `e8e_fail_wrong_min_q.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Checks

| Check | Meaning |
|-------|---------|
| E8E_1 | schema version matches `QA_E8_EMBEDDING_ORBIT_CLASSIFIER_CERT.v1` |
| E8E_CARTAN_LOAD | E_8 Cartan from [244] matches; det = 1 |
| E8E_T_ORBITS | m=9 T-orbit partition is `{1, 8, 24, 24, 24}`, sum 81 |
| E8E_DIAG_FORMULA | symbolic identity + exhaustive numeric on `[1..9]²` |
| E8E_DIAG_MIN_Q | per-orbit `min Q` = `(8, 16, 28, 72, 162)`, 5 distinct |
| E8E_DIAG_MULTISET | per-orbit Q-multisets pairwise distinct under `E_diag` |
| E8E_TRI_PROFILE | per-orbit Q-multisets pairwise distinct under `E_tri` (informational) |
| E8E_SRC | Wildberger 2020 + cert [244] markers |
| E8E_WITNESS | required witness kinds + diag_rows match BFS output + canonical_embedding=`E_diag` |
| E8E_F | fail ledger well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `e8e_pass_e_diag.json` | PASS | Packages T-orbit partition, per-orbit Q profile under E_diag and E_tri, witness set |
| `e8e_fail_wrong_min_q.json` | FAIL | Tampers `diag_min_Q_sorted` to break the 5-distinct invariant |

## Family Relationships

- Builds directly on [244] `qa_mutation_game_root_lattice_cert` (uses its E_8 Cartan + adjacency).
- Relates to [233] `qa_uhg_orbit_diagonal_profile_cert` (m=9 orbit structure, different group action).
- First QA → ℤ^8 embedding cert; sets up downstream Weyl-chamber / W-orbit classification work.
