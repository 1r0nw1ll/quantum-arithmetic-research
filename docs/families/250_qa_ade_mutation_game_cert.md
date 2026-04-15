# [250] QA ADE Mutation Game Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-15
**Source:** Will Dale + Claude; (Wildberger, 2020) Mutation Game; (Humphreys, 1972) GTM 9 §9.3 Table 1; cert [244].
**Theory:** `docs/theory/QA_E8_EMBEDDING_AND_ADE.md` §B.

## Claim

The integer Mutation Game BFS from cert [244] (E_8 only) extends to the
full simply-laced ADE classification. For each of `A_5, D_5, E_6, E_7, E_8`,
seeding a single simple root `δ_0 = (1, 0, …, 0)` and closing under all
mutation operators `{s_0, …, s_{n−1}}` reaches exactly `|R(X)|` distinct
integer populations, split equally into positive and negative roots, each
with squared norm `2` under the type's Cartan quadratic form `G = 2I − A`.

| Type | `n` | `det G` | `|R|` | `|R+|` |
|---|---:|---:|---:|---:|
| A_5 | 5 | 6 | 30  | 15 |
| D_5 | 5 | 4 | 40  | 20 |
| E_6 | 6 | 3 | 72  | 36 |
| E_7 | 7 | 2 | 126 | 63 |
| E_8 | 8 | 1 | 240 | 120 |

The Cartan determinants equal the orders of the centers of the simply
connected simple groups of these types — `n+1` for `A_n`, `4` for `D_n`,
`3, 2, 1` for `E_{6,7,8}`.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_ade_mutation_game_cert_v1/qa_ade_mutation_game_cert_validate.py`
- Fixtures: `ade_pass_orbit_sizes.json` (PASS), `ade_fail_wrong_orbit_size.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Checks

| Check | Meaning |
|-------|---------|
| ADE_1 | schema version matches `QA_ADE_MUTATION_GAME_CERT.v1` |
| ADE_CARTAN_DETS | per-type Cartan determinant matches the order-of-center table |
| ADE_BFS_SIZES | per-type BFS orbit size matches Humphreys Table 1 |
| ADE_ROOT_NORM | every generated `v` has `v^T G v = 2` (exhaustive across all 5 types) |
| ADE_SIGN_SPLIT | equal positive/negative split per type, R−=−R+ |
| ADE_SRC | Wildberger 2020 + Humphreys 1972 + cert [244] markers |
| ADE_WITNESS | required witness kinds + per-type edge lists match canonical |
| ADE_F | fail ledger well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `ade_pass_orbit_sizes.json` | PASS | Per-type adjacency edges, orbit sizes, and Cartan determinants for A_5..E_8 |
| `ade_fail_wrong_orbit_size.json` | FAIL | Sets `E_6.orbit_size = 71` (off-by-one) to break the Humphreys match |

## Family Relationships

- Direct extension of [244] `qa_mutation_game_root_lattice_cert` (E_8 only) to all five simply-laced types.
- Complements [240] (sl(3) diamond), [245] (sl(3) hexagonal ring) — those use different integer constructions for sl(3) = A_2.
- Closes the Wildberger integer-Lie-algebra program inside QA at the cert layer (was theoretical in `docs/theory/QA_WILDBERGER_E8_RECONCILIATION.md` §7).
