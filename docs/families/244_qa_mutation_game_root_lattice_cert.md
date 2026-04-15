# [244] QA Mutation Game Root Lattice Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-14
**Source:** Will Dale + Claude; Wildberger, *The Mutation Game, Coxeter–Dynkin Graphs, and Generalized Root Systems*, Algebra Colloquium 27 (2020), pp. 10-11

## Claim

Wildberger's Mutation Game gives an integer-only construction of the full
E_8 root system from simple-root mutations on `Z^8`.

Using the E_8 Dynkin graph, the validator seeds the Weyl orbit at a simple
root, closes under the eight mutation operators, and checks:

`|Orbit| = 240`, with `120` positive roots and `120` negative roots, every
root satisfying `v^T G v = 2` for `G = 2I - A`, and the Weyl involution and
braid relations holding on integer populations.

The cert is Theorem-NT clean because the whole construction stays in integer
tuple space. The only matrix computation is the exact Cartan determinant
check.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_mutation_game_root_lattice_cert_v1/qa_mutation_game_root_lattice_cert_validate.py`
- Fixtures: `mgr_pass_e8_240_roots.json` (PASS), `mgr_fail_wrong_orbit_size.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_MUTATION_GAME_ROOT_LATTICE_CERT.v1` |
| `family` | Family slug `qa_mutation_game_root_lattice_cert` |
| `case` | Fixture case name `e8_240_roots` |
| `cartan` | Exact E_8 Cartan matrix `G = 2I - A` |
| `adjacency_edges` | Zero-indexed E_8 Dynkin edges used by the validator |
| `roots_positive` | Lexicographically sorted 120 positive roots from the BFS orbit |
| `roots_negative_are_negation` | Boolean witness that the negative roots are exactly the negations of `roots_positive` |
| `orbit_size` | Total orbit size `240` |
| `witnesses` | Required witness kinds |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| MGR_1 | schema version matches |
| MGR_CARTAN | E_8 Cartan matrix is exact and has determinant `1` |
| MGR_BFS_240 | BFS closure from all eight simple roots lands on the same `240`-tuple orbit |
| MGR_ROOT_NORM | every root satisfies `v^T G v = 2` |
| MGR_SIGN_SPLIT | the orbit splits into `120` positive and `120` negative roots, with negation pairing |
| MGR_INVOLUTION_BRAID | `s_i^2 = I` and the simple braid relations hold on a sample root |
| MGR_SRC | Wildberger 2020 source markers are present |
| MGR_WITNESS | all five required witness kinds are present and the positive-root witness matches the BFS output |
| MGR_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `mgr_pass_e8_240_roots.json` | PASS | Packages the exact Cartan matrix, 240-root orbit, 120 positive roots, sign split, and Weyl-relations witness set |
| `mgr_fail_wrong_orbit_size.json` | FAIL | Drops the last positive root from the witness list to force a witness mismatch |

## Family Relationships

- First integer-only E_8 root-lattice cert in the Wildberger corpus.
- Complements [240] and [245] by moving from representation dimensions and ring identities to the full Weyl-orbit root enumeration.
- Source theory note: `docs/theory/QA_MUTATION_GAME_ROOT_LATTICE.md`.
