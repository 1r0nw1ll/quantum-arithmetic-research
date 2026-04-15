# Cert [244] `QA_MUTATION_GAME_ROOT_LATTICE_CERT.v1` — Codex Implementation Spec

**Orchestrator:** Claude (session `wild-e8`, 2026-04-14).
**Executor:** Codex (Claude's .py authorship is revoked per memory/feedback file).
**Theory doc:** `docs/theory/QA_MUTATION_GAME_ROOT_LATTICE.md` (read first — contains the mutation rule, Cartan matrix, Dynkin adjacency, and §6 claim list).
**Primary source:** `~/Downloads/MutationGameCoxeterGraphs.pdf` (Wildberger 2020, Algebra Colloquium 27).
**Registry claim:** `[244]` — verified free 2026-04-14 (latest is [246]; 243 and 244 unused; 245/246 live).

## 1. Deliverables

```
qa_alphageometry_ptolemy/qa_mutation_game_root_lattice_cert_v1/
    mapping_protocol_ref.json
    qa_mutation_game_root_lattice_cert_validate.py
    fixtures/
        mgr_pass_e8_240_roots.json
        mgr_fail_wrong_orbit_size.json
```

Plus:
- `qa_alphageometry_ptolemy/qa_meta_validator.py` — one `FAMILY_SWEEPS` entry, short description block mirroring the existing [245]/[246] format.
- `docs/families/244_qa_mutation_game_root_lattice_cert.md` — one-page family doc linking the theory note + cert spec + fixture summary. Follow the shape of `docs/families/245_qa_sl3_hexagonal_ring_identity_cert.md`.
- `docs/families/README.md` — one table-row entry under the Wildberger corpus section (it already lists [245]/[246]).

## 2. E_8 Dynkin graph (Bourbaki convention, canonical in this repo)

Vertices `1..8` in the Bourbaki diagram. In the Python validator and JSON
fixtures, encode them as zero-indexed vertices `0..7` with adjacency pairs
`{(0,2), (2,3), (3,4), (4,5), (5,6), (6,7), (1,3)}`.

Adjacency matrix `A`: 8×8, symmetric, zero diagonal, entries in `{0,1}` consistent with the above edge list. Cartan / Gram matrix `G = 2·I − A`.

**Sanity:** `det(G) == 1`. If Codex gets a different determinant, the edge list is wrong — do not patch `det` or norms to compensate; re-check the Bourbaki diagram.

## 3. Mutation rule (integer-only)

For population `p ∈ ℤ^8` and vertex index `i ∈ {0..7}`:

```
def s(p, i, A):
    q = list(p)
    q[i] = -p[i] + sum(A[j][i] * p[j] for j in range(len(p)) if j != i)
    return tuple(q)
```

No floats. `q[i]` stays `int`. Represent populations as tuples for hashability in the BFS frontier.

## 4. Validator checks (exhaustive, integer)

All five MGR checks listed in theory doc §6 plus the standard `SRC`/`WITNESS`/`F` closers. Map them to check IDs:

| Check | ID | What it verifies |
|---|---|---|
| Gate 0 pass | `MGR_1` | `mapping_protocol_ref.json` hash matches canonical |
| Cartan matrix | `CARTAN_DEFN` | `G = 2·I − A`, `det(G) == 1`, entries in `{2, 0, −1}` |
| BFS orbit size | `BFS_240` | BFS from `δ_0 = (1,0,...,0)` under 8 mutations reaches exactly 240 tuples; union over the other 7 starts is identical (single Weyl orbit) |
| Root norm | `ROOT_NORM` | every `v` in the orbit satisfies `v · G · v == 2` (pure int) |
| Sign split | `SIGN_SPLIT` | 120 non-negative + 120 non-positive, and `R− == −R+` |
| Involution + braid | `INVOLUTION_BRAID` | `s_i(s_i(p)) == p` exhaustive on 240; `(s_i s_j)^m_ij == I` on a sampled population (`m = 2` non-adjacent, `3` adjacent) |
| Primary-source citation | `SRC` | docstring names Wildberger 2020 + page range |
| Witness match | `WITNESS` | enumerated 240 roots match the `mgr_pass_e8_240_roots.json` fixture (sorted lexicographically for stable comparison) |
| Failure rejection | `F` | `mgr_fail_wrong_orbit_size.json` fixture (orbit of size 239 or 241, or wrong Cartan) fails cleanly |

No check uses floats. No check imports `numpy` for the BFS — pure Python `int`/`tuple`/`set` is enough and keeps the axiom linter happy. (numpy import inside `CARTAN_DEFN` for determinant is fine if you cast to int; prefer sympy `Matrix(...).det()` to stay integer end-to-end.)

## 5. Fixture generation

`mgr_pass_e8_240_roots.json`:

```json
{
  "family": "qa_mutation_game_root_lattice_cert",
  "case": "e8_240_roots",
  "cartan": [[2,0,-1,0,0,0,0,0], ...],  // fill from A above
  "adjacency_edges": [[0,2],[2,3],[3,4],[4,5],[5,6],[6,7],[1,3]],
  "roots_positive": [[1,0,0,0,0,0,0,0], ... 120 integer vectors ...],
  "roots_negative_are_negation": true,
  "orbit_size": 240
}
```

Generate the 120 positive roots by the BFS itself (seed `δ_0`, mutate until closure, filter `all(c >= 0 for c in v) and any(c > 0 for c in v)`). Sanity-check against `sympy.liealgebras.root_system.RootSystem("E8")` or Bourbaki tables (Bourbaki Ch.VI Plate VII).

`mgr_fail_wrong_orbit_size.json`: same schema but with 119 positive roots (drop one arbitrary root). Validator must reject on `BFS_240` OR `WITNESS`.

## 6. QA axiom compliance

Top of validator file, in the module docstring:

```python
"""
QA_COMPLIANCE: {
    'signal_injection': 'none (static algebraic construction)',
    'dynamics': 'integer Weyl mutations on ℤ^8',
    'float_state': false,
    'observer_projection': 'none — classification is QA-discrete-layer',
    'time': 'integer path-length in Weyl group (T1 clean)'
}
"""
```

This follows the existing pattern in `qa_representational_geometry.py` and cert [245]/[246].

## 7. Smoke test

After implementation:

```
python tools/qa_axiom_linter.py --all          # must be CLEAN
cd qa_alphageometry_ptolemy && python qa_meta_validator.py   # [244] GREEN + all prior
```

## 8. Commit handoff

Codex commits using the existing convention:
```
feat(cert): [244] Mutation Game E_8 root lattice — integer Theorem-NT-clean
```
File list: the five files above. Claude's session will then broadcast `file_updated` on the collab bus and shutdown.

## 9. Execution command (bypasses 180s bridge timeout)

```
nohup codex exec --skip-git-repo-check --full-auto < /tmp/codex_244_prompt.txt > /tmp/codex_244.log 2>&1 &
```

Prompt file content should be this spec plus: "Implement files listed in §1 per the check list in §4 using the E_8 Dynkin adjacency in §2. Verify with §7 smoke test. Commit per §8."
