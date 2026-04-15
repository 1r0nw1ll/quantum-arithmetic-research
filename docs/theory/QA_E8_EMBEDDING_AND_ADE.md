# QA → E_8 Embedding + Full ADE Mutation Game Extension

**Status:** theory note, draft 2026-04-15.
**Certs:** [249] `QA_E8_EMBEDDING_ORBIT_CLASSIFIER_CERT.v1`, [250] `QA_ADE_MUTATION_GAME_CERT.v1`.
**Builds on:** [244] `QA_MUTATION_GAME_ROOT_LATTICE_CERT.v1` (E_8 mutation primitive). See `docs/theory/QA_MUTATION_GAME_ROOT_LATTICE.md`.

**Primary sources:**
- Wildberger, N.J. *The Mutation Game, Coxeter–Dynkin Graphs, and Generalized Root Systems.* Algebra Colloquium **27** (2020). Local PDF: `~/Downloads/MutationGameCoxeterGraphs.pdf`.
- Bourbaki, N. *Groupes et algèbres de Lie, Ch. IV–VI.* Hermann (1968). Standard Dynkin + Cartan matrices for all ADE types.
- Humphreys, J.E. *Introduction to Lie Algebras and Representation Theory.* Springer GTM 9 (1972). Root-system counts by type (Table 1, §9.3).

---

## Part A — Cert [248]: QA → ℤ^8 embedding, orbit classifier

### A.1 Problem

[244] establishes the E_8 mutation primitive: 240 roots = W(E_8)·δ_0, all integer. It does **not** fix a `(b,e,d,a) → ℤ^8` embedding. This note proposes two candidates and specifies the empirical test.

### A.2 Candidate embeddings

For QA tuple `(b, e)` at `m = 9` with `d = ((b+e-1) mod 9) + 1`, `a = ((b+2e-1) mod 9) + 1`:

- **E_diag** (4+4 diagonal): `p_diag = (b, e, d, a, 0, 0, 0, 0)`.
- **E_tri** (triune replication): `p_tri = (b, e, b, e, d, d, a, a)` — replicates across Dynkin branch structure to engage all 8 vertices.

### A.3 Invariants under W(E_8)

`W(E_8)` preserves the integer quadratic form `Q(v) = v^T · G_{E_8} · v`. So for each QA T-orbit `O ⊂ {1..9}²`, compute `{Q(p(b,e)) : (b,e) ∈ O}` under each embedding. The cert ships **iff at least one embedding produces a T-orbit-invariant** — i.e. `Q` is constant within every T-orbit.

### A.4 m = 9 T-orbits (verified empirically 2026-04-15 via direct enumeration)

T-step `(b,e) → (e, ((b+e−1) mod 9)+1)` partitions `[1..9]²` into **5 orbits**:

- 1 Singularity (size 1): `{(9,9)}`.
- 1 Satellite (size 8): orbit through `(3,3)` (closes after 8 steps).
- 3 Cosmos (size 24 each): orbits through `(1,1)`, `(1,3)`, `(1,4)`.

Total `1 + 8 + 24 + 24 + 24 = 81 = 9²`. (Earlier note about 1+2+6 partition referenced a different group action — T-step is the canonical QA dynamics for this cert.)

### A.5 Empirical findings (2026-04-15, exhaustive on m=9)

**Pointwise `Q` is NOT T-invariant** under either `E_diag` or `E_tri` (only the size-1 Singularity is trivially invariant).

**Q-multiset per T-orbit IS a complete classifier** under both embeddings: 5/5 orbits have distinct Q-multisets.

**Stronger: `min Q` per T-orbit is already a complete classifier under `E_diag`:**

| Orbit | size | first | `min Q` | `max Q` | `#distinct Q` |
|---|---:|---|---:|---:|---:|
| Cosmos₁ | 24 | (1,1) | 8 | 148 | 15 |
| Cosmos₂ | 24 | (1,3) | 16 | 142 | 14 |
| Cosmos₃ | 24 | (1,4) | 28 | 154 | 13 |
| Satellite | 8 | (3,3) | 72 | 144 | 4 |
| Singularity | 1 | (9,9) | 162 | 162 | 1 |

The five `min Q` values `(8, 16, 28, 72, 162)` are **all distinct**. So the integer functional `m_E8(O) := min{ v^T G v : v = E_diag(b,e), (b,e) ∈ O }` is a complete T-orbit invariant.

`E_diag` is also analytically clean: `Q(E_diag(b,e)) = 2(b² + e² + d² + a²) − 2(bd + ea + da)` with `d, a` per A1.

### A.6 Cert [249] claims

1. **E8E_CARTAN_LOAD**: `G_{E_8}` matches [244] exactly.
2. **E8E_T_ORBITS**: 5 T-orbits at m=9, sizes `{1, 8, 24, 24, 24}`.
3. **E8E_DIAG_FORMULA**: `Q(E_diag(b,e)) = 2(b²+e²+d²+a²) − 2(bd+ea+da)` — symbolic identity (sympy) and exhaustive numeric on `[1..9]²`.
4. **E8E_DIAG_MIN_Q**: `min{Q(E_diag(b,e)) : (b,e) ∈ O}` per orbit equals `(8, 16, 28, 72, 162)` — five distinct integers, complete T-orbit classifier.
5. **E8E_DIAG_MULTISET**: per-orbit Q-multisets are pairwise distinct (5/5).
6. **E8E_TRI_PROFILE**: same stats for `E_tri` recorded in fixture (informational; `E_diag` is canonical).
7. **SRC**, **WITNESS**, **F**.

`E_diag` ships as the canonical QA → ℤ⁸ embedding for cert [249].

---

## Part B — Cert [249]: Full ADE mutation game

### B.1 Generalization

The BFS from [244] trivially parameterizes to any simply-laced Dynkin graph. The claim is the same structure holds: BFS from `δ_0` generates exactly `|R(X)|` populations, split as `|R(X)|/2` positive + `|R(X)|/2` negative, each with `v^T·G·v = 2`.

### B.2 Expected orbit sizes + Cartan determinants (verified 2026-04-15)

| Type | `|R(X)|` | `det G` | n | Edge list (0-indexed) |
|---|---:|---:|---:|---|
| A_5 | 30  | 6 | 5 | `[(0,1),(1,2),(2,3),(3,4)]` |
| D_5 | 40  | 4 | 5 | `[(0,1),(1,2),(2,3),(2,4)]` (branch at vertex 2) |
| E_6 | 72  | 3 | 6 | `[(0,2),(2,3),(3,4),(4,5),(1,3)]` |
| E_7 | 126 | 2 | 7 | `[(0,2),(2,3),(3,4),(4,5),(5,6),(1,3)]` |
| E_8 | 240 | 1 | 8 | `[(0,2),(2,3),(3,4),(4,5),(5,6),(6,7),(1,3)]` |

`det G_T` equals the order of the center of the simply-connected group of type T (`n+1` for `A_n`, `4` for `D_n`, `3, 2, 1` for `E_{6,7,8}`). All five BFSs confirmed via direct enumeration: orbit size matches expected, `v^T·G·v = 2` for every `v`, equal positive/negative split.

### B.3 Cert [250] claims

For each of `A_5, D_5, E_6, E_7, E_8`:
1. **ADE_CARTAN_T**: `G_T = 2·I − A_T`, det matches table B.2.
2. **ADE_BFS_T**: orbit size matches table B.2.
3. **ADE_ROOT_NORM_T**: every generated `v` has `v^T·G_T·v = 2`.
4. **ADE_SIGN_SPLIT_T**: equal positive/negative split; `R− = −R+`.

Plus SRC / WITNESS / F. Covers the full integer-Lie-algebra program at once.

### B.4 Why bother (given [244] is E_8)

- **Sanity check**: [244]'s BFS is correct only if the same code produces the right counts for simpler cases.
- **Completeness**: finishes Wildberger's integer program within QA at the cert layer (was theoretical in `docs/theory/QA_WILDBERGER_E8_RECONCILIATION.md` §7).
- **Enables type-parametric classifiers**: future certs can pick a Dynkin type by QA-orbit structural features (Satellite size 8 ↔ D_4? etc.) — research territory unlocked once the primitives are live.

---

## References

- Wildberger, N.J. *The Mutation Game, Coxeter–Dynkin Graphs, and Generalized Root Systems.* Algebra Colloquium **27** (2020).
- Humphreys, J.E. *Introduction to Lie Algebras and Representation Theory.* GTM 9 (1972), §9.3 Table 1.
- Bourbaki, N. *Groupes et algèbres de Lie, Ch. IV–VI.* Hermann (1968).
- QA cert [244] `QA_MUTATION_GAME_ROOT_LATTICE`, this repo.
- QA cert [233] `QA_UHG_ORBIT_DIAGONAL_PROFILE`, this repo — source of m=9 T-orbit enumeration.

Will Dale + Claude, 2026-04-15.
