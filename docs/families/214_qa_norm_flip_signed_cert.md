# Family [214] QA_NORM_FLIP_SIGNED_CERT.v1

## One-line summary

The Eisenstein quadratic form `f(b, e) = b·b + b·e − e·e` satisfies the integer identity `f(T(b, e)) = −f(b, e)` where `T(b, e) = (e, b+e)`. Corollary: T² preserves f mod m, giving the T-orbit graph of S_m a signed-temporal structure. On S_9 the 5 T-orbits decompose into three signed cosmos pairs (norms {1,8}/{4,5}/{2,7}) and a null satellite + singularity subgraph, reproducing the Pythagorean Families classification (Fibonacci / Lucas / Phibonacci / Tribonacci / Ninbonacci).

## Mathematical content

### Theorem 1 (Norm Flip — integer identity)

For all (b, e) ∈ ℤ²,
```
f(e, b+e) = −f(b, e)
```
where `f(b, e) = b·b + b·e − e·e` is the Eisenstein quadratic form.

**Proof** (direct polynomial expansion):
```
f(e, b+e) = e·e + e·(b+e) − (b+e)·(b+e)
          = e·e + b·e + e·e − b·b − 2·b·e − e·e
          = e·e − b·e − b·b
          = −(b·b + b·e − e·e)
          = −f(b, e)   ∎
```

Each step uses only ring-of-integers identities, so the theorem holds over any commutative ring. Exhaustively verified on {1..9}² — 81/81.

### Theorem 2 (T² preserves the norm)

**Corollary of Theorem 1**: `f(T²(b, e)) mod m = f(b, e) mod m` for all (b, e) ∈ S_m.

**Proof**: `f(T²(s)) = f(T(T(s))) = −f(T(s)) = −(−f(s)) = f(s)`. Mod reduction commutes with the identity. ∎

Exhaustively verified on S_9 — 81/81.

### Theorem 3 (Signed-Orbit Classification on S_9)

The 5 T-orbits on S_9 decompose with exact norm-pair structure:

| Orbit | Representative | Length | Family | Norms (mod 9) | Pair type | Classical name |
|-------|----------------|--------|--------|---------------|-----------|----------------|
| 0 | (1, 1) | 24 | cosmos | {1, 8} | signed | Fibonacci |
| 1 | (1, 3) | 24 | cosmos | {4, 5} | signed | Lucas |
| 2 | (1, 4) | 24 | cosmos | {2, 7} | signed | Phibonacci |
| 3 | (3, 3) | 8  | satellite | {0} | null | Tribonacci |
| 4 | (9, 9) | 1  | singularity | {0} | null | Ninbonacci |

The three cosmos orbits are **bipartite signed**: 12 states carry norm +k and 12 carry norm −k ≡ m−k (mod m), alternating under T. The satellite and singularity form the **null subgraph** where f is identically 0 mod 9 (any (b, e) with both divisible by 3 gives f(b, e) ∈ 9·ℤ).

Names follow the Pythagorean Families paper (verify_classification.py).

### Theorem 4 (Temporal Sign Formula)

On the ℤ² integer lift,
```
sign(f(T^t(s_0))) = (−1)^t · sign(f(s_0))
```
for all t ≥ 0. On mod-m reduced orbits, the norm-mod-m values exhibit a bipartite {+k, −k} = {+k, m−k} coloring with period 2 in the temporal index.

## Checks

| ID | Description |
|----|-------------|
| NFS_1        | schema_version == 'QA_NORM_FLIP_SIGNED_CERT.v1' |
| NFS_FLIP     | integer identity `f(e, b+e) = −f(b, e)` recomputed 81/81 on {1..9}² |
| NFS_T2       | `f(T²(s)) mod 9 = f(s) mod 9` recomputed 81/81 on S_9 |
| NFS_PAIRS    | orbit classification matches the 5-family Pythagorean table (signed {1,8}/{4,5}/{2,7}; null {0},{0}) |
| NFS_TEMPORAL | temporal sign formula `(−1)^t · sign(f(s_0))` declared |
| NFS_155      | cross-reference to family 155 (Bearden phase conjugate) |
| NFS_133      | cross-reference to family 133 (Eisenstein norm) |
| NFS_SRC      | source attribution to Eisenstein present |
| NFS_WIT      | ≥ 5 witnesses (one per orbit family) |
| NFS_F        | fail_ledger well-formed |

## Source grounding

- **Gotthold Eisenstein**, "Untersuchungen über die cubischen Formen mit zwei Variabeln" (*Journal für die reine und angewandte Mathematik* 27, 1844) — binary/ternary quadratic forms with determinant-1 linear action
- **Pythagorean Families paper** (Will Dale + Claude, 2026-03, arxiv-ready) — 5-orbit classification on S_9 with classical names
- Prerequisite: family [133] `qa_eisenstein_cert` — the Eisenstein norm identities `F² − F·W + W² = Z²`
- Related: family [155] `qa_bearden_phase_conjugate_cert` — QCI opposite-sign phenomenon (this cert [214] supplies the algebraic mechanism)
- Related: family [191] `qa_bateson_learning_levels_cert` — cosmos/satellite/singularity IS the signed/null partition
- Verification module: `qa_lab/qa_graph/signed_temporal.py :: verify_all_on_s9()`

## Connection to the graph types initiative (slots 1–5)

This cert is the fifth and final slot. Slots 1–5 are five dual views of the same QA 4-tuple `(b, e, d, a)`:

| Slot | Cert | View | What it exposes |
|------|------|------|------------------|
| 1 | [211] | **Cayley graph** (structural) | Bateson tier reachability = nested Cayley components |
| 2 | [212] | **Fibonacci hypergraph** (dynamical) | Sliding window, uniform vertex degree, orbit-multiset collapse |
| 3 | [210] | **Conversation KG** (operational) | Typed edges, role-diagonal, real-world instance |
| 4 | [213] | **Causal DAG** (structural equations) | Y-structure with A2 as SCM; pair invertibility; Pearl-level collapse |
| 5 | [214] | **Signed-temporal orbit graph** (this cert) | Eisenstein norm flip under T; signed/null stratification |

Each view exposes a different property of the same arithmetic object. Together they cover structural, dynamical, operational, algebraic, and temporal-sign perspectives on `(b, e, d, a)`.

## Fixture files

- `fixtures/nfs_pass_norm_flip.json` — PASS: declares the flip identity, T² preservation, full 5-orbit classification with norm pairs and classical names, temporal sign formula, and cross-refs to [133]/[155]/[191]/[211]/[212]/[213]; validator independently recomputes on S_9
- `fixtures/nfs_fail_wrong_orbit.json` — FAIL: declares Lucas orbit with wrong norm pair {3, 6} and omits [155] cross-ref; validator must flag NFS_PAIRS and NFS_155

## Verification Note (2026-07-07)

Independently re-derived the entire theorem from scratch in a standalone
script (not reusing the validator's own code): confirmed the polynomial
identity `f(e, b+e) = -f(b, e)` by hand expansion, then brute-forced
`flip_identity_s9` (81/81), `t2_preservation_s9` (81/81), and orbit
enumeration on S_9 — exactly reproduced `(24,{4,5}), (24,{2,7}),
(24,{1,8}), (8,{0}), (1,{0})`. The validator
(`qa_norm_flip_signed_cert_validate.py`) already genuinely recomputes
every check (NFS_FLIP/NFS_T2/NFS_PAIRS/etc.) from primitives — no
fixture-trusting gap found. `nfs_fail_wrong_orbit.json` declaring
`"result": "PASS"` internally is intentional, not a bug: that field only
gates `validate()`'s early-return-on-declared-FAIL shortcut, and this
fixture needs its detailed checks to actually run so NFS_PAIRS/NFS_155
can catch the planted defects.

**Found and fixed a real data error**: `nfs_pass_norm_flip.json`'s
`cosmos_lucas` witness declared `"integer_values": {"f_1_3": -5,
"f_3_4": -7, "f_4_7": -37}` — but `f_3_4` and `f_4_7` were never
consistent with the witness's own prose, which computed `f(3,4)=5`
(not -7) and never touched `f(4,7)` at all. Independently recomputed:
`f(3,4) = 9+12-16 = 5` and `f(4,7) = 16+28-49 = -5`, not -37. The
corrected value `f_4_7 = -5` also now correctly demonstrates the
T²-preservation corollary in the fixture data itself (`f(T²(1,3)) =
f(4,7) = -5 = f(1,3)`), which the stale `-37` value did not. Also
cleaned up a messy embedded "wait this should be... let me recompute"
scratch-thought artifact left in the same witness's `note` field
(final stated values were correct, just unedited working-out prose).
