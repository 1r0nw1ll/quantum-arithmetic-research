# QA G_2 Mutation Game — non-simply-laced extension of [244]

**Status:** theory note, draft 2026-04-15.
**Cert:** [251] `QA_G2_MUTATION_GAME_CERT.v1` (proposed).
**Depends on:** [244] `QA_MUTATION_GAME_ROOT_LATTICE_CERT.v1` (E_8 simply-laced) + [250] `QA_ADE_MUTATION_GAME_CERT.v1` (A_5, D_5, E_6, E_7, E_8).

**Primary source:**
- Wildberger, N.J. *The Mutation Game, Coxeter–Dynkin Graphs, and Generalized Root Systems.* Algebra Colloquium **27** (2020), 79 pp. Theorems 0.1 and 0.3. Local PDF: `~/Downloads/MutationGameCoxeterGraphs.pdf` (persistent).
- Humphreys, J.E. *Introduction to Lie Algebras and Representation Theory.* GTM 9 (1972), §9.4 Table 1 (classical root-system data, G_2 row).

**Companion files:** `~/Downloads/MutationGameCoxeterGraphs.pdf`.

---

## 1. Motivation

Cert [244] verified that Wildberger's 2020 Mutation Game produces the 240 E_8 roots exactly via integer mutations on an undirected (equivalently: bidirected with A(z,x) = A(x,z) = 1 per edge) multigraph. Cert [250] extended this to the full simply-laced ADE family (A_5, D_5, E_6, E_7, E_8), confirming Weyl orbit sizes (30, 40, 72, 126, 240) match Humphreys 1972 Table 1 exactly.

The remaining classical Lie algebras are **non-simply-laced**: F_4, G_2, B_n, C_n. Wildberger's 2020 Theorem 0.1 explicitly includes all these — the mutation game extends to them via **directed** (or equivalently: bidirected-with-asymmetric-weights) multigraphs, where the Cartan matrix entry `C[i,j]` becomes the number of edges `j → i`.

This note records the G_2 construction (simplest non-simply-laced, rank 2, 12 roots) as the first non-simply-laced cert in the QA ecosystem.

## 2. G_2 Cartan data

Standard G_2 Cartan matrix (Humphreys 1972 §9.4, Bourbaki 1968 Pl. IX):

```
C = | 2  -1|
    |-3   2|
```

Read as "α_0 = short, α_1 = long":

- `(α_0, α_0) = 2` (short squared length)
- `(α_1, α_1) = 6` (long squared length, 3× short)
- `(α_0, α_1) = -3` (so `C[0,1] = 2·(-3)/6 = -1`, `C[1,0] = 2·(-3)/2 = -3`).

**Mutation-game edge counts:** `A(z→x) = -C[x, z]` for `x ≠ z` (off-diagonal). So:

- `A(1 → 0) = 1` (one edge from long to short)
- `A(0 → 1) = 3` (three edges from short to long)

This is the directed-multigraph encoding of the G_2 Dynkin diagram (double arrow from long to short in standard convention, but here expressed via asymmetric edge counts).

## 3. Mutation rule (from [244] §2, extended to directed)

For each vertex `x`, the mutation `s_x` updates population `p` by

```
new_p[x] = -p[x] + Σ_{z ≠ x} A(z → x) · p[z]
new_p[y] = p[y]    for y ≠ x
```

For G_2 with `x, z ∈ {0, 1}`:

```
s_0(p_0, p_1) = (-p_0 + 1·p_1,   p_1) = (p_1 - p_0,  p_1)
s_1(p_0, p_1) = ( p_0,          -p_1 + 3·p_0) = (p_0, 3·p_0 - p_1)
```

Both are ℤ-linear involutions: `s_0^2 = s_1^2 = I`.

## 4. Root orbit (exhaustive BFS from δ_0 and δ_1) — computationally verified

Starting from `δ_0 = (1, 0)` and `δ_1 = (0, 1)` and applying `{s_0, s_1}` until closure (Python BFS, 2026-04-15), the mutation game produces **12 distinct integer populations** (6 positive + 6 negative, zero mixed-sign):

```
Positive populations R_+(G_2):
  (1, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3)

Negative populations R_−(G_2) = −R_+:
  (−1, 0), (0, −1), (−1, −1), (−1, −2), (−1, −3), (−2, −3)
```

(Populations `(p_0, p_1)` are the mutation-game coordinates, not Humphreys's simple-root expansion coefficients. §5 records the bijection.)

Computational verification (2026-04-15):
- `s_0^2 = s_1^2 = I` on all 12 populations: VERIFIED.
- `(s_0 · s_1)^6 = I` on all 12 populations: VERIFIED.
- `(s_0 · s_1)^k ≠ I` for `k ∈ {1, 2, 3, 4, 5}`: VERIFIED — zero fixed points at any earlier `k`.
- `|R| = 12`, equal sign-split `|R_+| = |R_-| = 6`, `R_- = -R_+`: VERIFIED.

**Coxeter relation** (Wildberger 2020 §"Braid relations", Humphreys 1972 §3.1):

```
(s_0 · s_1)^6 = I
```

i.e., the Weyl group `W(G_2)` is the dihedral group of order 12, in agreement with `|R(G_2)| = 12` roots and `|W(G_2)| = 12`.

## 5. Bijection to Humphreys simple-root expansion

Every positive root in G_2 has the form `a·α_0 + b·α_1` with `(a, b) ∈ {(1,0), (0,1), (1,1), (2,1), (3,1), (3,2)}` (Humphreys 1972 §12.1 Table 1). The bijection from mutation-game population `(p_0, p_1)` to Humphreys simple-root coordinates `(a, b)` is the **coordinate swap**:

```
(a, b)_Humphreys = (p_1, p_0)_mutation-game
```

| Mutation-game `(p_0, p_1)` | Humphreys `(a, b)` | Length |
|---:|---:|---|
| (1, 0) | (0, 1) | long (6) |
| (0, 1) | (1, 0) | short (2) |
| (1, 1) | (1, 1) | short (2) |
| (1, 2) | (2, 1) | short (2) |
| (1, 3) | (3, 1) | long (6) |
| (2, 3) | (3, 2) | long (6) |

**Reason for the swap:** in the mutation-game edge-count convention `A(0 → 1) = 3`, `A(1 → 0) = 1`, the vertex `0` is the **short-root** vertex and `1` is the **long-root** vertex. In Humphreys 1972 §12.1, the convention places long-root coefficients first (`a = a_long`, `b = b_short`). The swap realigns conventions.

Computationally verified (2026-04-15): under the Humphreys Gram `G_sr = [[2, -3], [-3, 6]]`, the 6 Humphreys-basis positive roots split as 3 short + 3 long, for a total (positive + negative) of 6 short + 6 long — matching Humphreys 1972 Table 1 for G_2 exactly.

## 6. Cert [251] claims (to validate)

Given the G_2 Cartan `C = [[2,-1],[-3,2]]` and directed-multigraph adjacency `A(1→0)=1, A(0→1)=3`:

**G2M_1 — exhaustive 12 roots.** BFS from `δ_0` and `δ_1` under `{s_0, s_1}` closes after finitely many steps at exactly 12 distinct integer populations.

**G2M_2 — sign split.** 6 populations are all-nonneg (R_+), 6 are all-nonpos (R_−), and R_− = −R_+.

**G2M_3 — Humphreys-basis root lengths.** Under the G_2 Gram matrix `G_sr = [[2, -3], [-3, 6]]` (symmetric bilinear form on simple-root coordinates per Humphreys 1972 §12.1), after applying the coordinate-swap bijection of §5 the 6 Humphreys-basis positive roots `{(1,0), (0,1), (1,1), (2,1), (3,1), (3,2)}` have exact norms `{2, 6, 2, 2, 6, 6}` — 3 short (norm 2) and 3 long (norm 6). Total across positive + negative: 6 short + 6 long, matching Humphreys 1972 Table 1 for G_2.

**G2M_4 — involution.** `s_0^2 = s_1^2 = I` exhaustively on all 12 roots and selected test populations.

**G2M_5 — Coxeter relation.** `(s_0 · s_1)^6 = I` but `(s_0 · s_1)^k ≠ I` for `k ∈ {1, 2, 3, 4, 5}`, exhaustively on the 12 roots.

**G2M_6 — Humphreys bijection.** The 6 positive roots in mutation-game coordinates map linearly to the 6 positive roots in Humphreys simple-root expansion basis `{(1,0), (0,1), (1,1), (2,1), (3,1), (3,2)}`, with the bijection recorded explicitly.

## 7. Out of scope for [251]

- F_4 (48 roots, rank 4, 2-valued edge-multiplicity pattern) — follow-up cert.
- B_n / C_n (2n² roots, rank n, parametric family) — follow-up cert.
- QA `(b,e,d,a)` → G_2 root-class classifier (analog of [249] for E_8). The G_2 rank is 2, not 8, so direct `(b,e,d,a) → ℤ^2` embedding is the natural analog; deferred.

## 8. Closure of Wildberger integer-Lie-algebra program

With [244]+[250]+[251] (and proposed F_4, B_n, C_n follow-ups), the Wildberger integer-Lie-algebra program covers all finite-type Coxeter diagrams:

| Family | Certs | Status |
|---|---|---|
| Simply-laced (A, D, E) | [244], [250] | Closed |
| Non-simply-laced rank-2 (G_2) | [251] | This note |
| Non-simply-laced rank-4 (F_4) | — | Follow-up |
| Non-simply-laced family (B_n, C_n) | — | Follow-up |

This closes the "next priorities" OB followup noted after [250] shipped (2026-04-15 wild-e8-embed session).
