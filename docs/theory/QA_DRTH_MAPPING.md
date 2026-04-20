<!-- PRIMARY-SOURCE-EXEMPT: reason=primary source is Chase's locally-received DRTH DIAZAI.docx (2026-04-19); this doc is the structural mapping worksheet from Chase's DRTH algebra to canonical QA, not a derivation from published literature -->

# QA ↔ DRTH — Structural Mapping from Chase's DR Triality Hypermatrix

**Primary source**: `~/Downloads/DRTH DIAZAI.docx` (Chase F. Diaz, received 2026-04-19). 11 sections, proven theorems, 26,244-order automorphism group computation.

**Companion**: `docs/theory/QA_NUCLEUS_MAPPING.md` (the earlier mapping of Chase's `Nucleus.html`). DRTH is the formal-paper successor to that artifact.

**Methodology**: Same "Map Best-Performing to QA" protocol. DRTH is a rigorous algebraic construction (totient split, automorphism group, triality) — substantially stronger than `Nucleus.html`. We catalogue the structural overlaps with canonical QA (`qa_orbit_rules.py`, `qa_core.js`), the divergences, and flag the speculative parts.

---

## Top-line relationship

**DRTH and QA share primitive foundations but build different objects.** Both live on `D₉ = {1..9}` with `qa_mod`-style A1 arithmetic, and both recognise the 3-ideal `{3,6,9}` as structurally distinguished. They then diverge:

| Axis | Canonical QA | DRTH (Chase) |
|---|---|---|
| State space | Pairs `(b,e) ∈ D₉²` | Triples `(i,j,k) ∈ D₉³` |
| Third-coord extension of `(b,e)` to D₉³ | `d = qa_mod(b+e)` — **Fibonacci forward closure** | `d' = qa_mod(2e−b)` (superplane A) — **arithmetic-progression continuation**; see §2 |
| Full tuple | 4-tuple `(b, e, d, a = qa_mod(b+2e))` | Triple `(i, i⊕q, i⊕2q)` for a chosen stride V_P |
| Generator philosophy | **Dynamical**: `σ(b,e) = (e, qa_mod(b+e))` (Fibonacci time evolution) | **Structural**: `C_P(i,q) = Λ(i) ⊕ q⊙V_P` (affine section of a 3D lattice — not time) |
| Orbit partition | Cosmos (72) / Satellite (8) / Singularity (1) under σ | Superplanes A/B/C (243 each, partition D₉³\Λ) + Λ_diag (9) |
| Classification rule | `3∣b ∧ 3∣e ∧ not (9,9)` → Satellite | Stator `s ∈ {9,3,6}` confines orbit to coset Γ_{χ(i)} |
| Invariant | `norm_f(b,e) = b² + be − e²` with `v₃(f) ∈ {0} ∪ {≥2}` | Archetype function `χ(x) = ρ(x) mod 3` |

**QA and DRTH-A are not rewritings of one another.** They are two distinct affine extensions of the same mod-9 base pair `(b,e)` into `D₉³`: QA completes `(b,e)` by forward additive closure `b + e` (a time step), while DRTH-A completes it by arithmetic-progression continuation `2e − b` (a linear section). Same base; different third-coordinate rules; different semantics (dynamics vs. structure).

---

## Component overlap table

| DRTH object (source §) | QA equivalent | Notes |
|---|---|---|
| `D₉ = {1..9}`, `π: Z → D₉` (§2.1) | A1 + `qa_mod` | Identical operationally. |
| `⊕`, `⊗` DR operations (§2.1) | `qa_mod(a+b)`, `qa_mod(a·b)` | Identical. |
| Rotor `R = {1,2,4,5,7,8}` = U(D₉) (§2.2) | — (1D) | 1D projection of Cosmos; our Cosmos 2D = 72 pairs whose σ-orbits have period 24. |
| Stator `S = {9,3,6}` (§2.2) | — (1D) | 1D projection of "3∣ coord". Relates to Satellite ∪ Singularity 2D = 9 pairs. |
| Archetype cosets `Γ_G,Γ_R,Γ_B` (§2.3) | Z₉ / 3Z three cosets | Standard group-theoretic. We don't currently colour by this. |
| Complement `x* = π(9−x)` (§2.4) | Phason-flip / `b ↔ 10−b` | Already mapped in `QA_NUCLEUS_MAPPING.md`. |
| Triple cell `(i, i⊕q, i⊕2q)` (§3.3, superplane A) | **3-step window of σ** (with caveat below) | See §2 below for the exact correspondence. |
| Spinor transposition `σ(x,y,z) = (x,z,y)` (§4.1) | — | No direct 2D analogue; this is a D₉³-specific involution. |
| **Triple Mirror Guarantee** (Thm 4.2) | — | Structural theorem about A↔B under σ. No direct QA analogue. |
| **Stator Confinement Theorem** (Thm 5.2) | **`v₃(norm_f) ≥ 2 ⇔ non-Cosmos`** (see §3 below) | Same underlying fact — proven two different ways. |
| **Rotor Dispersion Theorem** (Thm 5.3) | Cosmos orbits have period 24 = π(9) | Same fact: rotors `r ∈ {1,2,4,5,7,8}` have order 9 in (D₉,⊕). |
| **Lambda Emergence** (Thm 6.2) | — | DRTH-specific; Λ is emergent as intersection of λ-columns. |
| Stride `V_A = (0,1,2)` | **Identifies a 3-step σ-window** | See §2. |
| Stride `V_B = (0,2,1)` | σ-window read in reverse | Spinor transpose of V_A. |
| Stride `V_C = (0,1,8)` = `(0,1,-1)` | `(b, b+q, b−q)` — additive dual | Not a σ-window. |
| Stride `V_Λ = (0,3,6)` (§3.3) | 3-torsion direction | Maps to our `d ≡ 0 (mod 3)` stratum. |
| **Autmorphism order 26,244** = 729·36 (§9.2) | — | Structural to D₉³, no 2D analogue. |
| **Triality Z₃ ⊂ Out(Aut)** (§9.3) | — | DRTH-specific. |
| SO(8) triality correspondence (§10.3) | — | Formal analogy only; see §5 below. |

---

## §1. D₉ and totient split are a clean 1D projection of our 2D QA

Chase's 1D rotor/stator split on D₉ is **the 1D shadow** of our 2D Cosmos/Satellite/Singularity split on D₉². Explicitly:

- Project `(b,e) ∈ D₉²` onto either coordinate.
- Chase's **stator condition** `x ∈ {3,6,9}` in 1D ⇔ our **`3∣b`** in one coord.
- Our **Satellite** = `3∣b ∧ 3∣e` (both coords in stator).
- Our **Singularity** = `b=e=9` (both coords at stator identity).
- Our **Cosmos** = at least one coord in rotor {1,2,4,5,7,8}.

So Chase's 1D `R/S = 6/3` maps to our 2D counts:
- Both coords in S: `3×3 = 9` (Satellite 8 + Singularity 1)
- At least one in R: `81 − 9 = 72` (Cosmos)

**No contradiction; his is the univariate marginal, ours the joint.**

---

## §2. Superplane A ↔ QA: two third-coordinate completions of `(b,e)`

Chase's superplane A cell is `C_A(i, q) = (i, i⊕q, i⊕2q)` — a 3-term arithmetic progression in D₉ with first term `i` and common difference `q`.

**Parameterization bridge.** Set `i = b` and `q = e ⊖ b` (the common difference that sends the first term to our `e`). Then:

- Second coord: `i ⊕ q = b ⊕ (e ⊖ b) = e` ✓
- Third coord: `i ⊕ 2q = b ⊕ 2(e ⊖ b) = qa_mod(2e − b)`

So **every (b,e) pair determines a unique superplane-A cell** under this parameterization:
```
C_A(b, e ⊖ b) = (b, e, qa_mod(2e − b))
```

Call its third coord `d' := qa_mod(2e − b)`. This is DRTH-A's completion of `(b,e)` to a triple.

QA's completion of the same `(b,e)` is `d := qa_mod(b + e)`.

**Direct comparison:**

| System   | Third coord given `(b,e)`     | Semantics                               |
|----------|-------------------------------|-----------------------------------------|
| QA       | `d  = qa_mod(b + e)`          | Forward additive closure — a σ time-step |
| DRTH-A   | `d' = qa_mod(2e − b)`         | Arithmetic-progression continuation     |

**Linear identity** (verified by direct computation, not conservation):
```
d + d'  =  (b + e) + (2e − b)  =  3e    (in Z, and so ≡ 3e mod 9)
d − d'  =  (b + e) − (2e − b)  =  2b − e
```

Both are linear functions of `(b,e)` — so their sum/difference is not a hidden topological invariant; it's just that the two extensions lie in a 2D linear sublattice of D₉³ over the base `(b,e)`. The sum `qa_mod(3e)` depends on `e` alone, via `e`'s archetype.

**When do `d` and `d'` coincide?** `d = d'` iff `b + e ≡ 2e − b`, i.e. `2b ≡ e (mod 9)`, i.e. `e = qa_mod(2b)`. That's a 1-parameter family — exactly the 9 pairs `{(b, qa_mod(2b)) : b ∈ D₉}`. For all other 72 pairs, QA and DRTH-A disagree on the third coordinate.

**Conclusion.** Superplane A and the QA σ-step are **two distinct linear extensions** `(b,e) → D₉³`, sharing the base but differing in the third coordinate rule. They are siblings, not rewrites, and not dual in any precise sense — just two linear sections of D₉³ rooted at the same `(b,e)` plane.

**Tooling.** `qa_core.js` exports both `qa_coord_d(b,e)` and `drth_coord_A(b,e)` so the two extensions can be compared live in the visualizations.

---

## §3. Stator Confinement ↔ `v₃(norm_f) ≥ 2` — same fact, different proofs

**Chase (Thm 5.2)**: if `s ∈ S`, then `C_P(i, s)` stays in coset `Γ_{χ(i)}`, because `q ⊗ v ≡ 0 (mod 3)` for any `v`.

**Ours (`qa_orbit_rules.py` docstring)**: `norm_f(b,e) = b² + be − e² ≡ 0 (mod 3) ⇔ b ≡ 0 ∧ e ≡ 0 (mod 3)`, and then `9 ∣ norm_f` automatically, so `v₃(f) ∈ {0} ∪ {≥2}` — never exactly 1.

Both say: **the 3-adic behaviour of D₉ arithmetic is all-or-nothing, skipping level 1**. Chase proves it via the generation action (multiplicative); we prove it via a quadratic form norm. Same underlying fact that gcd(3,9)=3 so `3∣x ⇒ 9∣x²` in Z/9Z.

**Worth porting**: we could expose this symmetry in `qa_core.js` as a helper `archetype(x) = ((x−1) % 3) + 1` and optionally add "color by archetype" as a third color mode alongside `class` and `orbit`.

---

## §4. Chase's Λ_diag ≠ our Satellite

**Important**: Chase's Lambda diagonal is `Λ = {(i,i,i) : i ∈ D₉}` = {111, 222, …, 999}. In his framework this is a **maximal-symmetry scalar-equivariant stratum**, not an orbit class.

In canonical QA, the 2D diagonal `b=e` is almost entirely **Cosmos** (see `qa_core.js` classifier): only (3,3), (6,6), (9,9) are non-Cosmos — (3,3) and (6,6) are Satellite, (9,9) is Singularity. The other diagonals (1,1), (2,2), (4,4), (5,5), (7,7), (8,8) are Cosmos.

So **"diagonal" means different things** in DRTH vs QA. DRTH's "diagonal" is a symmetric stratum of triples; our "diagonal" is merely the subset of pairs where both coordinates match. Don't conflate.

---

## §5. SO(8) triality — formal analogy, not derivation

Chase (§10.3) maps Fiber_A ↔ 8_v, Fiber_B ↔ 8_s, Fiber_C ↔ 8_c, Λ ↔ scalar, and identifies his τ with the D₄ outer automorphism.

**What's actually shown**: DRTH has a Z₃ automorphism rotating A → B → C and fixing Λ. This IS a real structural Z₃.

**What's NOT shown**: That this Z₃ is *the* D₄ triality, or that DRTH's fibers are in any sense the 8-dimensional representations. The dimensions don't even match (DRTH fibers have 243 elements, not 8).

The correspondence is at the level of **triality abstractly** (order-3 cyclic action on three objects with one fixed point), which is generic group theory. Calling this "discrete SO(8) triality" overstates the connection. Chase's own framing in §11 (Problem 1: "hybrid DR–octonionic envelope") implicitly acknowledges this is open.

**We should not repeat his SO(8) framing in QA contexts without independent verification.**

---

## §6. Claims to independently verify if we build on DRTH

1. **Aut completeness** (§9.2): `order 26,244`. Arithmetic `729 × 6 × 2 × 3` is correct; that these exhaust all automorphisms needs computer algebra (GAP / SageMath).
2. **Tensor rank 5 over Z₉** (Table B.4): asserted without proof.
3. **Möbius identification** (§8.3): "non-orientable discrete surface" from TRI-BLOCK fold — needs formal construction.
4. **Problem 2 (open)**: whether the non-lambda column family forms a complete set of mutually orthogonal Latin squares of order 9.
5. **Triple Mirror Guarantee** (Thm 4.2): proof sketch works; re-verify that the stride-vector-based derivation covers all edge cases (boundary at λ-column).

---

## §7. What could cleanly port into `qa_core.js` / Three.js

Low-risk, high-value additions inspired by DRTH without adopting its speculative parts:

1. **`archetype(x) = ((x−1) % 3) + 1`** in `qa_core.js` — the Γ_G/Γ_R/Γ_B coset labelling.
2. **Third color mode "archetype"** in both HTMLs — colours (b,e) by `(archetype(b), archetype(e))` pair, exposing the 3×3 = 9 coset-product blocks. These are Chase's "TRI-BLOCK" blocks.
3. **Stator/rotor indicator** per coordinate — already implicit in classify() but could be surfaced in telemetry as `(rotor(b) ∈ R/S, rotor(e) ∈ R/S)`.

**Declined (stays in DRTH's framework, not ours)**:
- Triple-cell D₉³ viz — different dimensionality; we'd be re-implementing Chase, not extending QA.
- SO(8) triality framing — speculative.
- Möbius TRI-BLOCK fold — speculative.

---

## Recommendation

Treat DRTH as a **legitimate sibling algebra** over the same alphabet, not as a claim about QA. It is not a competitor or a subsumer. The clean overlaps (D₉, totient split, stator confinement ↔ v₃ rule, complement involution) are already captured in our primitives; the divergent parts (triple cells, superplanes, triality group, SO(8) analogy) are Chase's own construction and should be cited, not absorbed.

**Concrete next action, if any**: add archetype coloring to `qa_core.js` (§7 item 1-2). That's ~15 min of work, directly motivated by DRTH, and purely QA-native.
