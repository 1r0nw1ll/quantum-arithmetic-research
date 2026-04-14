# Wildberger's G2 Integer Construction ↔ QA

**Status:** theory note, draft 2026-04-13
**Primary source:** Wildberger, N.J. *An Easy Construction of G2*. (UNSW preprint, read 2026-04-13, 9pp).
**Companion:** `QA_WILDBERGER_E8_RECONCILIATION.md` (negative result for E_8 via minuscule posets).

---

## 1. Summary

Wildberger's 2003 G2 construction gives an explicit 14-dimensional Lie algebra presentation using only an integer-weighted directed graph (the "G2 Hexagon") with 7 vertices and edges labeled by integers in {−2, −1, 1, 2}. This complements the simply-laced construction's E_8 exclusion with a **positive, integer-clean** template for an exceptional Lie algebra that *is* QA-friendly.

## 2. The G2 Hexagon — primary-source structure

Vertices (7 total): center v_φ plus 6 hexagonal vertices labeled by prefixes of the mnemonic **βαββαβ**:

  v_φ, v_β, v_βα, v_βαβ, v_βαββ, v_βαββα, v_βαββαβ.

Twelve roots (verbatim from Figure 2):

  ±β, ±(α+β), ±α, ±(α+2β), ±(α+3β), ±(2α+3β).

Operators X_γ for each root γ act on V = span(7 vertices). Each X_γ sends v to n·w when an edge of direction γ and weight n connects v → w, else 0. Every matrix entry of every X_γ is in {−2, −1, 0, 1, 2}, with at most one nonzero entry per row or column (monomial matrices).

Cartan operators H_γ = [X_{−γ}, X_γ] are scalar operators (multiplication by a root-dependent integer).

**Theorem 1.1** (verbatim): *The span of {X_γ, H_γ | γ a root} is closed under brackets and forms a 14-dimensional Lie algebra g ≃ g_2. A basis is {X_γ | γ a root} ∪ {H_α, H_β}.*

Structure constants are integers in {−2, −1, 0, 1, 2}, readable directly off the hexagon graph.

## 3. QA identifications

**Vertex count 7.** The representation space has dim 7. The integer 7 is the Keely Law 17 modulus (MEMORY: 21 = 7 · 3, noted in `project_keely_40_laws_complete.md`) and the smallest prime p for which Z/pZ is a field ≠ 2, 3 (UHG natural habitat).

**Root lattice Z α + Z β.** The 12 roots are integer linear combinations of basis vectors α, β. Under the QA identification (b, e) ↔ (coefficient of α, coefficient of β), the roots are:

| Root | (b, e) | On diagonal? |
|------|--------|--------------|
| α      | (1, 0) | e = 0 boundary |
| β      | (0, 1) | b = 0 boundary |
| α + β  | (1, 1) | **D_1** |
| α + 2β | (1, 2) | D_2 |
| α + 3β | (1, 3) | D_3 |
| 2α + 3β| (2, 3) | sibling of D_{3/2} |

Plus their negatives. **All 12 roots occupy 6 distinct QA diagonal classes** (one per sign pair). The hexagonal symmetry of G2 roots corresponds in QA to pairs (p, −p) of diagonal classes.

**Structure constants in {−2, −1, 0, 1, 2}.** Exactly the set {−2, −1, 1, 2} of signed small integers that appear as QA step multipliers under T-operator iteration mod small moduli. This is not a proof of identification but a suggestive coincidence.

**Cert candidacy.** Genuine identification requires proving that the QA orbit structure at m = 7 (or some related modulus) reproduces the g_2 commutation relations. This is a *computation* rather than a structural claim — verifiable by Codex in one session. Preliminary check worthwhile: do the 12 roots, viewed as (b, e) pairs mod 7, form a closed orbit under some T-step rule?

## 4. Contrast with the E_8 reconciliation

The companion note `QA_WILDBERGER_E8_RECONCILIATION.md` records a **negative** result: Wildberger's minuscule-poset construction cannot produce E_8.

The G2 construction is a **positive** complement: it is an integer Lie-algebra construction whose natural QA-embedding has visible structure (12 roots on 6 diagonal classes in a lattice of integer linear combinations).

**Open strategic question.** The five exceptional simple Lie algebras are g_2 (dim 14), f_4 (dim 52), e_6 (dim 78), e_7 (dim 133), e_8 (dim 248). Wildberger has explicit integer constructions for g_2, e_6, e_7 (via minuscule posets for e_6/e_7, explicit hexagon for g_2). The f_4 case is not covered by either method. Whether QA offers any handle on f_4 or e_8 via different machinery — possibly through the exceptional Lie lattice's Gram matrix directly — is the open question from the E_8 doc §4.

## 5. Preliminary QA-embedding check (2026-04-13)

Quick exhaustive test: after shifting roots (b, e) to QA's no-zero alphabet via A1, the 12 G2 roots collapse to **10 distinct QA points** at each of m ∈ {7, 9, 12, 24}. Two pairs coincide (likely ±α or ±β pairs after the mod shift).

The 12 roots are **not** closed under QA addition (e.g., 2β = (0, 2) is not a root). Only 60/144 pairs of roots sum to another root; the rest escape to non-root points.

**Interpretation.** The naïve identification (b, e) = (coefficient of α, coefficient of β) does not give a clean orbit closure at any small m. The G2 root system is a genuine lattice rather than a QA orbit — they share integer scaffolding but are not isomorphic under the obvious map.

This is not a killing blow. The correct QA ↔ G2 embedding likely goes through the **representation space** (the 7-vertex hexagon) rather than the **root system** (the 12 roots). A 7-point orbit under some T-step at suitable m, decorated with the integer weights {−2, −1, 1, 2}, is a different candidate — closer in spirit to Wildberger's hexagon, different in QA-embedding.

## 6. Revised action items

- [ ] Look for T-orbits at small m (m = 7, 13, 14) with exactly 7 points, and check whether the induced adjacency/weight structure reproduces the G2 hexagon labels.
- [ ] Read Wildberger's e_6 / e_7 sections of Adv. Appl. Math. 2003 for cleaner integer-embedding candidates.
- [ ] Downgrade [223] G2 cert from candidate to **open exploration** — no cert scaffolding until the embedding is found.

## 6. References

- Wildberger, N.J. *An Easy Construction of G_2*. UNSW preprint (7-vertex hexagon + center, weight labels in {−2, −1, 1, 2}, all operators integer-matrix).
- Companion: `QA_WILDBERGER_E8_RECONCILIATION.md`, `QA_UHG_PROJECTIVE_COORDINATES.md`.
- MEMORY: `project_keely_40_laws_complete.md` (7 = Law 17 modulus, 21 = 7·3).
