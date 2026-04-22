# Fidelity Audit — wildberger_object_model.md

**Auditor:** general-purpose subagent, fresh context, 2026-04-22.
**Scope:** fidelity of the draft object model to its cited sources. Not utility.

**Sources consulted (line counts):**

- Draft: `/home/player2/signal_experiments/qa_tla_blind/mode_b/wildberger_object_model.md` (299 lines)
- Cert [231] `docs/families/231_qa_hyper_catalan_diagonal_cert.md` (47 lines)
- Cert [232] `docs/families/232_qa_uhg_diagonal_coincidence_cert.md` (47 lines)
- Cert [233] `docs/families/233_qa_uhg_orbit_diagonal_profile_cert.md` (53 lines)
- Cert [234] `docs/families/234_qa_chromogeometry_pythagorean_identity_cert.md` (46 lines)
- Cert [235] `docs/families/235_qa_super_catalan_diagonal_cert.md` (61 lines)
- Cert [236] `docs/families/236_qa_spread_polynomial_composition_cert.md` (60 lines)
- Cert [237] `docs/families/237_qa_4d_diagonal_rule_cert.md` (61 lines)
- Cert [241] `docs/families/241_qa_quadruple_coplanarity_cert.md` (64 lines)
- Cert [244] `docs/families/244_qa_mutation_game_root_lattice_cert.md` (first 30 lines)
- Cert [246] `docs/families/246_qa_chromogeometric_tqf_symmetry_cert.md` (75 lines)
- Theory `docs/theory/QA_UHG_PROJECTIVE_COORDINATES.md` (207 lines)
- Theory `docs/theory/QA_WILDBERGER_STRUCTURAL_FOLLOWUPS.md` (111 lines)
- Validator `qa_alphageometry_ptolemy/qa_chromogeometry_pythagorean_identity_cert_v1/qa_chromogeometry_pythagorean_identity_cert_validate.py` (103 lines)

---

## Per-primitive findings

### 1. Points (draft §Primitive objects, lines 46-54)

- Draft says: "A point is an integer pair `(x, y) ∈ Z²` or a projective class `[x : y : z] ∈ P²(Z)` (UHG)." Under QA coords a point is `(b, e)` with derived `d = b+e`, `a = b+2e`; the 4-tuple `(b, e, d, a)` is one point in four projections.
- Cert/theory source: `QA_UHG_PROJECTIVE_COORDINATES.md` §2 (line 52): `[b : e : d] = [b : e : b + e] ∈ P²(F)`. Cert [237] (lines 10-12) gives the 4-tuple and its embedding into R⁴.
- Verdict: **faithful.**

### 2. Lines (draft lines 56-58)

- Draft says: "A line is a projective triple `⟨l : m : n⟩` satisfying `lx + my + nz = 0` for incident points. Integer-coefficient."
- Source: `QA_UHG_PROJECTIVE_COORDINATES.md` §1 (line 24): "Dual line L associated to point a = [x:y:z] is the line with proportion (x:y:z)_line, i.e. points b = [u:v:w] satisfying xu + yv − zw = 0." (Note UHG is Minkowski-signature, so the correct incidence in that form uses `xu + yv − zw`, not `xu + yv + zw`. The draft states the *affine projective* incidence, not the UHG-bilinear one.)
- Verdict: **inferred-not-cited / minor notational ambiguity.** The generic affine projective line incidence `lx+my+nz=0` is standard and not wrong, but the draft collapses two different forms (generic affine projective incidence vs UHG's Minkowski-signature dual). For Mode B object-model purposes this is harmless.

### 3. Quadrance — blue/red/green (draft §3, lines 60-71)

- Draft says:
  - Blue: `Q_b(P₁, P₂) = (x₁ - x₂)² + (y₁ - y₂)²`
  - Red: `Q_r(P₁, P₂) = (x₁ - x₂)² - (y₁ - y₂)²`
  - Green: `Q_g(P₁, P₂) = 2 · (x₁ - x₂) · (y₁ - y₂)`
  - QA coords: `Q_b = b² + e²`, `Q_r = b² - e² = (b-e)(b+e) = (b-e)·d`, `Q_g = 2 b e`.
- Cert [234] says (lines 9-14): `Q_b=b*b+e*e`, `Q_r=b*b-e*e`, `Q_g=2*b*e`. Validator (line 55) fixes formula strings: `Q_r: (b-e)*d`, `Q_g: 2*b*e`, `Q_b: b*b + e*e`.
- Cert [241] (lines 17-19) and [246] (lines 12-14) restate these identically.
- Verdict: **faithful.**

### 4. Spread — scalar definition (draft §4, lines 74-80)

- Draft says: `s(L₁, L₂) = (m₁ - m₂)² / ((1 + m₁²)(1 + m₂²))`, and vector form `s = 1 - (v₁·v₂)²/(|v₁|²|v₂|²) = (v₁ × v₂)²/(|v₁|²|v₂|²)`.
- Source: matches `QA_UHG_PROJECTIVE_COORDINATES.md` §1 line 36 structurally: `S(L₁, L₂) = 1 − ⟨L₁, L₂⟩² / (⟨L₁,L₁⟩·⟨L₂,L₂⟩)` (UHG dual form). Divine Proportions scalar form is standard.
- Verdict: **inferred-not-cited.** The formula is standard Wildberger and correct; it is not restated verbatim in the nine cited cert docs. It is covered by `QA_UHG_PROJECTIVE_COORDINATES.md` §1 and is background for cert [236].

### 5. Triple Quad Formula — TQF (draft §5, lines 84-92)

- Draft says: `TQF(Q₁, Q₂, Q₃) = (Q₁ + Q₂ + Q₃)² - 2(Q₁² + Q₂² + Q₃²)`; "TQF = 0 ⟺ the three points are collinear"; "`TQF_r = TQF_g = -TQF_b`"; "`TQF_b = 16 · (signed area)²`".
- Cert [246] lines 14-26: `TQF(Q1,Q2,Q3) = (Q1+Q2+Q3)*(Q1+Q2+Q3) - 2*(Q1*Q1+Q2*Q2+Q3*Q3)`; `TQF_r = TQF_g = -TQF_b`; `TQF_b = 4*area2*area2 = 16*A*A`.
- Verdict: **faithful.** (Note for pedantry: the canonical Divine-Proportions triple-quad identity for collinear points is `(Σq)² = 2·Σq² + 4q₁q₂q₃`. Cert [246] adopts the *residue* form `(Σq)² - 2·Σq²`, which equals `4q₁q₂q₃` = `16·A²`. The draft's statement matches cert [246]'s residue convention; this is consistent in-repo.)

### 6. Cross-ratio (draft §6, lines 94-98)

- Draft says: "For four collinear points: `(A, B; C, D) = (AC · BD) / (AD · BC)`".
- Source: not explicitly stated in any of the nine cited certs. `QA_UHG_PROJECTIVE_COORDINATES.md` §3 (line 81) introduces a *different* object called "QA cross-ratio" `κ(a₁,a₂) := (b₁e₂+e₁b₂)²/(4b₁e₁b₂e₂)` which is a two-point UHG quantity, not the classical four-point projective cross-ratio.
- Verdict: **inferred-not-cited.** Draft states the textbook classical projective cross-ratio (correct Wildberger-adjacent material), but it is not backed by any of the nine cited certs. Separately, the draft should not be read as equating this with the "QA cross-ratio" in the UHG theory doc (the draft doesn't claim this; noted for completeness).

### 7. Spread polynomials (draft §7, lines 100-105)

- Draft says: `S_0 = 0, S_1 = s, S_{n+1} = 2(1-2s)S_n - S_{n-1} + 2s`; composition monoid `S_n ∘ S_m = S_{n·m}` (cert [236]).
- Cert [236] lines 10-12: `S_0=0`, `S_1=s`, `S_{n+1}=2(1-2s)*S_n - S_{n-1} + 2s`; "`S_n` composed with `S_m` equals `S_{n*m}`".
- Verdict: **faithful.**

### 8. Hexagonal ring decomposition / SL(3) (draft §8, lines 107-119)

- Draft says: `ring(a, b) = T_{d+1} + b · e where d = a + b, T_n = n(n+1)/2`. Decomposition: D-component `T_{d+1}` (sum-only) + B-component `b·e` (bilinear).
- Source: `QA_WILDBERGER_STRUCTURAL_FOLLOWUPS.md` §1 line 19-24: "ring(a, b) := dim π[a, b] − dim π[a−1, b−1] = ... = **T_{d + 1} + a·b** where d = a + b" and under QA coords `(b_QA, e_QA) = (a, b)`, "ring = T_{d+1} + b·e".
- Verdict: **faithful.** Also correctly flags the D/B-component split per line 43 of the source.

### 9. 4D diagonal rule (draft §9, lines 121-126)

- Draft says: "A 2-plane `Π ⊂ R^4` with orthogonal basis `{u, v}` has a *diagonal quadrance* `Q(Π) = (u·v)² - (u·u)(v·v)` (Plücker-style). Cert [237] shows this corresponds exactly to QA's 4-tuple `(b, e, d, a)` under a canonical embedding."
- Cert [237] lines 9-16: "The QA tuple `(b,e,d,a)` with `d=b+e` and `a=b+2e` is exactly the point `b*v1+e*v2` in the 2-plane of `R^4` spanned by `v1=(1,0,1,1)` and `v2=(0,1,1,2)`. The Gram matrix is `[[3,3],[3,6]]`, with determinant `9`... Diagonal Rule `Q1+Q2=Q3`."
- Verdict: **drift-minor.** Cert [237]'s claim is **not** `Q(Π) = (u·v)² - (u·u)(v·v)`. Cert [237] asserts the diagonal rule in the form `Q1+Q2=Q3` for perpendicular pairs, backed by a specific Gram matrix `[[3,3],[3,6]]` with `det = 9 = canonical modulus m`. The draft's `Q(Π) = (u·v)² - (u·u)(v·v)` expression is the Plücker / Gramian-determinant form, which is mathematically related (it is the negative of the Gram determinant for orthogonal `{u,v}`: if `u·v = 0` then `Q(Π) = -(u·u)(v·v)`). But that is **not** what cert [237]'s validator checks or what the cert's "Diagonal Rule" refers to. The draft's line "Cert [237] shows this corresponds exactly..." over-states the correspondence. Readers should see either the `Q1+Q2=Q3` diagonal-rule form **or** the Gram-determinant `det = 9` form, with attribution to which cert primitive is being used.

### 10. Hyper-Catalan / Super Catalan (referenced in draft source-material header, lines 19-22, 25-26; not re-listed in "Primitive objects")

- Draft says (header only): cert [231] "hyper-Catalan diagonal" and cert [235] super-Catalan exist (via family notations).
- Cert [231] lines 8-17: QA identification `b := V_m - 1`, `e := F_m`, `d = E_m`, Euler identity auto-satisfied; single-type sibling diagonals `b = (k-1)e + 1`.
- Cert [235] lines 8-16: `(b,e)=(m,n)` with `d=b+e`, `S(m,n)=(2m)!(2n)!/(m!n!(m+n)!)`, denominator is `d!`.
- Verdict: **faithful** (claim is only existence, no formula re-stated in draft; nothing to drift).

### 11. 4-tuple `(b, e, d, a)` embedding (draft §"4-tuple", lines 181-195)

- Draft says: `d = b+e`, `a = b+2e`, `a - d = e`, `d - a = -e`, 4-tuple is one degree of over-specification.
- CLAUDE.md (QA Axiom A2) and cert [237] agree: `d = b+e, a = b+2e`. Arithmetic check: `(b+2e) - (b+e) = e`. Correct.
- Verdict: **faithful.**

### 12. Three-metric chromogeometric structure (draft §"Three-metric", lines 161-178)

- Draft says: Blue `Q_b = dx² + dy²`, Red `Q_r = dx² - dy²`, Green `Q_g = 2 dx dy`; Pythagorean `Q_b² = Q_r² + Q_g²` (cert [234]); `TQF_r = TQF_g = -TQF_b` (cert [246]).
- Cert [234] lines 12-15 and cert [246] lines 12-21: match exactly.
- Note: Draft describes Red signature as `(1, -1)` and Green signature as `(0, 2)`. The Green quadrance `Q_g = 2·dx·dy` has matrix `[[0,1],[1,0]]` whose signature is `(+, -)` (one positive and one negative eigenvalue at `±1`), not `(0, 2)`. This is a **minor signature-label error**: green is a mixed / hyperbolic form, not degenerate. Cert [234]/[246] do not themselves give signature labels, so this mis-labeling is not *contradicted* by a cert, but it is a technically incorrect parenthetical. Flag as **drift-minor**.
- Verdict on identities themselves (the load-bearing content): **faithful**; signature-label parenthetical is **drift-minor**.

---

## Per-transform findings

### 1. Translations (draft §Transforms 1, lines 132-134)

- Draft says: `T_{(a,b)}(x, y) = (x + a, y + b)`, preserves all three quadrances.
- Source: standard; not in a specific cited cert.
- Verdict: **inferred-not-cited** (trivially true of all three chromogeometric forms since they depend only on differences).

### 2. Reflections (draft §Transforms 2, lines 136-140)

- Draft says: `R_x(x,y) = (-x, y)`, `R_y(x,y) = (x, -y)`. "Preserve `Q_b`; flip `Q_r → Q_r, Q_g → -Q_g` (or symmetric)."
- Check: under `R_y(x,y)=(x,-y)`, `Q_r = x² - y²` is preserved (`y`-sign drops in the square), `Q_g = 2xy` flips sign. Correct. The parenthetical "or symmetric" covers `R_x`.
- Source: not in a specific cited cert, but polynomial-level obvious.
- Verdict: **inferred-not-cited.** Correct.

### 3. Rotations via spread polynomials (draft §Transforms 3, lines 142-146)

- Draft says: "A rotation by spread `s` applied to `(x, y)`: use the rotation matrix parameterised by spread rather than angle. Composition via spread polynomial monoid. All coords stay rational."
- Cert [236] establishes the *composition monoid* on `S_n`, and the source paper is titled "Spread polynomials, rotations and the butterfly effect". The cert itself does not verify that a rotation-by-`s` applied to points `(x,y)` yields an integer or rational image (indeed, rational-spread rotations generally take Z² out of Z²; they stay in Q²). The draft's "all coords stay rational" is correct (Q²), but this is a plausibility/framework statement, not a cert-verified one.
- Verdict: **inferred-not-cited** (framework-level; correct, backed by the Goh-Wildberger paper title/abstract but not by cert [236]'s checks, which are purely about polynomial-composition identities).

### 4. Projective maps / UHG (draft §Transforms 4, lines 148-152)

- Draft says: `(x, y, z) ↦ M · (x, y, z)^T` for integer 3×3 M with nonzero det; preserves cross-ratios; does not preserve quadrances.
- Cert [232]/[233] context: UHG quadrance is defined on projective equivalence classes (invariance under scalar multiplication). Neither cert verifies cross-ratio invariance under generic projective transforms. `QA_UHG_PROJECTIVE_COORDINATES.md` §3 implies this via the QA-coincidence criterion `b₁/e₁ = b₂/e₂`.
- Verdict: **inferred-not-cited.** Standard projective-geometry fact; not verified by any cited cert but not contradicted either.

### 5. Mutation moves (draft §Transforms 5, lines 154-159)

- Draft says: "mutation sends a root `α_i → -α_i + Σ |⟨α_i, α_j⟩| α_j`. Integer-coefficient; generates the Weyl group. Cert [244] ties this to E_8 construction."
- Cert [244] (lines 1-22): **Status: PASS** (GREEN). "`|Orbit| = 240`, with `120` positive roots and `120` negative roots, every root satisfying `v^T G v = 2` for `G = 2I - A`". The cert **verifies** the mutation closure and E_8 construction; the validator uses an explicit 8-dimensional root orbit.
- Verdict: **faithful on status, but with one caveat.** The audit prompt hinted "cert [244] not green" and "model should flag this as 'staged, not live.'" This hint is **stale**: cert [244] is in fact GREEN/PASS (status line 3 of cert doc). The draft correctly treats [244] as live. (`QA_WILDBERGER_STRUCTURAL_FOLLOWUPS.md` table line 100 still shows [244] as "staged (2020 paper)" but the authoritative cert-doc status line is PASS.) No fix needed on this point; the draft is consistent with current cert status.
- One minor: the exact formula Wildberger uses in the mutation game does not have absolute-value bars `|⟨α_i,α_j⟩|` on every summand by default in all presentations; but cert [244]'s claim statement doesn't restate the mutation formula explicitly in the portion I can see, so the draft's rendition is reasonable and matches the standard generalized-root-system convention.

---

## Exclusions check (draft §"What this model DOES NOT include", lines 245-258)

Claims:
1. "No continuous / transcendental primitives" — consistent with Divine Proportions' rational-trig philosophy, no cert contradicts.
2. "Non-integer coefficients... everything is in Z[1/2] at worst" — no cert contradicts; cert validators are integer-exact.
3. "No axiom-of-choice moves / bounded quantifiers" — consistent with cert philosophy.
4. "No higher-grade Clifford elements (bivectors, trivectors as full GA multivectors)" — correct; no cert in the ecosystem asserts Clifford-algebra-in-the-Hestenes-sense primitives. Wildberger's chromogeometry is a three-metric projective/affine bundle, not a multivector algebra.

Verdict: **exclusions are consistent with the cert ecosystem.**

---

## Aggregate verdict

**`minor-drift`** — object model is substantially correct and lifts primitives accurately from cited certs. Three items need attention before commit:

1. **[drift-minor] §9 "4D diagonal rule"** (draft lines 121-126): the formula `Q(Π) = (u·v)² - (u·u)(v·v)` is a Plücker-style Gram-determinant expression, but cert [237]'s actual claim is `Q1+Q2=Q3` for perpendicular QA-tuple pairs with `Gram-det = 9 = canonical modulus`. The draft's "corresponds exactly" wording over-states the match. Either restate cert [237] in its own terms (diagonal rule `Q1+Q2=Q3` plus `Gram-det = 9`) or add a derivation line connecting the Plücker form to the cert's Gram form.

2. **[drift-minor] §"Three-metric" green signature label** (draft line 166): `Q_g` is labelled signature `(0, 2)`, which is incorrect. The matrix `[[0,1],[1,0]]` has eigenvalues `±1`, i.e. signature `(+, -)` = `(1, 1)`. The correct label is "mixed (indefinite, signature `(1, 1)` like red)" or simply "mixed, indefinite, null-cone = coordinate axes". This is a parenthetical label, not a load-bearing identity, but worth fixing.

3. **[inferred-not-cited, not a drift, just a disclosure nit] §6 cross-ratio**, §4 spread scalar/vector forms, §Transforms 1/2/3/4: these are correct Wildberger / Divine-Proportions material but are **not cert-backed** among the nine cited certs. The draft's provenance note (line 290-292) says "all primitives cited above are established in the repo's cert ecosystem" — that is over-broad. Several primitives are lifted from Wildberger's broader corpus / theory docs, not from the specific cert-family docs. Consider softening the provenance note to "all primitives are established in the repo's cert ecosystem **or in Divine-Proportions-level Wildberger theory docs cited in `QA_UHG_PROJECTIVE_COORDINATES.md`**".

No primitives are **misstated** in a way that would mislead a reproducer's encoding: the chromogeometric quadrances, TQF, spread-polynomial recurrence + composition law, SL(3) ring identity, and 4-tuple embedding are all formula-faithful.

---

## Fix list (ordered)

1. **Draft lines 121-126 (§9 "4D diagonal rule"):** Replace `Q(Π) = (u·v)² - (u·u)(v·v) (Plücker-style). Cert [237] shows this corresponds exactly to QA's 4-tuple...` with a statement closer to cert [237]: e.g.,
   > "Cert [237] shows the QA 4-tuple `(b,e,d,a)` is exactly the lattice point `b·v₁ + e·v₂` in the 2-plane of R⁴ spanned by `v₁=(1,0,1,1), v₂=(0,1,1,2)`. The Gram matrix is `[[3,3],[3,6]]` with `det = 9`, matching QA's canonical modulus. Perpendicular QA-tuple pairs in this 2-plane satisfy Wildberger's diagonal rule `Q₁ + Q₂ = Q₃`."
   The Plücker form can be mentioned as a related general-R⁴ expression, but do not claim it "corresponds exactly" to cert [237].

2. **Draft line 166 (§"Three-metric", Green signature):** change `(0, 2)` to `indefinite, null-cone = coordinate axes` or `(1, 1) signature, mixed`. Green's Gram `[[0,1],[1,0]]` has eigenvalues `±1`.

3. **Draft line 292 (§Provenance):** soften "all lifted from existing certs or theory docs" to "all lifted from existing certs, or from the Wildberger theory docs cited as background (Divine Proportions, Chromogeometry paper, UHG I-IV, spread-polynomial paper, Mutation Game paper). Transforms (translations, reflections, rotations-via-spread, projective maps) are Wildberger-standard background, not cert-verified."

4. **(Optional, nice-to-have):** draft line 158 mentions mutation moves with reference to cert [244]. The draft's rendering of the mutation formula (`α_i → -α_i + Σ |⟨α_i, α_j⟩| α_j`) is standard-textbook but the exact in-cert convention may differ slightly. If desired, adjust to match cert [244] validator's exact form. This is not blocking.

---

## Confidence

**High.** Every load-bearing formula in the draft was compared against either the cited cert's `## Claim` section or (for the spread-polynomial case) the cert validator's recurrence. The two drift-minor items (`Q(Π)` expression and green signature label) are each textually pinpointable. The inferred-not-cited items are all standard Wildberger primitives that the draft represents correctly; the only concern is the over-broad provenance claim, which is a disclosure fix, not a correctness fix.

**Recommendation:** fix the 3 items in the Fix list above (total: ~15 lines of textual edit across three sections), then the draft is commit-ready for Mode B purposes.
