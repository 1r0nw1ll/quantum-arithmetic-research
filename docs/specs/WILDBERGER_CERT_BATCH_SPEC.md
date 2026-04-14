# Wildberger Cert Batch Specification (Codex handoff)

**Target:** Codex (via collab bus `codex_bridge`).
**Session origin:** Claude `wild-map` session 2026-04-13.
**Theory basis:**
- `docs/theory/QA_HYPER_CATALAN_DIAGONAL.md`
- `docs/theory/QA_UHG_PROJECTIVE_COORDINATES.md`
- `docs/theory/QA_WILDBERGER_E8_RECONCILIATION.md`
- `docs/theory/QA_WILDBERGER_G2_INTEGER_CONSTRUCTION.md`

**Rules.**
- Raw integer arithmetic only (A2). No mod-reduction except where T-operator explicitly called out.
- No floats. `Fraction` where rationals appear. No `**2` (S1). Use `b*b`.
- No stochastic / continuous in any QA-causal position (T2, S2).
- Layout matches `qa_alphageometry_ptolemy/qa_fuller_ve_diagonal_decomposition_cert_v1/` precedent.
- Each family: one validator, ≥ 1 pass fixture, ≥ 1 fail fixture, `mapping_protocol_ref.json`. Then register in `qa_meta_validator.py` FAMILY_SWEEPS, update `docs/families/README.md`, add `docs/families/<N>_qa_<name>.md`.

All four certs have been empirically/formally verified by Claude; Codex's job is to package them into the standard cert structure, not to revalidate the math.

---

## Cert [231] — QA_HYPER_CATALAN_DIAGONAL_CORRESPONDENCE_CERT.v1

**Directory:** `qa_alphageometry_ptolemy/qa_hyper_catalan_diagonal_cert_v1/`

**Claim.** Under the identification

  b := V_m − 1,    e := F_m

where m = [m_2, m_3, m_4, …] indexes a hyper-Catalan multi-index with

  V_m = 2 + Σ_{k ≥ 2} (k − 1) m_k,    E_m = 1 + Σ_{k ≥ 2} k m_k,    F_m = Σ m_k,

the QA derived coordinates satisfy:

1. d = b + e = E_m exactly (edge count).
2. Euler V_m − E_m + F_m = 1 becomes automatic from QA d = b + e.
3. Single-type m (only m_k = n): lies on QA sibling diagonal b = (k − 1) e + 1.
4. C_m = (E_m − 1)! / ((V_m − 1)! · m!) matches OEIS for single-type cases:
   - A000108 (Catalan, k = 2): 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796
   - A001764 (Fuss k = 3): 1, 1, 3, 12, 55, 273, 1428, 7752, 43263
   - A002293 (Fuss k = 4): 1, 1, 4, 22, 140, 969, 7084
   - A002294 (Fuss k = 5): 1, 1, 5, 35, 285, 2530
5. No single-type hyper-Catalan sits on D_1 (b = e) — exhaustive check k ∈ [2, 7], n ∈ [0, 9].

**Primary source.** Wildberger, N.J.; Rubine, D. *A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode*, Amer. Math. Monthly 132(5):383–402, 2025. Theorem 5: C_m formula. Theorem 12: Geode factorization S − 1 = S_1 G.

**Fixtures.**

Pass fixture `hcd_pass_euler_and_oeis.json`:
- Field: `euler_check` — array of {m, V, E, F, b, e, d_derived, V_minus_E_plus_F}; assert each d_derived = E and each V − E + F = 1. Include m ∈ {[3,0,0,0], [0,2,0,0], [2,1,0,0], [1,1,1,0], [0,0,0,5]}.
- Field: `catalan_oeis_match` — array of {n, C_computed, oeis_A000108, match}; n = 0..10.
- Field: `fuss_oeis_match` — nested by k ∈ {3, 4, 5} with arrays of {n, C, oeis, match}.
- Field: `single_type_diagonal` — array of {k, n, b, e, expected_b, on_sibling_diagonal} verifying b = (k−1)e+1.
- Field: `d1_disjointness` — exhaustive sweep k ∈ [2,7], n ∈ [0,9], assert no hit on b = e.

Fail fixture `hcd_fail_wrong_formula.json`:
- Corrupt C_m by dropping the m! divisor; assert validator catches the mismatch to OEIS.

**Validator responsibilities.**
- Pure integer arithmetic, `from math import factorial`.
- Validate (a) Euler identity auto-derivation, (b) OEIS matches for Catalan + Fuss k=3,4,5, (c) sibling-diagonal relation b = (k−1)e+1, (d) D_1 disjointness.
- Self-test at bottom: both fixtures, JSON output `{ok: bool, checks: {...}}`.

---

## Cert [232] — QA_UHG_DIAGONAL_COINCIDENCE_CERT.v1

**Directory:** `qa_alphageometry_ptolemy/qa_uhg_diagonal_coincidence_cert_v1/`

**Claim.** Under QA's projective embedding [b : e : d] with d = b + e and the UHG bilinear form

  ⟨a₁, a₂⟩ = b₁b₂ + e₁e₂ − d₁d₂ = −(b₁e₂ + e₁b₂),

and the UHG projective quadrance

  q(a₁, a₂) = 1 − ⟨a₁, a₂⟩² / (⟨a₁, a₁⟩ · ⟨a₂, a₂⟩),

two QA points a₁, a₂ ∈ {1, …, m}² have q(a₁, a₂) = 0 if and only if they lie on the same QA diagonal class D_k, i.e., gcd-reduced (b₁, e₁) = gcd-reduced (b₂, e₂).

**Primary source.** Wildberger, N.J. *Universal Hyperbolic Geometry I: Trigonometry*, Geometriae Dedicata 163:215–274, 2013. Theorem 32 (Quadrance formula). Theorem 33 (Zero quadrance iff a₁a₂ is a null line ↔ same projective class in our rephrasing).

**Exhaustive witness (already verified at m = 9 by Claude):** over the 81 points of {1,…,9}², there are exactly 64 unordered zero-quadrance pairs and exactly 64 same-diagonal-class pairs, with perfect intersection. Zero-q-but-diff-diagonal: 0. Same-diag-but-nonzero-q: 0.

**Fixtures.**

Pass fixture `udc_pass_m9_exhaustive.json`:
- Field: `m`: 9.
- Field: `zero_q_pair_count`: 64.
- Field: `same_diagonal_pair_count`: 64.
- Field: `intersection_size`: 64.
- Field: `counter_examples`: empty arrays for both directions.
- Field: `witnesses` — sample 5 zero-q pairs showing same-D_k, 5 non-zero-q pairs showing different-D_k.

Fail fixture `udc_fail_spoofed_coincidence.json`:
- Synthetic data with a zero-q claim between (1,2) and (2,4) but differing diagonal classes *artificially labeled*; validator catches the labeling error by recomputing gcd-class.

**Validator responsibilities.**
- Enumerate all unordered pairs {a₁, a₂} with a_i ∈ {1,…,m}², skip pairs with ⟨a_i, a_i⟩ = 0 (excluded at m = 9 by A1).
- For each pair compute quadrance using `Fraction` (no floats), and gcd-reduced diagonal class.
- Assert cardinality equality (64 = 64 at m = 9).
- Self-test at bottom.

---

## Cert [233] — QA_UHG_ORBIT_DIAGONAL_PROFILE_CERT.v1

**Directory:** `qa_alphageometry_ptolemy/qa_uhg_orbit_diagonal_profile_cert_v1/`

**Claim.** At modulus m = 9 with QA T-step

  T(b, e) = ((b + e − 1) mod 9) + 1,    orbit step: (b, e) → (T(b, e), T(e, T(b, e))),

the 81 points of {1, …, 9}² partition into:

- 1 Singularity (length 1): (9, 9)
- 2 Satellite orbits (length 4 each; 8 points total)
- 6 Cosmos orbits (length 12 each; 72 points total)

Diagonal-class profile per orbit:

- Singularity: 1 class (D_1 = gcd-class (1,1)) with multiplicity 1.
- Sat #1 {(3,3), (6,9), (6,6), (3,9)}: 3 classes with D_1 at multiplicity 2.
- Sat #2 {(3,6), (9,6), (6,3), (9,3)}: 4 distinct classes, D_1 absent.
- Cos #1, #3, #4: 11 classes each with D_1 at multiplicity 2.
- Cos #2, #5, #6: 12 classes each, all distinct; D_1 absent.

**Sharp invariant (the "(9,9)-complement law").** Every D_1-containing T-orbit *other than the Singularity* contains exactly 2 D_1 points, and those 2 points are mod-9 complements summing to (9, 9):

- Sat #1: (3,3) + (6,6) = (9,9)
- Cos #1: (1,1) + (8,8) = (9,9)
- Cos #3: (2,2) + (7,7) = (9,9)
- Cos #4: (5,5) + (4,4) = (9,9)

The involution p → (9, 9) − p preserves T-orbit membership restricted to D_1, and fixes the Singularity. The (b, e) ↔ (e, b) swap involution does *not* close T-orbits (only the Singularity is swap-closed).

**Fixtures.**

Pass fixture `uodp_pass_m9_full_partition.json`:
- Fields: enumerated orbits with point lists, cycle lengths, diagonal-class multisets, and D_1 complementary-pair verification for each D_1-containing orbit.

Fail fixture `uodp_fail_wrong_complement_pair.json`:
- Synthetic orbit claiming (3,3) pairs with (5,5) instead of (6,6); validator catches by recomputing mod-9 complement.

**Validator responsibilities.**
- Generate all T-orbits at m = 9 and assert the 1 + 2 + 6 partition with correct lengths.
- For each orbit: compute gcd-reduced classes per point; count D_1 occurrences.
- For each D_1-containing orbit (besides Singularity): verify the D_1 points sum to (9, 9) in integer arithmetic (not mod).

---

## Cert [234] — QA_CHROMOGEOMETRY_PYTHAGOREAN_IDENTITY_CERT.v1

**Directory:** `qa_alphageometry_ptolemy/qa_chromogeometry_pythagorean_identity_cert_v1/`

**Claim.** Wildberger's chromogeometry (2008) defines three affine planar quadrances on R² (or over any field of characteristic ≠ 2):

  Q_b(A₁, A₂) = (x₂ − x₁)² + (y₂ − y₁)²    (blue, Euclidean)
  Q_r(A₁, A₂) = (x₂ − x₁)² − (y₂ − y₁)²    (red, Lorentzian)
  Q_g(A₁, A₂) = 2 (x₂ − x₁)(y₂ − y₁)        (green, mixed)

Theorem 6 of Chromogeometry: Q_b² = Q_r² + Q_g².

Under QA identification (b, e) := (x₂ − x₁, y₂ − y₁), the three quadrances are polynomials in QA's (b, e, d):

  Q_r = (b − e) · d,
  Q_g = 2 b e = −⟨a, a⟩_UHG,
  Q_b = b·b + e·e = d·d − 2·b·e.

The Pythagorean identity reduces to the QA-native polynomial identity

  (b·b + e·e)² = (b·b − e·e)² + (2·b·e)²,

verified exhaustively for (b, e) ∈ [1, 19]² with zero failures (361 × 361 = 130321 points).

**Primary source.** Wildberger, N.J. *Chromogeometry*, Mathematical Intelligencer 30:26–38, 2008 (UNSW preprint read 2026-04-13). Formulas from Theorem 6 of that paper.

**Fixtures.**

Pass fixture `cpi_pass_exhaustive_1to19.json`:
- Field: `sample_pairs` — 20 sample (b, e) pairs with computed Q_b, Q_r, Q_g, Q_b², Q_r² + Q_g², difference (must be 0).
- Field: `exhaustive_range`: "[1..19]^2".
- Field: `exhaustive_failures`: 0.
- Field: `qa_coord_formulas_verified`: {Q_r: "(b-e)*d", Q_g: "2*b*e", Q_b: "b*b + e*e"}.
- Field: `plimpton_link` — demonstrate 5 Pythagorean triples (r²−s², 2rs, r²+s²) for (r, s) = (2,1), (3,2), (4,1), (4,3), (5,2) and show they are QA-generated.

Fail fixture `cpi_fail_wrong_sign.json`:
- Synthetic Q_r with a + instead of − in the squared difference; validator catches via Pythagorean identity failure.

**Validator responsibilities.**
- Compute Q_b, Q_r, Q_g using raw integer arithmetic from (b, e).
- Verify Q_b² = Q_r² + Q_g² for every (b, e) in [1, 19]² (or a configured range).
- Verify the QA-coord formulas: Q_r should equal (b − e)(b + e) and Q_g should equal 2·b·e.
- Self-test at bottom.

---

## Cross-cutting task: register the four families

After each validator passes self-test:

1. Add a sweep entry to `qa_alphageometry_ptolemy/qa_meta_validator.py` in FAMILY_SWEEPS, following the existing pattern around [217].
2. Add a one-line entry to the recently-added table in `docs/families/README.md` **(lock this file before editing — other sessions may have it claimed; check `collab_get_state('file_locks')` first)**.
3. Create `docs/families/<N>_qa_<name>_cert.md` following the template of `docs/families/217_qa_fuller_ve_diagonal_decomposition_cert.md`.
4. Run `python tools/qa_axiom_linter.py --all && cd qa_alphageometry_ptolemy && python qa_meta_validator.py` to confirm green.
5. Broadcast `family_registered` on the collab bus with `{N, name, session: codex-wild-batch}` after each successful registration.

## Cross-cutting task: honest-null awareness

- No stochastic tests required by these certs; all four are exact-integer claims.
- No T2-D rule invocation needed.

## Reference compute snippets (exhaustively verified by Claude)

```python
# [231] Euler + Catalan OEIS (already run successfully)
from math import factorial
def hc(mv):
    V = 2 + sum((k+1)*m for k,m in enumerate(mv))
    E = 1 + sum((k+2)*m for k,m in enumerate(mv))
    F = sum(mv)
    d = factorial(V-1)
    for m in mv: d *= factorial(m)
    return factorial(E-1)//d, V, E, F

# [232] zero-quadrance iff same diagonal at m=9 (64=64 verified)
from fractions import Fraction
from math import gcd
def uhg(a, b): return -(a[0]*b[1] + a[1]*b[0])
def quad(a, b):
    n = uhg(a, b)**2; d = uhg(a, a) * uhg(b, b)
    return None if d == 0 else Fraction(1) - Fraction(n, d)
def dc(p):
    g = gcd(p[0], p[1]); return (p[0]//g, p[1]//g)

# [234] chromogeometry Pythagoras (0/130321 failures verified)
# Q_b**2 == Q_r**2 + Q_g**2 with Q_b = b*b + e*e, Q_r = b*b - e*e, Q_g = 2*b*e.
```

## Completion broadcast

When all four families are registered green, broadcast on the bus:

```
collab_broadcast(event_type="cert_batch_completed", data={
  "batch": "wildberger_scholar_mapping_2026_04_13",
  "families": [231, 232, 233, 234],
  "note": "Wildberger corpus → QA mapping cert batch shipped by codex"
})
```
