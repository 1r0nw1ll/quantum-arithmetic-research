# QA ↔ Wildberger–Rubine Hyper-Catalan / Geode Correspondence

**Status:** structural note, draft 2026-04-13
**Primary source:** Wildberger & Rubine, *A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode*, Amer. Math. Monthly 132(5):383–402 (Apr 2025).
**Related:** `QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md`, cert [217] Fuller VE, `QA_Q_SQRT_5_DIAGONAL_SYNTHESIS.md`.

---

## 1. Hyper-Catalan numbers — Wildberger & Rubine definition

For a multi-index **m** = [m_2, m_3, m_4, …] of non-negative integers:

- **V_m** = 2 + Σ_{k≥2} (k−1)·m_k      (vertex count)
- **E_m** = 1 + Σ_{k≥2} k·m_k            (edge count)
- **F_m** = Σ_{k≥2} m_k                   (face count)

Hyper-Catalan coefficient:

  **C_m = (E_m − 1)! / ((V_m − 1)! · m!)**      where m! = Π m_k!

Generating identity: the series α = Σ_m C_m · t^m (with t^m = Π t_k^{m_k}) is the unique formal power-series zero of the **general geometric polynomial**

  1 − α + Σ_{k≥2} t_k · α^k = 0.

Specializations:
- Only t_2 = t: C_n = (2n)! / (n!(n+1)!) — classical **Catalan**.
- Only t_3: C_n = (3n)! / (n!(2n+1)!) — **Fuss–Catalan**.

Combinatorial meaning: C_m counts subdivisions of a convex polygon into m_2 triangles, m_3 quadrilaterals, m_4 pentagons, etc.

The **Geode** G is the factor appearing when the hyper-Catalan series is factored; its coefficients satisfy recurrences that were the subject of Wildberger's three conjectures (now proved by Amdeberhan–Zeilberger 2025, arXiv:2506.17862).

## 2. QA identification (main result of this note)

**Claim.** Under the assignment

  **b := V_m − 1,  e := F_m**

the QA derived coordinates are:

  **d = b + e = V_m + F_m − 1 = E_m**              (edge count)
  **a = b + 2e = V_m + 2F_m − 1 = E_m + F_m = d + F_m**

**Euler's polygon identity** V − E + F = 1 becomes automatic:

  V − E + F = (b+1) − (b+e) + e = 1.   ✓

Hyper-Catalan multi-indices are therefore in canonical bijection with QA tuples (b, e, d, a) where (b, e) arises from a polygon-subdivision count.

## 3. Single-type subdivisions sit on QA sibling diagonals

For m with only m_k = n nonzero:

  V = 2 + (k−1)n,  F = n   ⇒   b = 1 + (k−1)n,  e = n.

Hence the QA constraint

  **b = (k−1)·e + 1**

This is a **sibling diagonal** in the sense of [217] Fuller VE (`QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md` §4): each k ≥ 2 picks out its own line in QA (b, e) space.

| k (polygon type) | QA diagonal class         | Count sequence (n = 0,1,2,3,…)    | Name |
|------------------|---------------------------|------------------------------------|------|
| 2 (triangles)    | b = e + 1 (near-D_1)      | 1, 1, 2, 5, 14, 42, 132, …         | Catalan |
| 3 (quadrilaterals)| b = 2e + 1               | 1, 1, 3, 12, 55, 273, 1428, …      | Fuss–Catalan (3,1) |
| 4 (pentagons)    | b = 3e + 1                | 1, 1, 4, 22, 140, 969, …           | Fuss–Catalan (4,1) |
| k general        | b = (k−1)e + 1            | (kn)! / (n!((k−1)n+1)!)            | Fuss–Catalan (k,1) |

Mixed subdivisions (several m_k > 0) sit off these single-type lines but still on the affine surface V − E + F = 1, which in QA coordinates is the identity b + 1 − d + e ≡ 0 (trivially).

## 4. Relation to existing QA diagonal structures

- **Sierpinski D_1 (b = e):** corresponds to V = F + 1, i.e. V_m − F_m = 1, i.e. Σ(k−1)m_k = F_m − 1 = Σ m_k − 1. For single-type m with only m_k: (k−1)n = n − 1, which has no non-negative integer solution for k ≥ 2 (requires k = (n−1)/n + 1 < 2). D_1 is therefore **not** reachable by single-type hyper-Catalans; only by genuinely mixed indices satisfying Σ(k−2)m_k = −1 — impossible, so D_1 is disjoint from hyper-Catalan (b, e) locus. Cross-check: Catalan lies on b = e + 1, never on b = e.

- **Fuller VE [217]:** S_n = 10n² + 2 sits on D_1 for n ≢ 0 (mod 3). Hyper-Catalan locus is disjoint from D_1, so no overlap. These are **complementary** diagonal classes — neither subsumes the other.

- **Fibonacci resonance [219] / Haramein [218]:** Q(√5) class. Hyper-Catalans are rational integers with no √5 signature in their closed form — **different algebraic class**. Candidate Q(1)-rational-integer class (pure combinatorial, no quadratic extension).

## 5. The Geode factor — primary-source definition (AMM Theorem 12)

From Wildberger & Rubine 2025, Section 11.3 "Face layering of S and the Geode factorization" — verbatim:

> **Theorem 12** (Subdigon polyseries factorization). *There is a unique polyseries G for which we have the identity S − 1 = S_1 · G.*

Here S = S[t_2, t_3, t_4, …] is the subdigon polyseries (the generating series of all subdigons); S_1 ≡ t_2 + t_3 + t_4 + … is the sum of single-face subdigons (one t_k per face type); G is the **Geode**, a multi-indexed polyseries uniquely determined by the identity. S_1 is not a power series with nonzero constant term, so invertibility is nontrivial — this makes Theorem 12 a substantive combinatorial existence statement.

**QA interpretation.** The Geode is a multiplicative "residue" in the formal-series ring: every non-trivial subdigon is decomposed as (choice of face) × (rest), with G tracking the "rest". In QA coordinates, S_1 corresponds to multi-indices with F_m = 1 — the minimal-face QA tuples (b, e) = (V_m − 1, 1) for a single-face-type subdigon. Concretely:

| t_k | V_m | (b, e) in QA | Name |
|-----|-----|--------------|------|
| t_2 (1 triangle)     | 3 | (2, 1) | Satellite-ish entry point |
| t_3 (1 quadrilateral)| 4 | (3, 1) | |
| t_4 (1 pentagon)     | 5 | (4, 1) | |
| t_k (1 k-gon)        | k+1 | (k, 1) | e = 1 line |

So S_1 lives on the QA line **e = 1**, which is the *first sibling diagonal* of D_1 (b = 0·e + c for c ≥ 2). The Geode factor G carries the hyper-Catalan content above this e = 1 boundary.

**Status.** The "Geode is observer-layer residue" framing from an earlier draft was imprecise. The actual content of Theorem 12 is a *multiplicative* factorization S − 1 = S_1 · G; the Geode is not a pointwise observable but a generating series. The QA-side meaning is: the e = 1 line acts as a multiplicative unit, and every other hyper-Catalan point is expressible as (e=1 point) × (G-coefficient).

## 6. Cert candidacy

Proposed cert family: **[231] QA_HYPER_CATALAN_DIAGONAL_CORRESPONDENCE_CERT.v1**

Witnesses (all verifiable on integer arithmetic, no mod reduction):
1. Euler identity V − E + F = 1 equivalent to QA derivation rule d = b + e under (b, e) = (V−1, F).
2. Classical Catalan C_n on QA diagonal b = e + 1 — verified for n = 0..10 against OEIS A000108.
3. Fuss–Catalan (k,1) on QA diagonal b = (k−1)e + 1 for k = 3, 4, 5 — verified against OEIS A001764, A002293, A002294.
4. D_1 disjointness: proof that no single-type hyper-Catalan lies on b = e.
5. Fuller VE disjointness: proof that 10n² + 2 is not expressible as E_m under any hyper-Catalan m (E_m is linear in the m_k, so E_m = 10n²+2 has many solutions — the claim is about the *coincidence class* at fixed (b, e), not mere E-value reachability; needs sharpening before certification).

**Open before cert draft:** verify the proof in item 5 (the disjointness is about which (b, e) pairs are hit, not which E-values). Acquire the full AMM paper PDF to confirm the Geode definition is being used correctly.

## 7. Empirical verification (2026-04-13)

All four structural claims checked via exact integer arithmetic:

1. **Euler automatic from QA derivation.** For m ∈ {[3,0,0,0], [0,2,0,0], [2,1,0,0], [1,1,1,0], [0,0,0,5]}: every tuple gives V − E + F = 1 and QA d = E_m exactly. ✓
2. **Classical Catalan on b = e + 1.** C_n for n = 0..10 matches OEIS A000108 exactly (1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796); all points satisfy b = e + 1. ✓
3. **Fuss–Catalan on b = (k−1)e + 1.** k = 3 against OEIS A001764 (9 terms), k = 4 against A002293 (7 terms), k = 5 against A002294 (6 terms) — all C_n match and all points satisfy the predicted sibling-diagonal relation. ✓
4. **D_1 disjointness.** Exhaustive sweep k = 2..7, n = 0..9: zero single-type hyper-Catalan points land on b = e. ✓

Cert scaffold is empirically ready. Next physical step (requires Codex, per AGENTS.md python-write policy): write validator + fixtures for cert family [231] QA_HYPER_CATALAN_DIAGONAL_CORRESPONDENCE_CERT.v1.

## 8. Remaining open items

- [ ] Acquire AMM 2025 paper PDF to confirm Geode factor definition matches §5 observer-residue interpretation.
- [ ] Extended null test: generate random integer sequences with matching V/E/F growth and check they do NOT satisfy Euler identity unless by construction (should be trivially true, included for completeness).
- [ ] Decide cert [231] scope: Euler-skeleton-only, or include Geode coefficient as observer-layer witness.

## 9. References

- Wildberger, N.J.; Rubine, D. *A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode*. Amer. Math. Monthly 132(5):383–402, 2025. **Open access**; primary source for Theorems 5 (C_m = (E−1)!/((V−1)! m!)), 6 (geometric polynomial series solution), 7 (general polynomial formula), 12 (Geode factorization S − 1 = S_1 G).
- Amdeberhan, T.; Zeilberger, D. *Proofs of Three Geode Conjectures*. arXiv:2506.17862, 2025.
- Wildberger, N.J. *Hyper-Catalan and Geode Recurrences and Three Conjectures of Wildberger*. arXiv:2507.04552, 2025.
- OEIS A000108 (Catalan), A001764 (Fuss k=3), A002293 (k=4), A002294 (k=5).
