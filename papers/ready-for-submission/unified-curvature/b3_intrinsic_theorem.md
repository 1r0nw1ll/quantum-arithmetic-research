# B3: Intrinsic Łojasiewicz Orbit Descent — Eliminating H-crit

**Status:** Proved. Closes the open problem stated in B2a/§8.3.
**Date:** 2026-03-09
**Depends on:** B1 (b1_phi_lemma.md), B2a (b2_orbit_theorem.md)

---

## The Open Problem (from B2a)

B2a required hypothesis (H-crit): ∇L(w_s) ≠ 0 for all s = t,...,t+L−1.
§8.3 (paper) stated: *"A fully intrinsic version — depending only on the orbit O
and loss class, with no condition on w_0 — remains open."*

This note proves that H-crit is **not an independent hypothesis**: it follows from the
Łojasiewicz condition and the fixed-point property of gradient descent at minimizers.
The open problem is resolved.

---

## Two Supporting Lemmas

### Lemma 1 (Łojasiewicz → H-crit at non-zero V)

*If V_s > 0 and (H-Łoj) holds at w_s, then ∇L(w_s) ≠ 0.*

**Proof.** By (H-Łoj): ‖∇L(w_s)‖² ≥ 2μ V_s^α. Since μ > 0, V_s > 0, α > 0:
V_s^α > 0, so ‖∇L(w_s)‖² > 0, hence ∇L(w_s) ≠ 0. ∎

*Note: This is the easy direction. The non-trivial content is Lemma 2.*

---

### Lemma 2 (Fixed-Point Propagation)

*Under gradient descent w_{s+1} = w_s − η_eff^(s) ∇L(w_s) and (H-smooth):*
*If V_s = 0 for some s, then V_{s+k} = 0 for all k ≥ 0.*

**Proof.** Case (i): L* is not attained. Then L(w) > L* for all w ∈ ℝ^d,
so V_s = L(w_s) − L* > 0 for all w_s, making V_s = 0 impossible.
The conclusion holds vacuously.

Case (ii): L* is attained. V_s = L(w_s) − L* = 0 means L(w_s) = L*, so w_s is
a global minimizer of L. For β-smooth L, global minimizers satisfy the first-order
necessary condition ∇L(w_s) = 0. Therefore:

```
w_{s+1} = w_s − η_eff^(s) · ∇L(w_s) = w_s − η_eff^(s) · 0 = w_s.
```

By induction: w_{s+k} = w_s for all k ≥ 0, so V_{s+k} = V_s = 0 for all k ≥ 0. ∎

*Scope note: this fixed-point argument applies to **exact** gradient descent
(∇L, not a stochastic approximation). For SGD, the stochastic gradient at w*
may be non-zero, so the fixed-point property is not guaranteed. The theorem's
update rule uses exact ∇L throughout.*

---

## The Intrinsic Theorem

**Theorem (Finite-Orbit Descent, Łojasiewicz — Intrinsic Form).**
*Let L: ℝ^d → ℝ be β-smooth and satisfy the Łojasiewicz condition on*
*S₀ = {w : L(w) ≤ L(w_t)}:*

```
‖∇L(w)‖² ≥ 2μ (L(w) − L*)^α    for all w ∈ S₀,   α ∈ (0,1), μ > 0.
```

*Let O be a QA cosmos orbit with η_eff^(s) ∈ (0, 2/β) for all s = 0,...,L−1. Define*

```
c_s  = 2μ η_eff^(s) (1 − β/2 · η_eff^(s)) > 0
C(O) = Σ_{s=0}^{L-1} c_s > 0
φ_t  = V_t^{1−α}   (V_t = L(w_t) − L*)
```

*Suppose V_t > 0. Then either:*

**(A)** V_{t+L} = 0 (exact convergence at the orbit endpoint), or

**(B)** φ_{t+L} ≤ φ_t − (1−α) · C(O).

**Hypotheses required:** (H-smooth), (H-Łoj), (H-orbit), and V_t > 0.
**H-crit is not required.**

---

## Proof

**Step 1: Case split on V_{t+L}.**

- **Case (A): V_{t+L} = 0.** Conclusion (A). Done. ∎

- **Case (B): V_{t+L} > 0.** We prove the φ-bound.

**Step 2: V_s > 0 for all s ∈ {t,...,t+L−1} (in Case B).**

Suppose for contradiction that V_s = 0 for some s ∈ {t+1,...,t+L−1}.
By Lemma 2, V_{s+k} = 0 for all k ≥ 0. In particular V_{t+L} = 0, contradicting
V_{t+L} > 0 (Case B assumption). Hence V_s > 0 for all s ∈ {t+1,...,t+L−1}.
Combined with V_t > 0 (hypothesis): V_s > 0 for all s ∈ {t,...,t+L−1}. ∎

**Step 3: H-crit holds on {t,...,t+L−1} (derived, not assumed).**

For each s ∈ {t,...,t+L−1}: V_s > 0 (Step 2), so by Lemma 1 (with Łojasiewicz):
∇L(w_s) ≠ 0. H-crit holds as a consequence. ✓

**Step 4: Apply B1 at each step s ∈ {t,...,t+L−1}.**

*B1 prerequisites at step s:*
- V_s > 0 (Step 2) ✓
- V_{s+1} ≥ 0 (L ≥ L* by definition) ✓
- V_{s+1} ≤ V_s − c_s V_s^α (from β-smooth descent + Łojasiewicz, Step 3 of B2a) ✓

*B1 Case split:*
- B1 Case 2 (overshoot → V_{s+1} = 0): for s ∈ {t,...,t+L−2}, V_{s+1} > 0 by Step 2;
  for s = t+L−1, V_{t+L} > 0 by Case (B) assumption. Both contradict Case 2. Hence:
- **B1 Case 1 holds at every step:** φ_{s+1} ≤ φ_s − (1−α) c_s.

**Step 5: Telescope.**

```
φ_{t+L} ≤ φ_{t+L−1} − (1−α)c_{t+L−1}
         ≤ ...
         ≤ φ_t − (1−α) Σ_{s=t}^{t+L−1} c_s
         = φ_t − (1−α) C(O).
```

Conclusion (B). ∎

---

## What changed compared to B2a

| | B2a | B3 (intrinsic) |
|---|---|---|
| Hypotheses | H-smooth, H-Łoj, H-orbit, **H-crit** | H-smooth, H-Łoj, H-orbit, **V_t > 0** |
| H-crit status | Explicit hypothesis | **Derived** (Lemma 1 + 2) in Case B |
| Proof structure | H-crit → V_s > 0 → Case split on V_{t+L} | Case split on V_{t+L} → V_s > 0 → H-crit |
| Dependency | H-crit assumed upfront | V_s > 0 proved from Case B + fixed-point |

The proof structure inverts: instead of deriving V_s > 0 from H-crit (B2a), we derive
H-crit from V_s > 0 (which itself follows from Case B + Lemma 2). The logical content
is the same; the dependency order is different.

---

## The key move: fixed-point propagation

The core observation (Lemma 2): **gradient descent is deterministic and fixes at
minimizers**. Once V_s = 0 (the trajectory reaches a global minimizer), it never
leaves. Therefore: if V_{t+L} > 0 (Case B), no minimizer was reached at any
s ∈ {t,...,t+L−1}. This eliminates H-crit as a separate assumption.

This argument uses only:
- Smoothness of L (for ∇L(w*) = 0 at a minimizer — first-order necessary condition)
- Determinism of gradient descent (w_{s+1} depends only on w_s)
- Definition of V_s = L(w_s) − L*

No semi-algebraic structure, no Morse theory, no measure-zero arguments.

---

## What the open problem actually required

§8.3 stated the open problem as: *"depending only on the orbit O and loss class,
with no condition on w_0."* The condition "V_t > 0" (starting point not at the
minimizer) is not a trajectory condition — it is a trivial non-degeneracy condition:
if V_t = 0, the orbit has already converged and there is nothing to prove.

The genuine trajectory condition was H-crit (requiring ∇L(w_s) ≠ 0 for all s along
the orbit). This is the condition that has been eliminated. What remains is only
V_t > 0, which is a starting-point condition equivalent to "the orbit is not trivially
at the minimum" — and without it, no convergence theorem can say anything.

**Verdict:** The open problem is resolved. The intrinsic form (B3) is the correct
final theorem.

---

## Cert implication for [102]

Cert family [102] (QA_LOJASIEWICZ_ORBIT_CERT.v1) currently requires
`h_crit_witnessed: true` as an explicit boolean. Under B3, this field can be
**derived** from V_t > 0 (witnessed via φ_t > 0) and the fixed-point property.

**[102] v2 upgrade path:** Remove `h_crit_witnessed` as a required field; add
Gate E that derives H-crit from `phi_t > 0` (already a required field). This is
a backward-incompatible schema change — reserve for v2 of the family.

For now, [102] v1 with explicit `h_crit_witnessed` is sound and conservative.

---

## Convergence rate and orbit count

Same as B2a: under Case (B) repeatedly, L(w_t) → L* in at most
⌈φ_0 / [(1−α)C(O)]⌉ orbit windows. Proof is identical — only the final theorem
hypotheses changed.

---

## Summary

> B3 proves the intrinsic form of the Łojasiewicz orbit-window theorem: under
> β-smooth + Łojasiewicz + orbit feasibility + V_t > 0 (not at minimum), either
> exact convergence occurs within the orbit window, or φ_{t+L} ≤ φ_t − (1−α)C(O).
> H-crit is derived, not assumed. The open problem from §8.3 is closed.
