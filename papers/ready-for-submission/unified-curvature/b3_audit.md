# B3 Adversarial Audit

**Auditor role:** hostile reviewer. Attack every step. Accept nothing without verification.
**Date:** 2026-03-09
**Subject:** b3_intrinsic_theorem.md — claim that H-crit follows from V_t > 0 + Łojasiewicz

---

## Checkpoint 1: Does V_s = 0 imply "global minimizer"?

**Claim in B3:** V_s = L(w_s) − L* = 0 means L(w_s) = L*, so w_s is a global
minimizer of L, hence ∇L(w_s) = 0.

**Attack:** Does V_s = 0 actually imply ∇L(w_s) = 0?

**Analysis:**

Case (i) — L* not attained: V_s = L(w_s) − L* > 0 for all w_s (since
L(w) > L* = inf L for all w). So V_s = 0 is impossible. Vacuous. ✓

Case (ii) — L* attained by some w*: V_s = 0 means L(w_s) = L* = L(w*).
So w_s achieves the global infimum. For any β-smooth (hence C^1) function,
a necessary condition at a local minimum is ∇L = 0. A global minimizer is
a local minimizer (trivially: any open ball around w_s satisfies
L(w_s) ≤ L(w) for all w in the ball, since L(w_s) is the global minimum).
Therefore ∇L(w_s) = 0. ✓

**Verdict: PASS.** Both cases handled. The step is rigorous.

**Scope note:** The step "global minimizer → ∇L = 0" uses only C^1 regularity
(which follows from β-smoothness). No stronger condition needed.

---

## Checkpoint 2: Does the update rule fix minimizers for every admissible step?

**Claim in B3:** If ∇L(w_s) = 0, then w_{s+1} = w_s − η_eff^(s) · 0 = w_s.
By induction w_{s+k} = w_s for all k ≥ 0.

**Attack 1:** The step size η_eff^(s+k) varies with k. Does it matter?

No — the update is w_{s+k+1} = w_{s+k} − η_eff^(s+k) · ∇L(w_{s+k}).
If w_{s+k} = w_s (global min), then ∇L(w_{s+k}) = ∇L(w_s) = 0,
so w_{s+k+1} = w_{s+k} regardless of η_eff^(s+k). The step size multiplies 0. ✓

**Attack 2:** Is the gradient well-defined at w_s = w*?

β-smooth means L is C^1, so ∇L(w_s) is defined everywhere including w_s = w*. ✓

**Attack 3:** What about η_eff^(s+k) = 0 for some k? Could the orbit be degenerate?

(H-orbit) requires η_eff^(t) ∈ (0, 2/β) for all t — strictly positive. But even if
η_eff = 0 were allowed, the update would give w_{s+1} = w_s − 0 · ∇L(w_s) = w_s.
The fixed-point property holds regardless. ✓

**Verdict: PASS.** The induction is rigorous. Step sizes do not affect the conclusion.

---

## Checkpoint 3: Top-level case split phrased correctly?

**Claim in B3:** The theorem concludes with:
"either (A) V_{t+L} = 0, or (B) φ_{t+L} ≤ φ_t − (1−α)C(O)."

**Attack:** Is this an exhaustive and non-overlapping case split?

V_{t+L} ≥ 0 always (since L(w) ≥ L* by definition of L*). So V_{t+L} = 0 or V_{t+L} > 0.
- Case A: V_{t+L} = 0. Stated directly.
- Case B: V_{t+L} > 0. Proof derives φ-bound.
These are exhaustive and non-overlapping. ✓

**Attack:** Can case (A) and case (B) overlap — could V_{t+L} = 0 while the φ-bound also holds?

If V_{t+L} = 0, then φ_{t+L} = 0. The bound φ_{t+L} ≤ φ_t − (1−α)C(O) would require
0 ≤ φ_t − (1−α)C(O). This could hold or fail depending on φ_t. The theorem states
*either A or B*, not *A xor B*, so overlap is allowed. No issue. ✓

**Attack:** Does the proof actually use V_{t+L} > 0 in Case B, or does it sneak in V_s > 0 at other steps?

Step 2 of B3's proof establishes V_s > 0 for s ∈ {t+1,...,t+L−1} from the
assumption V_{t+L} > 0 (by Lemma 2 contrapositive). For s = t, V_t > 0 is the
theorem hypothesis. For s = t+L, V_{t+L} > 0 is the Case B assumption. So
V_s > 0 for all s ∈ {t,...,t+L}, established cleanly. ✓

**Verdict: PASS.** The case split is correct and exhaustive.

---

## Checkpoint 4: Scope — exact gradient descent only?

**Concern:** Lemma 2's fixed-point argument uses ∇L(w_s) = 0 at w_s = w*.
This is the EXACT gradient. For stochastic gradient descent (SGD) with update
w_{s+1} = w_s − η · ∇̃L(w_s, ξ_s), the stochastic gradient ∇̃L(w*, ξ_s) may
be non-zero even at the minimizer, so the trajectory would not fix.

**Assessment:** The theorem statement in B3 uses the exact gradient:
"Under gradient descent w_{s+1} = w_s − η_eff^(s) ∇L(w_s)." This is explicit.
The theorem is NOT claimed for SGD. The scope is correct and the limitation is
visible in the hypothesis. ✓

**Flagged for transparency:** B3 docs should explicitly note that the fixed-point
argument uses exact GD. Add one sentence to Lemma 2.

---

## Checkpoint 5: H-Łoj applies at all intermediate steps?

**Attack:** The Łojasiewicz condition (H-Łoj) is assumed on S₀ = {w : L(w) ≤ L(w_t)}.
Are the intermediate iterates w_s in S₀?

B2a Step 3 establishes: L(w_{s+1}) ≤ L(w_s) at each step (by β-smooth descent with
η_eff^(s) ∈ (0, 2/β)). By induction, L(w_s) ≤ L(w_t) for all s ≥ t, so w_s ∈ S₀. ✓

B3 references "from β-smooth descent + Łojasiewicz, Step 3 of B2a" in Step 4, which
covers this. The reference is valid. ✓

**Verdict: PASS.**

---

## Checkpoint 6: B1 application in B3 Step 4

**Attack:** B3 Step 4 applies B1 at each step s ∈ {t,...,t+L−1}. B1 requires
V_{s+1} ≤ V_s − c_s V_s^α (hypothesis H1 of B1). Where does this come from?

From β-smooth descent (Step 2 of B2a): L(w_{s+1}) ≤ L(w_s) − η_eff^(s)(1−β/2·η_eff^(s))‖∇L(w_s)‖².
From (H-Łoj) (Lemma 1 establishes ∇L(w_s) ≠ 0, hence ‖∇L(w_s)‖² ≥ 2μ V_s^α).
Combining: V_{s+1} ≤ V_s − c_s V_s^α with c_s = 2μ η_eff^(s)(1−β/2·η_eff^(s)) > 0. ✓

**Verdict: PASS.** The prerequisite for B1 is established by the same chain as in B2a.

---

## Checkpoint 7: Could B1 Case 1 give V_{s+1} = 0 (not V_{s+1} > 0)?

**Attack:** B1 Case 1 (no overshoot) gives φ_{s+1} ≤ φ_s − (1−α)c_s.
But it does NOT rule out V_{s+1} = 0 in Case 1 (Case 1 only says r_s ≥ 0,
so V_{s+1} ∈ [0, r_s]). If V_{s+1} = 0 (exact landing at minimizer) in Case 1,
this would be conclusion (A), not (B).

**Assessment:** In Case B (V_{t+L} > 0), Step 2 established V_{s+1} > 0 for all
s+1 ∈ {t+1,...,t+L}. So V_{s+1} = 0 cannot happen in Case B. B1 Case 2 is the
only mechanism giving V_{s+1} = 0 in B1, and it's already ruled out. B1 Case 1
does allow V_{s+1} = 0, but V_{s+1} > 0 (from Step 2) means this doesn't occur.

No contradiction or gap. The logic is: Step 2 proves V_s > 0 for all intermediate
s (before B1 is applied), so whatever B1 Case is invoked at each step, the outcome
must be consistent with V_{s+1} > 0. Only Case 1 is consistent with V_{s+1} > 0.
(Case 2 forces V_{s+1} = 0 directly, contradicting Step 2.) ✓

**Verdict: PASS.**

---

## Checkpoint 8: Is "V_t > 0 only" really sufficient (vs B2a needing H-crit)?

**Attack:** In B2a, H-crit was used to establish V_s > 0 for intermediate steps
s ∈ {t,...,t+L−1} BEFORE the case split on V_{t+L}. In B3, V_s > 0 is established
INSIDE Case B (after the split). Is this circular?

No. The dependency order in B3 is:
1. ASSUME V_t > 0 (hypothesis) — no circularity.
2. SPLIT on V_{t+L}: Case A (V_{t+L}=0) or Case B (V_{t+L}>0).
3. IN CASE B, DERIVE V_s > 0 for intermediate s by Lemma 2 + contrapositive.
4. IN CASE B, APPLY B1 using these derived V_s > 0 values.

The derivation of V_s > 0 (Step 3) uses only: the Case B assumption V_{t+L} > 0,
Lemma 2, and the gradient descent dynamics. It does NOT use H-crit. ✓

B3 is not circular. The key structural change from B2a is that Step 2 (establishing
V_s > 0) is moved INSIDE Case B, where V_{t+L} > 0 provides the needed handle.

**Verdict: PASS.**

---

## Overall audit result

| Checkpoint | Issue? | Verdict |
|---|---|---|
| 1. V_s=0 → global min → ∇L=0 | None | PASS |
| 2. Update fixes minimizers, step sizes irrelevant | None | PASS |
| 3. Case split exhaustive and correctly phrased | None | PASS |
| 4. Scope: exact GD only (not SGD) | Flag: add explicit note to Lemma 2 | PASS (scope correct, transparency improvable) |
| 5. H-Łoj applies throughout (sublevel set) | None | PASS (via B2a Step 3 reference) |
| 6. B1 prerequisites established | None | PASS |
| 7. B1 Case 1 can give V_{s+1}=0 | Handled by Step 2 inside Case B | PASS |
| 8. Dependency order not circular | None | PASS |

**Verdict: B3 is sound.** No logical gaps or hidden case distinctions found.

The one transparency flag (Checkpoint 4) is a documentation issue, not a mathematical gap. Add one sentence to Lemma 2: *"Note: this argument applies to exact gradient descent; for stochastic GD the stochastic gradient at w* may be non-zero, so the fixed-point property is not guaranteed."*

---

## Effect on the open problem

The open problem stated in B2a was:
> *"A fully intrinsic version — depending only on the orbit O and loss class,
> with no condition on w_0 — remains open."*

B3 provides the intrinsic version. The residual hypothesis is V_t > 0 (at the
starting point). This is not a "condition on w_0" in the sense of the open problem:
- The original open problem asked to eliminate H-crit (a trajectory condition:
  ∇L ≠ 0 along the entire orbit).
- V_t > 0 is a non-degeneracy condition at a single point: the orbit window starts
  from a non-minimizer. Without this, there is nothing to prove.

**The open problem is resolved.** H-crit is eliminated. V_t > 0 is the minimal
non-trivial hypothesis: it is equivalent to "the orbit window is not already done."

---

## One remaining caveat: H-crit is now redundant in [102] v1

Cert [102] v1 requires `h_crit_witnessed: true`. Under B3, this field is always
derivable from `phi_t > 0` (which [102] already requires as `exclusiveMinimum: 0`).
So [102] v1 is conservative: it requires a redundant field. It is sound; the cert
just asks the issuer to explicitly state something that is already guaranteed by
the schema. A [102] v2 (family [103]) can remove this field, making the cert match
the intrinsic theorem exactly.
