# Nonconvex Extension Roadmap: Łojasiewicz Exponent Theorem Program

**Date:** 2026-03-09
**Status:** Stage A — theorem map. §8.3 of companion paper is the upstream reference.
**Goal:** Extend the Finite-Orbit Descent Theorem to Łojasiewicz-conditioned losses (exponent α ∈ (0,1)).

---

## 1. Current Theorem Status

| Section | Loss class | Result | Status |
|---|---|---|---|
| §8.1 | Scalar quadratic (β=μ) | L_{t+L} = ρ(O)·L_t exactly | Proved |
| §8.2 | β-smooth, μ-PL (α=1) | L(w_{t+L})−L* ≤ ρ_PL(O)·(L(w_t)−L*) | Proved |
| §8.3 | β-smooth, Łojasiewicz α∈(0,1) | V_{t+L} ≤ ρ_α(O)·V_t^α, V_t = L(w_t)−L* | **Open conjecture** |

The PL theorem (§8.2) is the α=1 limit. The remaining open problem is the **sublinear-exponent case** α<1, which covers the bulk of nonconvex machine learning losses in practice (neural networks, sparse coding, matrix factorisation).

---

## 2. The Three Obstructions (from §8.3)

### O1 — α is not computable from the QA orbit
The Łojasiewicz exponent α depends on the loss geometry, not on the orbit. It cannot be read off from κ_t values. The cert would have to take α as a verified input.

### O2 — Flat-region avoidance is not intrinsic to QA
The no-flat-regions condition `||∇L(w_t)|| > 0 for all t in O` is a property of the trajectory, not the orbit alone. A loss could have a saddle point that the QA sequence passes through exactly.

### O3 — The α<1 per-step bound does not telescope multiplicatively (primary obstruction)
In the PL case (α=1), the per-step bound is:

```
V_{t+1} ≤ (1 − 2μ η_t (1 − β/2 η_t)) · V_t
```

This is multiplicative: composing L steps gives ρ_PL = ∏(1−c_t), a product of constants, with no dependence on V_t. The proof of §8.2 uses this.

For α<1, the Łojasiewicz descent lemma gives:

```
V_{t+1} ≤ V_t − c_t · V_t^α
       = V_t · (1 − c_t · V_t^{α−1})
```

The contraction factor `1 − c_t · V_t^{α−1}` **depends on V_t**. It cannot be pre-computed from the orbit alone. Composing L steps gives a factor that depends on the entire trajectory of V values, not just the orbit κ-sequence. This breaks the orbit-computable certificate property that makes §§8.1–8.2 useful.

---

## 3. Candidate Approaches

### 3A — Change of variable: track V^{1-α} instead of V (recommended first attempt)

Standard Łojasiewicz convergence proofs define φ_t = V_t^{1-α}. Then by concavity of x^{1-α}:

```
φ_{t+1} = V_{t+1}^{1-α} ≤ (V_t − c_t V_t^α)^{1-α}
```

Using the inequality (a−b)^{1-α} ≤ a^{1-α} − (1−α)·b·a^{-α} (tangent bound on concave function):

```
φ_{t+1} ≤ V_t^{1-α} − (1−α) c_t   =   φ_t − (1−α) c_t
```

This is **additive decay in φ**: φ decreases by at least (1−α)c_t per step, independent of V_t. Composing over the orbit window:

```
φ_{t+L} ≤ φ_t − (1−α) · Σ_{s=0}^{L-1} c_s   =   φ_t − (1−α) · C(O)
```

where `C(O) = Σ c_s = Σ 2μ η_s(1 − β/2 η_s)` is **computable from the orbit** (it depends only on η_eff^(t) values, which are orbit-determined).

This gives the orbit-window theorem in φ-coordinates:

**Candidate Theorem (φ-contraction):**
*Let L be β-smooth and satisfy the Łojasiewicz condition with exponent α ∈ (0,1) and constant μ > 0. Let O be a QA cosmos orbit with C(O) > 0. Define φ_t = (L(w_t) − L*)^{1-α}. Then:*

```
φ_{t+L} ≤ φ_t − (1−α)·C(O)
```

*Consequently, L(w_t) → L* in at most φ_0 / [(1−α)·C(O)] orbit windows.*

**Why this is weaker than §8.2:** The φ-contraction is additive, not multiplicative. It gives convergence in O(1/ε^{1/α}) steps (slower than geometric), matching the known Łojasiewicz rates. The orbit contribution C(O) replaces the multiplicative ρ_PL.

**What addresses O3:** φ-coordinates remove the V_t dependence from the contraction factor. The per-step φ-decrement (1−α)c_t is computable from the orbit alone.

**What remains for O1:** α must still be provided as a verified input. But this is the same status as μ and β in §8.2, which are also required inputs. This is acceptable.

**What remains for O2:** The flat-region avoidance condition is needed to ensure ||∇L(w_t)||² > 0 for all t in O. See §3B.

---

### 3B — Semi-algebraic structure addresses O2

If L is **semi-algebraic** (e.g., polynomial, rational, or a composition of polynomials with absolute values/ReLU), then:

1. **Łojasiewicz gradient inequality holds automatically** on any compact sublevel set, with rational exponent α ≥ 1/d where d is related to the degree.
2. **Flat regions are isolated:** by the Łojasiewicz inequality, if L is semi-algebraic and w is not a critical point of L, then ||∇L(w)|| > 0. Along a semi-algebraic trajectory (which QA orbits are, since T is linear), the set of times where ||∇L(w_t)|| = 0 is finite.

**QA orbit trajectories are semi-algebraic:** The map w_t = (I − η_eff^(t)·∇²L)^t w_0 for quadratic L; for general smooth L, the trajectory is the composition of a linear map (the QA update) with L's gradient. If L is a polynomial, the gradient is polynomial, and the trajectory satisfies a polynomial recurrence — hence is semi-algebraic.

Under semi-algebraic L, O2 reduces to: "the orbit does not pass through a critical point of L." This is generically true (measure-zero failure set) and can potentially be verified by interval arithmetic along the orbit for specific L.

---

### 3C — Orbit-window Lyapunov candidate (alternative)

The paper (§8.3) proposes the orbit-window Lyapunov candidate:

```
W_T = max_{t ∈ [T·L, (T+1)·L]} V_t
```

Showing W_{T+1} ≤ γ·W_T^α for some γ < 1 would prove the conjecture directly. This requires bounding the intra-orbit trajectory of V relative to its maximum — which depends on the orbit structure and β-smoothness more strongly than the φ-approach.

The φ-approach (3A) is more tractable because it gives a clean additive bound that decouples V_t from the orbit constants.

---

## 4. Candidate Theorem (Synthesis)

Combining 3A + 3B, the most tractable extension is:

**Theorem (Finite-Orbit Descent, Łojasiewicz Loss) — candidate statement:**
*Let L: ℝ^d → ℝ be β-smooth and semi-algebraic, satisfying the generalised Łojasiewicz condition:*
```
‖∇L(w)‖² ≥ 2μ (L(w) − L*)^α,   α ∈ (0,1], μ > 0
```
*on a sublevel set containing the orbit trajectory. Let O be a QA cosmos orbit with η_eff^(t) < 2/β for all t, and suppose L has no critical points along the orbit trajectory. Define:*
```
C(O, μ, β) := Σ_{t=0}^{L-1} 2μ η_eff^(t) (1 − β/2 · η_eff^(t))
```
*(a computable quantity depending only on O). Then for V_t = L(w_t) − L*:*

**(i) φ-additive contraction:** φ_{t+L} ≤ φ_t − (1−α)·C(O) where φ_t = V_t^{1-α}

**(ii) Convergence rate:** L(w_t) → L* in at most ⌈φ_0 / [(1−α)·C(O)]⌉ orbit windows.

**(iii) Reduction to §8.2 when α=1:** In the limit α→1, φ_t = V_t, the additive bound
φ_{t+L} ≤ φ_t − C(O) ≡ V_{t+L} ≤ (1 − C(O)/V_t)·V_t matches ρ_PL·V_t when V_t ≈ 1. (Note: the limiting form is not identical to §8.2 because §8.2 gives multiplicative contraction. The φ-approach gives the weaker additive bound even at α=1.)

---

## 5. Proof Sketch for (i)

**Step 1.** β-smooth descent: L(w_{t+1}) ≤ L(w_t) − η_t(1−β/2·η_t)·‖∇L(w_t)‖²

**Step 2.** Łojasiewicz: ‖∇L(w_t)‖² ≥ 2μ V_t^α

**Step 3.** Combine: V_{t+1} ≤ V_t − c_t·V_t^α where c_t = 2μ η_t(1−β/2·η_t) > 0

**Step 4.** Change of variable, φ_t = V_t^{1-α}:
```
φ_{t+1} = V_{t+1}^{1-α} ≤ (V_t − c_t V_t^α)^{1-α}
```
By concavity of x^{1-α} (for α ∈ (0,1)):
```
(V_t − c_t V_t^α)^{1-α} ≤ V_t^{1-α} − (1−α)c_t V_t^{-α} · V_t^α
                         = φ_t − (1−α)c_t
```
(using f(a−b) ≤ f(a) + f'(a)·(−b) for concave f at a = V_t, b = c_t V_t^α ≥ 0)

**Step 5.** Telescope over L steps:
```
φ_{t+L} ≤ φ_t − (1−α) Σ_{s=0}^{L-1} c_s = φ_t − (1−α)·C(O)
```

**Open gap (O2 residual):** Step 4 requires V_t > 0 at all intermediate steps (otherwise φ_t is not differentiable). This is guaranteed if L has no critical points along the trajectory — the semi-algebraic condition gives this generically.

**Status of the sketch:** Steps 1–3 are standard. Step 4 is a standard concavity argument. Step 5 is clean. The only open technical point is rigorously establishing O2 under the semi-algebraic hypothesis.

---

## 6. What This Changes for Certificates

### Current cert structure (§8.2 regime)

- Witness fields: lr, gain, H_QA, β, μ (with β=μ for quadratic)
- Computable orbit quantity: ρ_PL(O) = ∏(1 − 2μ η_t (1 − β/2 η_t))
- Certificate claim: "V_{t+L} ≤ ρ_PL · V_t"

### New cert structure (Łojasiewicz extension)

- Additional witness field: **α** (Łojasiewicz exponent, verified lower bound)
- Computable orbit quantity: **C(O)** = Σ c_t (sum instead of product — simpler to compute)
- Certificate claim: "φ_{t+L} ≤ φ_t − (1−α)·C(O), i.e., V^{1-α} decreases by (1−α)·C(O) per orbit"
- **κ_min remains necessary** (ensures c_t > 0 for all t, hence C(O) > 0)
- New cert gate: **α > 0** must be verified (e.g., by SOS certificate or interval bound on ‖∇L‖²/V^α)
- The multiplicative ρ_PL is **replaced** by the additive decrement (1−α)·C(O). The cert is structurally different but still orbit-computable.

### Impact on existing families

No existing cert family ([89]–[101]) changes. The new theorem would motivate a new family, tentatively [102], specifically for Łojasiewicz-loss convergence certification.

---

## 7. Priority Order for Proof Attempts

| Stage | Task | Difficulty | Blocks |
|---|---|---|---|
| B1 | Formalise the concavity argument (Step 4) rigorously | Low | Nothing |
| B2 | State and prove "semi-algebraic L ⇒ no QA-orbit flat regions generically" | Medium | B1 |
| B3 | Close the φ-contraction theorem under the semi-algebraic hypothesis | Medium | B1, B2 |
| B4 | Derive α from the SOS certificate for specific polynomial loss classes | High | B3 |
| C1 | Specify the [102] cert family schema (α, C(O), φ-contraction claim) | Low (after B3) | B3 |

**Minimum viable result (B1+B3, skipping B2):** State the theorem with "no critical points along orbit" as an explicit hypothesis, not derived from semi-algebraic structure. This gives a clean theorem with one unverified hypothesis. It is already stronger than the current §8.3 conjecture (which has no proof sketch at all).

---

## 8. The Minimum Additional Hypothesis

Paraphrasing ChatGPT's framing: *what is the minimum additional hypothesis beyond §8.2 that buys a true nonconvex extension?*

**Answer:** The minimum hypothesis is:

> L is β-smooth and satisfies the Łojasiewicz condition with exponent α ∈ (0,1) and constant μ > 0 on the orbit trajectory, with no critical points along the trajectory (‖∇L(w_t)‖ > 0 for all t ∈ O).

Under this hypothesis, the φ-contraction theorem follows by the proof sketch in §5. The hypothesis is:
- **Strictly weaker than PL:** PL (α=1) guarantees no saddle points with non-zero gradient; α<1 requires the same but along the orbit only.
- **Not verifiable from κ alone:** requires knowledge of α. The cert must take α as input.
- **Verifiable for polynomial losses:** α is computable (rational) for semi-algebraic L; the no-critical-points condition holds generically.

---

*Next session: attempt B1 (formalise the concavity argument) and check whether the tangent bound at Step 4 holds with equality constraints that could tighten the orbit-sum formula.*
