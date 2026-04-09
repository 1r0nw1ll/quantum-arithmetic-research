# B1: The One-Step φ-Decrement Lemma

**Status:** Proved with case split. Clean. Roadmap §3A confirmed viable.
**Date:** 2026-03-09

---

## Setup and Hypotheses

Let α ∈ (0,1), V_t > 0, c_t > 0. Suppose:

**(H1)** `V_{t+1} ≤ V_t − c_t · V_t^α`   (Łojasiewicz descent bound)

**(H2)** `V_{t+1} ≥ 0`   (loss excess is non-negative)

Define the change-of-variable: `φ_t = V_t^{1−α}`.

---

## Lemma (One-Step φ-Decrement)

Under (H1) and (H2), exactly one of the following holds:

**(A)** `V_{t+1} = 0` (exact convergence in one step), or

**(B)** `φ_{t+1} ≤ φ_t − (1−α) c_t`

---

## Proof

### Case split on whether the descent step overshoots

Define `r_t = V_t − c_t V_t^α = V_t(1 − c_t V_t^{α−1})`.

**Case 2 (overshoot): `r_t < 0`**

Then (H1) gives `V_{t+1} ≤ r_t < 0`. Combined with (H2), `V_{t+1} ≥ 0`, we get `V_{t+1} = 0`. This is conclusion (A). ∎ (Case 2)

**Case 1 (no overshoot): `r_t ≥ 0`**

We have `r_t = V_t − c_t V_t^α ≥ 0` and (H1) gives `V_{t+1} ≤ r_t`.

Since `f(x) = x^{1−α}` is non-decreasing on [0, ∞) (because 1−α > 0):
```
φ_{t+1} = V_{t+1}^{1−α} ≤ r_t^{1−α} = (V_t − c_t V_t^α)^{1−α}    ...(*)
```

It remains to show: `(V_t − c_t V_t^α)^{1−α} ≤ V_t^{1−α} − (1−α) c_t`.

### Applying the tangent inequality for concave functions

Let `f(x) = x^{1−α}` for x > 0.

**f is strictly concave:** `f''(x) = (1−α)(−α) x^{−α−1} = −α(1−α) x^{−(1+α)} < 0` for all x > 0. ✓

For any concave differentiable function on an open interval, the **tangent line is an upper bound**:
```
f(x) ≤ f(a) + f'(a)(x − a)    for all x ≥ 0, a > 0.
```
This holds because concavity means the function lies below every tangent line.

Apply at `a = V_t > 0` (satisfies a > 0 ✓) and `x = V_t − c_t V_t^α = r_t ≥ 0` (in domain ✓):
```
f(r_t) ≤ f(V_t) + f'(V_t) · (r_t − V_t)
```

Computing each term:
- `f(V_t) = V_t^{1−α} = φ_t`
- `f'(V_t) = (1−α) V_t^{−α}`
- `r_t − V_t = (V_t − c_t V_t^α) − V_t = −c_t V_t^α`

Substituting:
```
f(r_t) ≤ φ_t + (1−α) V_t^{−α} · (−c_t V_t^α)
        = φ_t − (1−α) c_t
```

Combining with (*):
```
φ_{t+1} ≤ r_t^{1−α} = f(r_t) ≤ φ_t − (1−α) c_t
```

This is conclusion (B). ∎ (Case 1)

---

## Domain conditions — all verified

| Condition | Requirement | Status |
|---|---|---|
| f concave on domain | f''(x) < 0 for x > 0, α ∈ (0,1) | ✓ proved above |
| Tangent inequality applies | a = V_t > 0 (interior point) | ✓ from V_t > 0 hypothesis |
| x = r_t in domain of f | r_t ≥ 0 (Case 1 hypothesis) | ✓ by case split |
| f non-decreasing (for step (*)) | 1−α > 0 ✓ | ✓ always holds for α < 1 |
| V_{t+1} ≥ 0 | (H2) | ✓ required hypothesis |

No hidden small-step assumption. No approximation. The only hypothesis beyond (H1) and (H2) is **V_t > 0**, which is the "no critical points along orbit" condition.

---

## What V_t = 0 means (edge case)

If V_t = 0 at some step t, then L(w_t) = L* (exact minimiser reached). The orbit-window theorem terminates. No step-decrement applies because there is nothing more to bound. This is consistent: the lemma requires V_t > 0.

---

## Orbit-window theorem (immediate corollary)

**Theorem (φ-orbit contraction):** Let O be a QA cosmos orbit of length L. Suppose V_t > 0 for all t ∈ {0,...,L−1} (no convergence during the orbit). Then:

```
φ_{t+L} ≤ φ_t − (1−α) · C(O)
```

where `C(O) = Σ_{s=0}^{L-1} c_s = Σ_{s=0}^{L-1} 2μ η_eff^(s) (1 − β/2 · η_eff^(s))`

is computable from the orbit alone.

**Proof:** Apply the lemma at each step s = 0,...,L−1. Since V_s > 0 for all s, Case 2 does not occur, and conclusion (B) holds at each step:
```
φ_{s+1} ≤ φ_s − (1−α) c_s
```
Telescope over L steps:
```
φ_{t+L} ≤ φ_t − (1−α) Σ_{s=0}^{L-1} c_s = φ_t − (1−α) C(O)   ∎
```

**Combined statement (with convergence case):** If V_t > 0 fails at some step s within the orbit, then convergence has already occurred (V_s = 0). Either way, L(w_t) → L*.

---

## Convergence rate corollary

If V_t > 0 for every full orbit (no finite-step convergence), then for k complete orbits:
```
φ_{kL} ≤ φ_0 − k(1−α) C(O)
```
This reaches φ = 0 (i.e., V = 0) after at most `⌈φ_0 / [(1−α) C(O)]⌉` orbits.

Translating back: `V_t → 0` in at most `⌈V_0^{1−α} / [(1−α) C(O)]⌉ · L` steps.

---

## Comparison with §8.2 (PL/α=1)

The φ-approach gives a **sublinear analogue** of §8.2, not a reduction:

| | §8.2 (PL, α=1) | φ-theorem (Łojasiewicz, α<1) |
|---|---|---|
| Per-orbit statement | V_{t+L} ≤ ρ_PL · V_t (multiplicative) | φ_{t+L} ≤ φ_t − (1−α)C(O) (additive in φ) |
| Orbit quantity | ρ_PL = ∏(1−c_t) ∈ (0,1) | C(O) = Σ c_t > 0 |
| Rate | Geometric (exponential in steps) | Polynomial in steps (via φ) |
| Convergence | Infinite steps to V=0 (geometric) | Finite φ-budget: ≤ φ_0/[(1−α)C(O)] orbits |

The PL case is **strictly stronger**: multiplicative contraction implies additive φ-contraction (check: if V_{t+L} ≤ ρV_t then φ_{t+L} = V_{t+L}^{1-α} ≤ ρ^{1-α} φ_t, a different form). The two theorems are not equivalent at α=1.

---

## What remains open after B1

B1 establishes the one-step lemma and orbit corollary rigorously. The remaining open items for the full theorem:

| Item | Status | What's needed |
|---|---|---|
| O1: α not computable from orbit | Remains open | α as verified input to cert (acceptable — same as β, μ in §8.2) |
| O2: V_t > 0 along orbit | Hypothesis in current lemma | B2: semi-algebraic L ⇒ no critical points along QA orbit (generically) |
| C(O) > 0 | ✓ Follows from orbit feasibility condition η_eff < 2/β and μ > 0 | Proved: c_t = 2μη_t(1−βη_t/2) > 0 for η_t ∈ (0, 2/β) |

**B1 is clean and solid.** The minimum viable theorem is now:

> *Let L be β-smooth, satisfy the Łojasiewicz condition with exponent α ∈ (0,1) and constant μ > 0, and have no critical points along the QA cosmos orbit O. Then `φ_{t+L} ≤ φ_t − (1−α)C(O)` where C(O) is orbit-computable and φ_t = (L(w_t)−L*)^{1−α}.*

This is a proved theorem (modulo B2 for the "no critical points" hypothesis), not a conjecture.
