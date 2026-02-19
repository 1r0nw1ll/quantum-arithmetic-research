# QA Theorem: Generator Interaction Curvature

**Theorem ID**: QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.v1
**Status**: Active | **Date**: 2026-02-18
**First certified instance**: Family [64] — QA Kona EBM QA-Native Orbit Reg (`2be5ce6`)

---

## Scope

This theorem applies to any QA cert family where:
- A state `x_t` evolves under a stochastic update rule
- A **restoring generator** (regularizer, consensus, contraction) acts on a deviation subspace
- A **stochastic generator** (noise, approximation error, exploration) acts against contraction

It is not family-specific. It applies to: EBM training, QARM stochastic transitions, control-loop regularization, any "regularizer + noise" generator composition.

---

## Setup

**D1. State and deviation operator**

Let `x_t ∈ ℝⁿ` be the parameter state at step `t`. Let `Q ∈ ℝⁿˣⁿ` be a symmetric idempotent projector (`Q² = Q`, `Qᵀ = Q`) extracting deviation modes. Define:

```
Δ_t := Q · x_t   (intra-subspace deviation)
```

**D2. Update law**

```
x_{t+1} = x_t − η_t · (∇R(x_t) + g_t)
```

where `R` is the restoring potential, `g_t` is the stochastic disturbance, `η_t > 0` is the step size. Projecting into deviation space:

```
Δ_{t+1} = Δ_t − η_t · (Q∇R(x_t) + ξ_t),   ξ_t := Q·g_t
```

**D3. Restricted strong convexity**

```
⟨Δ, Q∇R(x)⟩ ≥ λ·‖Δ‖²   for some λ > 0
```

> This condition holds whenever (a) R acts only on Im(Q) — i.e., R(x) = R̃(Qx), so Q∇R(x) = Q∇R(Qx) — or (b) the full-space Hessian satisfies Q·∇²R(x)·Q ⪰ λQ. The case R(x) = ½λ‖Qx‖²_F (family [64]) satisfies both.

**D4. Deviation noise bound**

```
E[ξ_t | F_t] = 0,   E[‖ξ_t‖² | F_t] ≤ σ²_dev(t)
```

Note: in CD-1/CD-k, `ξ_t` includes both minibatch sampling noise and negative-phase truncation error. `σ²_dev(t)` is an effective covariance that can increase in stalled or oscillatory regimes.

**D5. Generator interaction curvature**

```
κ_QA(t) := λ − ½ · η_t · σ²_dev(t)
```

> Note: D5 is a heuristic contraction-margin proxy valid when σ²_dev is observable.
> The rigorous discrete-time contraction coefficient (exact, from Lemma X.1) is:
>
>     κ_QA^disc(t) = 1 − √(1 − 2η_t·λ + η_t²·L²)
>
> For small η_t this approximates η_t·λ. In the closed-form special case (H = λQ, L = λ),
> the one-line certified formula κ̂_QA = 1 − |1 − η_t·λ| is the EXACT deviation-subspace
> spectral contraction margin — not an approximation.

**D6. Lipschitz smoothness of the restoring gradient**

There exists L ≥ λ such that:

```
‖Q∇R(x) − Q∇R(y)‖ ≤ L·‖Q(x − y)‖
```

Equivalently on deviation modes: ‖Q∇R(x)‖ ≤ L·‖Δ‖.

In the quadratic special case R(x) = ½λ‖Qx‖²_F, we have L = λ exactly.

---

## Theorem T1: Curvature-Dominance Implies Contraction

Under D1–D4, the deviation energy `V(Δ) = ½‖Δ‖²` satisfies:

```
E[V(Δ_{t+1}) | F_t] ≤ (1 − 2η_t·λ + η_t²·L²)·V(Δ_t) + ½·η_t²·σ²_dev(t)
```

where `L` is (L from D6). Whenever `κ_QA(t) > 0`, the process is mean-square stable and admits the steady-state noise floor:

```
E[‖Δ_t‖²] ≲ η·σ²_dev / (2λ)
```

Mean-square contraction requires the coefficient (1 − 2η_t·λ + η_t²·L²) < 1,
which simplifies to: 0 < η_t < 2λ/L²  (discrete-time stability window, explicit in L).
For the quadratic special case L = λ: 0 < η_t < 2/λ.

---

## Corollary C1: Learning-Rate Noise Suppression

For constant `η` and bounded `σ²_dev ≤ σ²`:

Steady-state noise floor (exact bound):

```
E[‖Δ_t‖²] ≤ η·σ²_dev / (2λ − η·L²)
```

For η ≪ 2λ/L² (well within stability window), this approximates η·σ²_dev / (2λ).

Decreasing `η` by factor `r = η'/η` reduces the noise floor by the same factor `r`. The contraction rate `λ` is unchanged. Signal-to-noise ratio improves as `1/η` (in the regime η ≪ 2λ/L²).

---

## Corollary C2: Flat LR Can Eventually Destabilize

If `η` is constant while `σ²_dev(t)` increases (e.g., stalled CD-1 regime), then `κ_QA(t)` can cross zero:

```
∃t: κ_QA(t) < 0  ⟹  deviation growth / basin escape becomes inevitable
```

This is the formal explanation of "Phase C collapse" observed in family [64].

---

## Corollary C3: Freidlin–Wentzell Escape Bound

In the continuous-time (Ornstein–Uhlenbeck) limit with effective temperature `T = η·σ²`:

```
P(escape from basin of radius r) ≍ exp(−λ·r² / (2·η·σ²))
```

LR decay reduces the exponent denominator, exponentially suppressing basin escape.

---

## Closed-Form Special Case: H = λQ

When the restoring potential is the mean-subtraction quadratic `R(x) = ½‖Qx‖²_F`, the full-space Hessian is ∇²R = λQ. Restricted to Im(Q), it acts as λI (since Qv = v for all v ∈ Im(Q)). Both descriptions are correct from different perspectives; the full-space expression is λQ, the restriction to the deviation subspace is λI. The deviation linear map is `A = I − η·H`, with:

```
ρ_dev(A) = |1 − η·λ|   (deviation-subspace spectral radius)

κ̂_QA = 1 − |1 − η·λ|   (one-line formula, no probe required)
```

**Stability condition**: `0 < η·λ < 2`

This is the ideal QA pattern: certifiable curvature with no extra measurement channels.

**Gate 3 recomputation** (one-liner):
```python
kappa_hat = round(1.0 - abs(1.0 - lr * lambda_orbit), 8)
```

**Instability threshold**: `η·λ = 2` (spectral radius = 1, marginal stability)

---

## Certifiable Invariant Structure

The following quantities are certifiable as Gate-level invariants:

| Quantity | Formula | Attestation |
|---------|---------|-------------|
| `kappa_hat_per_epoch[t]` | `1 − \|1 − η_t·λ\|` (exact spectral margin for H=λQ; rigorous form of D5 in this case) | Gate 3 recompute |
| `min_kappa_hat` | `min(kappa_hat_per_epoch)` | Gate 3 argmin check |
| `min_kappa_epoch` | `argmin(kappa_hat_per_epoch)` (tie: first) | Gate 3 argmin check |
| `max_dev_norm` | `max(dev_norm_per_epoch)` | Gate 3 argmax check |
| `max_dev_epoch` | `argmax(dev_norm_per_epoch)` (tie: first) | Gate 3 argmax check |
| `kappa_hash` | SHA-256 of `kappa_hat_per_epoch` | Gate 4 hash-chain |

---

## Obstruction Classes

| Obstruction | Trigger | Type |
|------------|---------|------|
| `NEGATIVE_GENERATOR_CURVATURE` | Recomputed κ̂ < 0 | Structural (recompute-driven) |
| `CURVATURE_RECOMPUTE_MISMATCH` | Cert value ≠ recomputed | Integrity |
| `CURVATURE_PROBE_MISMATCH` | Probe definition mismatch | Integrity (probe regime) |
| `MAX_DEV_SPIKE_ATTESTATION_MISMATCH` | Claimed max ≠ actual argmax | Integrity |

`NEGATIVE_GENERATOR_CURVATURE` is a **structural obstruction**: it cannot be silenced by cert edits because Gate 4's hash-chain seal prevents undetected tampering. It represents a genuine claim that the dynamics were in an instability regime.

---

## Implementations

| Family | Regime | κ̂ Formula | Status |
|--------|--------|-----------|--------|
| [64] QA Kona EBM Orbit Reg | Closed-form (H=λQ) | `1 − \|1 − lr·lambda_orbit\|` | ✓ Shipped (`2be5ce6`) |
| Future EBM families | Closed-form or probe | TBD | Planned |
| QARM families | Probe-based | TBD | Planned |

---

## Mapping to Logged Quantities (Family [64])

| Theory | Code | Meaning |
|--------|------|---------|
| `‖Δ_t‖_F` | `reg_norm_per_epoch[t]` | Deviation norm |
| `η_t` | `lr_per_epoch[t]` | Learning rate (temperature) |
| `λ` | `lambda_orbit` | Regularizer strength (restoring force) |
| `σ²_dev(t)` | (not directly logged; absorbed into κ via closed-form) | Effective noise covariance |
| `κ̂_QA(t)` | `generator_curvature.kappa_hat_per_epoch[t]` | Generator interaction curvature |

---

## References

- `docs/QA_DYNAMICS_SPINE.md` — opt-in standard for Dynamics-Compatible families
- `docs/families/64_kona_ebm_qa_native_orbit_reg.md` — reference implementation human tract + theoretical interpretation
- `memory/family64_theory.md` — derivations, Mandt SDE limit, spectral radius proof, paper-ready result paragraph
- ChatGPT theoretical review (2026-02-18) — source of universal theorem formulation + journal-level appendix
