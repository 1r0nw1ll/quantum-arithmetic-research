# QA Theorem: Monoidal Curvature Functor

**Theorem ID**: QA_THEOREM_MONOIDAL_CURVATURE_FUNCTOR.v1
**Status**: Active | **Date**: 2026-02-19
**Sibling theorem**: `docs/theory/QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.md`
**First certified instance**: Family [64] — QA Kona EBM QA-Native Orbit Reg (`2be5ce6`)

---

## Scope

This theorem elevates the curvature scalar κ from a per-family stability coefficient to a monoidal functor norm on QA dynamics morphisms. It applies to any QA family that declares a deviation functional and an affine contraction bound. Parallel composition under ⊗ preserves only the minimum curvature (bottleneck law).

It is not family-specific. It applies to: EBM training, QARM stochastic transitions, control-loop regularization, and any QA dynamical system where an affine contraction bound can be stated.

---

## Setup

**D1. Symmetric Monoidal QA Dynamics Category**

Let `(DynQA, ⊗, 𝕀)` be a symmetric monoidal category where:

- Objects `X` are QA dynamical systems (state space `S_X`, generator algebra `G_X`, admissible noise policy)
- Morphisms `f: X → X` are one-step evolution maps (deterministic or stochastic kernels)
- Tensor product `X ⊗ Y` denotes independent parallel composition: state space `S_{X⊗Y} = S_X × S_Y`, independent generators, independent noise
- Unit object `𝕀` is the trivial (zero-deviation) system

**D2. Deviation Functional and Functor**

For each object `X`, define a measurable deviation functional `D_X: S_X → ℝ_≥0`.

Additive deviation under ⊗:

```
D_{X⊗Y}(x,y) := D_X(x) + D_Y(y)
```

The (lax) monoidal deviation functor `𝒟: DynQA → (ℝ_≥0, ⊕, 0)` sends `X ↦ D_X` and `f ↦` induced deviation evolution bound.

**D3. Affine Contraction Bound**

For each system `X` and time `t`, there exist:

- contraction factor `α_X(t) ∈ [0,1]`
- dispersion term `b_X(t) ≥ 0`

such that for the stochastic evolution `s_{t+1} ~ Π_X(·|s_t)`:

```
E[D_X(s_{t+1}) | s_t] ≤ α_X(t)·D_X(s_t) + b_X(t)
```

**D4. Monoidal Composition Law**

For independent systems `X` and `Y`:

```
α_{X⊗Y}(t) := max(α_X(t), α_Y(t))
b_{X⊗Y}(t) := b_X(t) + b_Y(t)
```

Parallel composition takes the worst contraction factor and adds noise floors.

**D5. Monoidal Curvature Norm**

```
|κ|⊗(X,t) := 1 − α_X(t)
```

Interpretation:

- `|κ|⊗ > 0` — strict contraction (restorative generators dominate)
- `|κ|⊗ = 0` — neutral stability (boundary)
- `|κ|⊗ < 0` — expansive instability (`NEGATIVE_GENERATOR_CURVATURE`)

**D6. Stationary Contraction Regime**

Assume `α_X(t) ≡ α_X < 1` and `b_X(t) ≡ b_X < ∞`. This defines a stationary contraction regime.

---

## Theorem T_mono: Curvature Bottleneck Under Monoidal Composition

Under D1–D5:

```
|κ|⊗(X⊗Y, t) = min(|κ|⊗(X,t), |κ|⊗(Y,t))
```

**Proof sketch**: From D3 on both subsystems:

```
E[D_{X⊗Y}(t+1)] ≤ α_X·D_X(t) + b_X + α_Y·D_Y(t) + b_Y
```

Since `D_{X⊗Y} = D_X + D_Y` and `α_{X⊗Y} = max(α_X, α_Y)` (D4):

```
|κ|⊗(X⊗Y) = 1 − max(α_X, α_Y) = min(1−α_X, 1−α_Y)
```

∎

**Interpretation**: Parallel systems are limited by the weakest curvature component. Certification of coupled systems requires all factors to have positive curvature.

---

## Corollary C_mono: Noise Floor Bound

Under D6, the deviation process satisfies the geometric decay bound:

```
E[D_X(t)] ≤ α_X^t · D_X(0) + b_X / (1 − α_X)
```

Stationary deviation bound:

```
E[D_X(∞)] ≤ b_X / |κ|⊗(X)
```

For product systems:

```
E[D_{X⊗Y}(∞)] ≤ (b_X + b_Y) / min(|κ|⊗(X), |κ|⊗(Y))
```

---

## Specialization to Existing QA Families

**Quadratic Projection Case (Family [64], H = λQ)**:

```
α_X(t) = |1 − η_t·λ|    →    |κ|⊗(t) = 1 − |1 − η_t·λ|
```

Exactly matches the Gate-3 certified one-liner.

**Drift-Noise Curvature Case** (general affine bound):

```
E[D_{t+1}] ≤ (1 − η_t·λ_b)·D_t + η_t²·σ²

α_X(t) = 1 − η_t·λ_b    →    |κ|⊗(t) = η_t·λ_b

Noise floor: E[D(∞)] ≤ η·σ² / |κ|⊗
```

---

## Certifiable Invariant Structure

The following quantities are certifiable as Gate-level invariants:

| Quantity | Formula | Attestation |
|---------|---------|-------------|
| `kappa_hat_per_epoch[t]` | `\|κ\|⊗(X,t) = 1 − α_X(t)` | Gate 3 recompute |
| `min_kappa_hat` | `min_t(\|κ\|⊗(X,t))` | Gate 3 argmin check |
| `min_kappa_epoch` | `argmin_t(\|κ\|⊗(X,t))` (tie: first) | Gate 3 argmin check |
| `max_dev_norm` | `max_t(D_X(t))` | Gate 3 argmax check |
| `max_dev_epoch` | `argmax_t(D_X(t))` (tie: first) | Gate 3 argmax check |
| `kappa_hash` | SHA-256 of `kappa_hat_per_epoch` | Gate 4 hash-chain |

---

## Obstruction Classes

| Obstruction | Condition | Type |
|------------|-----------|------|
| `NEGATIVE_GENERATOR_CURVATURE` | `\|κ\|⊗(t) ≤ 0` | Structural |
| `CURVATURE_RECOMPUTE_MISMATCH` | cert value ≠ recomputed | Integrity |
| `MAX_DEV_SPIKE_ATTESTATION_MISMATCH` | argmax deviation epoch mismatch | Integrity |
| `DISPERSION_BOUND_MISSING` | no certified noise bound `b_X` | Missing attestation |
| `BASIN_ESCAPE` | system exits certified contraction region | Structural |

`NEGATIVE_GENERATOR_CURVATURE` is a **structural obstruction**: it cannot be silenced by cert edits because Gate 4's hash-chain seal prevents undetected tampering. It represents a genuine claim that the dynamics were in an instability regime.

---

## Time Composition (Serial)

Under serial composition (two steps in sequence), log-contraction accumulates additively:

```
α(f ∘ g) ≤ α(f)·α(g)    →    −log(α(f∘g)) ≥ −log(α(f)) + −log(α(g))
```

This gives two orthogonal composition structures:

- **⊗ (parallel)**: curvature = bottleneck-min
- **∘ (serial/time)**: curvature accumulates additively in log-space

---

## Relation to Generator Interaction Curvature Theorem

This theorem generalizes `QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.v1`:

| Property | Generator Interaction Theorem | Monoidal Curvature Functor |
|----------|------------------------------|---------------------------|
| Setting | Differentiable restoring potential R | Affine contraction bound (any generator type) |
| Curvature form | `κ_QA = λ − ½·η·σ²_dev` (heuristic) | `\|κ\|⊗ = 1 − α_X` (exact) |
| Closed-form case | `κ̂_QA = 1 − \|1 − ηλ\|` (exact spectral) | Same (`α_X = \|1 − ηλ\|`) |
| Composition | Not addressed | Bottleneck-min under ⊗ |
| Applies to | Gradient + stochastic systems | Any QA dynamical system |

---

## Implementations

| Family | Regime | `\|κ\|⊗` Formula | Status |
|--------|--------|-----------------|--------|
| [64] QA Kona EBM Orbit Reg | Closed-form (H=λQ) | `1 − \|1 − lr·lambda_orbit\|` | ✓ Shipped (`2be5ce6`) |
| Future EBM families | Closed-form or affine bound | TBD | Planned |
| QARM families | Discrete generator affine bound | TBD | Planned |

---

## References

- `docs/QA_DYNAMICS_SPINE.md` — opt-in standard for Dynamics-Compatible families
- `docs/theory/QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.md` — sibling theorem (differentiable regime)
- `memory/family64_theory.md` — derivations, paper-ready result paragraph
- ChatGPT theoretical review (2026-02-19) — source of monoidal functor formulation and composition laws
