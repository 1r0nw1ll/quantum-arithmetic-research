# QA Curvature Unification: arXiv Appendix

**Document ID**: QA_CURVATURE_UNIFICATION_APPENDIX.v1
**Status**: Active | **Date**: 2026-02-19
**Unifies**:
- `QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.v1` — gradient/CD-1 regime
- `QA_THEOREM_QARM_GENERATOR_INTERACTION_CURVATURE.v1` — discrete QARM regime
- `QA_THEOREM_MONOIDAL_CURVATURE_FUNCTOR.v1` — monoidal composition
**Authored by**: ChatGPT architectural review (2026-02-19); formatted for QA docs style

---

## A.1 Unified Setup

Let `X` be a finite-dimensional real vector space. Let `Q: X → X` be a symmetric idempotent
projection (`Q² = Q`, `Qᵀ = Q`). Define the deviation component:

```
Δ_t := Q·x_t
```

All three theorems are statements about the deviation process `Δ_t`. We consider
discrete-time dynamics:

```
x_{t+1} = x_t − η_t·(D(x_t) + ζ_t)
```

where `D` is a restorative drift operator, `ζ_t` is zero-mean noise, and `η_t > 0` is step
size. Projecting via Q:

```
Δ_{t+1} = Δ_t − η_t·(Q·D(x_t) + ξ_t),    ξ_t := Q·ζ_t
```

---

## A.2 Gradient Generator Curvature (Theorem A.1)

**Applies to**: SGD, CD-1, and related stochastic gradient systems.

**Assumptions G1–G4**:

- **(G1) Restricted Strong Convexity**: `⟨Δ, Q∇R(x)⟩ ≥ λ·‖Δ‖²` for some λ > 0
- **(G2) Lipschitz Smoothness**: `‖Q∇R(x) − Q∇R(y)‖ ≤ L·‖x−y‖`, L ≥ λ
- **(G3) Noise**: `E[ξ_t | F_t] = 0`, `E[‖ξ_t‖² | F_t] ≤ σ²`
- **(G4) Step size**: `0 < η_t < 2λ/L²`

**Definition A.1 (Gradient Curvature)**:

```
κ_grad(t) := λ − η_t·L²/2
```

Note: `κ_grad > 0` is a noise-dominance certificate, not the primary stability condition.
The stability condition is G4.

**Theorem A.1**: Under G1–G4:

```
E[‖Δ_{t+1}‖²] ≤ (1 − 2η_t·λ + η_t²·L²)·‖Δ_t‖² + η_t²·σ²
```

If `0 < η_t < 2λ/L²`:

```
lim sup E[‖Δ_t‖²] ≤ η·σ² / (2λ − η·L²)
```

**Interpretation**: Noise floor scales as η·σ². LR decay reduces the floor and exponentially
suppresses basin escape (Freidlin–Wentzell limit).

---

## A.3 QARM Generator Curvature (Theorem A.2)

**Applies to**: Discrete QA generator algebra (QARM), reachability systems, symbolic AI
pipelines.

**Assumptions Q1–Q4** (basin-local):

- **(Q1) Basin-Local Mean Contraction**: Within basin `B`,
  `E[V(s_{t+1}) | F_t] ≤ (1 − 2η_t·λ_b)·V(s_t) + η_t²·E[‖w_t‖² | F_t]`
- **(Q2) Drift Lipschitz Bound**: `‖Δ(g(s)) − Δ(g(s'))‖ ≤ L_b·‖Δ(s) − Δ(s')‖` within `B`
- **(Q3) Dispersion Bound**: `E[w_t | F_t] = 0`, `E[‖w_t‖² | F_t] ≤ σ²_dev(t)`
- **(Q4) Basin Containment**: trajectory remains in `B`

**Definition A.2 (QARM Curvature)**:

```
κ_QARM(t) := λ_b − η_t·L_b²/2
```

**Theorem A.2**: Under Q1–Q4, if `0 < η < 2λ_b/L_b²`:

```
lim sup E[V(s_t)] ≤ η·σ²_dev / (2λ_b − η·L_b²)
```

**Corollary (Escape destroys guarantees)**: All bounds hold only while `s_t ∈ B`. Basin
escape → `BASIN_ESCAPE` obstruction, guarantees cease.

---

## A.4 Monoidal Curvature Bottleneck (Theorem A.3)

**Applies to**: Parallel composition of any QA dynamical systems.

**Setup**: Each system `X` satisfies an affine contraction bound:

```
E[D_X(s_{t+1}) | s_t] ≤ α_X(t)·D_X(s_t) + b_X(t)
```

**Definition (Monoidal Curvature Norm)**:

```
|κ|⊗(X,t) := 1 − α_X(t)
```

**Theorem A.3 (Bottleneck Law)**: For independent parallel systems X, Y:

```
α_{X⊗Y}(t) = max(α_X(t), α_Y(t))

|κ|⊗(X⊗Y, t) = min(|κ|⊗(X,t), |κ|⊗(Y,t))
```

**Corollary (Noise floor)**:

```
E[D_X(∞)] ≤ b_X / |κ|⊗(X)
```

**Composition laws**:
- `⊗` (parallel): curvature = bottleneck-min
- `∘` (serial/time): log-contraction accumulates additively

---

## A.5 Unified View

All three theorems are instances of one curvature–contraction principle:

| System Type | Curvature | Noise Floor |
|-------------|-----------|-------------|
| Gradient (A.1) | `λ − η·L²/2` | `η·σ² / (2λ − η·L²)` |
| QARM (A.2) | `λ_b − η·L_b²/2` | `η·σ²_QARM / (2λ_b − η·L_b²)` |
| Monoidal (A.3) | `1 − α` | `β / (1−α)` |

Stability in QA dynamics is controlled by: **contraction margin minus stochastic
dispersion**.

Curvature is:
- **Spectral** — quadratic case (H=λQ): `κ̂ = 1 − |1 − ηλ|` (exact)
- **Basin-local** — nonconvex/QARM case: `κ = λ_b − η·L_b²/2`
- **Algebraic** — discrete generator algebra
- **Functorial** — monoidal composition

---

## A.6 Certification Mapping (Unified Gate 3/4)

| Mathematical Quantity | QA Cert Field |
|----------------------|---------------|
| `‖Δ_t‖` | `reg_norm_per_epoch[t]` (gradient) or `deviation_norm_per_step[t]` (QARM) |
| `κ(t)` | `kappa_hat_per_epoch[t]` |
| `min_t κ(t)` | `min_kappa_hat` |
| `argmin_t κ(t)` | `min_kappa_epoch` |
| `max_t ‖Δ_t‖` | `max_dev_norm` |
| `argmax_t ‖Δ_t‖` | `max_dev_epoch` |
| Hash of curvature trace | `kappa_hash` (SHA-256) |

**Gate 3**: Recompute curvature from certified primitives; check hash, argmin/argmax
consistency, non-negativity.

**Gate 4**: Include `kappa_hash` in deterministic replay hash-chain. Tampering →
`TRACE_HASH_MISMATCH`.

---

## A.7 Universal Obstruction Algebra

| Obstruction | Trigger | Applies to |
|------------|---------|-----------|
| `NEGATIVE_GENERATOR_CURVATURE` | κ(t) ≤ 0 | All regimes |
| `CURVATURE_RECOMPUTE_MISMATCH` | cert ≠ recomputed | All regimes |
| `MAX_DEV_SPIKE_ATTESTATION_MISMATCH` | argmax epoch mismatch | All regimes |
| `BASIN_ESCAPE` | trajectory exits certified basin | QARM regime |
| `DRIFT_CONTRACTION_VIOLATION` | D3′ fails empirically | QARM regime |
| `DISPERSION_BOUND_MISSING` | no certified σ²_dev | QARM regime |
| `MOVE_FAIL_TYPE_MISMATCH` | logged failure algebra ≠ replay | QARM regime |

---

## A.8 Conclusion

Theorems A.1–A.3 are instances of a single principle: **stability in QA dynamics is
controlled by a curvature–noise inequality**. The curvature object `κ` is:

> A universal QA dynamical invariant — spectral in the gradient/quadratic case, basin-local
> in the nonconvex/QARM case, and functorial under monoidal composition.

---

## References

- `docs/theory/QA_THEOREM_GENERATOR_INTERACTION_CURVATURE.md` — Theorem A.1 (full)
- `docs/theory/QA_THEOREM_QARM_GENERATOR_INTERACTION_CURVATURE.md` — Theorem A.2 (full)
- `docs/theory/QA_THEOREM_MONOIDAL_CURVATURE_FUNCTOR.md` — Theorem A.3 (full)
- `docs/QA_DYNAMICS_SPINE.md` — opt-in certification standard
- `memory/family64_theory.md` — derivations, paper-ready result paragraph
- ChatGPT architectural review (2026-02-19) — source of unified appendix structure
