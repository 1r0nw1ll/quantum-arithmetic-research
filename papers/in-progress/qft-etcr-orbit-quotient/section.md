# ETCR as Slice Artifact, Dirac Bracket as Quotient Correction

## Two QFT Convergences with QA's Structural Firewall

**Draft status**: Cert-backed section, FST Part 3 candidate. Author: Will Dale. Primary-source-grounded; mapping claims flagged as *structural* throughout. Certs [260] and [262] shipped 2026-04-21 (see §6).

**Cert-backfill complete.** The two supporting cert families are in the ledger:

- **[260] QA Orbit-Dirac Bracket Cert v1** — delivers the MC-3 / MC-4 construction on the Blaschke-Gieres side (§4).
- **[262] QA Unequal-k CCR Invariant Cert v1** — delivers the MC-1 / MC-2 construction on the Mannheim side (§3).

Both are registered in `qa_alphageometry_ptolemy/qa_meta_validator.py` and pass 1 PASS + 1 FAIL fixture under the standard self-test harness. Artifacts under `qa_alphageometry_ptolemy/qa_orbit_dirac_bracket_cert_v1/` and `qa_alphageometry_ptolemy/qa_unequal_k_ccr_invariant_cert_v1/`.

All explicit constructions referenced here are validated in the QA cert system ([260], [262]); the paper includes only the minimal formulas required for conceptual completeness, with full machine-verified proofs living in the cert layer.

---

## 1. Introduction

Quantum Arithmetic (QA) posits a **structural firewall** — Theorem NT — separating discrete dynamical objects (integer tuples evolving on modular orbits) from continuous observer-side measurements (real or complex amplitudes at the input/output boundary). The firewall permits exactly two boundary crossings per computation: input encoding and output projection. All causal dynamics lives on the discrete side; continuous functions are observer projections only.

Two well-known results in canonical quantum field theory (QFT) land structurally inside this firewall when re-read through QA vocabulary, without either result having been framed this way in its own literature.

The first is due to Mannheim [Phys. Rev. D 102 025020 (2020)]: **equal-time commutation relations in instant-time and light-front quantization cannot be transformed into each other, yet unequal-time commutators of the two schemes are unitarily equivalent to all orders in interacting theory.** The invariant object of canonical quantization is the unequal-time, not the equal-time, commutator.

The second is due to Blaschke & Gieres [Nucl. Phys. B 965 (2021)]: **canonical and gauge-invariant energy-momentum vectors do not generate spacetime translations via the Poisson bracket in a general gauge.** The correct generator requires the Dirac bracket, constructed by quotienting second-class constraints out of the phase-space bracket.

We read both results as QFT-side structural analogues of a single QA principle: **slices and unquotiented brackets are observer-side constructs; paths and quotient-corrected brackets are the physical invariants.** The convergence is structural, not a derivation of QA from QFT or vice versa. QA predicts the shape of the distinction; QFT exhibits it in its own canonical vocabulary.

This section formalizes both mappings with primary-source equations, flags the precision guardrails, and sketches the two certificate families that backfill the argument.

---

## 2. QA Primitives

We use six primitives, all defined in the QA axioms spec and the firewall theorem [QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1].

- **A1 (no zero).** QA states live in `{1,...,m}`, not `{0,...,m−1}`. Zero residue never labels a state.
- **A2 (derived coords).** Given a base pair `(b, e)` with `b, e ∈ {1,...,m}`, the remaining coordinates are `d = b + e` and `a = b + 2e`, always derived, never independently assigned.
- **T1 (path time).** QA time is integer path length `k ∈ ℕ` — the number of T-operator steps applied to the initial pair. QA carries no continuous time variable.
- **T2 / Theorem NT (firewall).** Continuous real or complex amplitudes are observer-projection-side only. No continuous input is causally read by discrete QA logic; no QA state is transmitted across the boundary except at the two permitted crossings.
- **Orbits.** The T-operator partitions the pair space `{1,...,m}² \ A1-excluded` into orbit classes. For `m = 9`: one Singularity (fixed point at (9,9)), a Satellite family (period-8 orbits, 8 pairs each), and Cosmos (period-24, 72 pairs). The orbit is the QA structural object that witnesses "which dynamics a state belongs to."
- **[191] Tiered Reachability Theorem.** On `S_9`, Level-I reachability ceilings at 26% of pair space. Orbits stratify further by whether they are within the structural ceiling.

Throughout, we write `T : ℋ_raw → ℋ_raw` for the QA evolution operator on the pair space `ℋ_raw = ({1,...,m})²`, and use `T^k` for `k` iterations.

---

## 3. Mannheim: Equal-Time ETCRs Differ; Unequal-Time Commutator is the Invariant

### 3.1 The slice dependence

Mannheim's first observation concerns the free scalar field with action

```
I_S = ∫ d⁴x ½ [(∂₀φ)² − (∂ᵢφ)² − m²φ²].                                (1.1)
```

Instant-time canonical quantization yields the equal-time commutation relation

```
[φ(x⁰, x⃗), ∂₀φ(x⁰, y⃗)] = i δ³(x⃗ − y⃗),                                (1.2)
[φ(x⁰, x⃗), φ(x⁰, y⃗)]   = 0.                                           (1.6)
```

Light-front quantization (coordinates `x± = x⁰ ± x³`) yields instead

```
[φ(x⁺, x⃗⊥, x⁻), 2∂₋φ(x⁺, y⃗⊥, y⁻)] = i δ(x¹−y¹) δ(x²−y²) δ(x⁻−y⁻).     (1.4)
```

Mannheim observes directly: these two equal-time commutators "cannot be transformed into each other" and "take completely different forms." For fermions the difference is sharper still — the light-front canonical anticommutator carries a non-invertible projection operator `Λ₊ = ½γ⁰γ⁺` [Mannheim eq (1.14)], whereas the instant-time anticommutator does not.

### 3.2 The unequal-time restoration

Using the Fock expansion (eq 2.1), Mannheim computes the unequal-instant-time commutator between two free scalar fields:

```
i∆(IT; x) = [φ(x⁰, x⃗), φ(0)]
          = ∫ d³p / [(2π)³ 2E_p] × (e^{−i(E_p x⁰ + p⃗·x⃗)} − c.c.).      (2.2)
```

This is a c-number, not a q-number, and its equal-time limit reproduces both (1.2) and (1.6). The analogous light-front c-number `i∆(LF; x)` recovers (1.4) at `x⁺ = 0`.

### 3.3 All-order equivalence via the Lehmann representation

For the interacting theory, Mannheim's central technical result [eqs (8.9), (8.16)] is:

```
i∆(IT, FULL; x−y) = ∫₀^∞ dσ² ρ(σ², IT) · i∆(IT, FREE; x−y, σ²),         (8.9)
i∆(LF, FULL; x−y) = ∫₀^∞ dσ² ρ(σ², LF) · i∆(LF, FREE; x−y, σ²).         (8.16)
```

These are Lehmann spectral representations for the full commutators. Mannheim constructs a unitary `U` with `U|Ω(IT)⟩ = |Ω(LF)⟩` and `Uφ(IT;0)U⁻¹ = φ(LF;0)` [eq (8.26)], yielding the operator identity

```
P₀²(IT) − P₁²(IT) − P₂²(IT) − P₃²(IT) = 4P₊(LF)P₋(LF) − P₁²(LF) − P₂²(LF)   (9.12)
```

to all orders. Conclusion: the unequal-time commutator (the c-number `i∆`) and the full Lehmann representation are scheme-invariant; the equal-time commutator is not.

### 3.4 QA reading

| Mannheim object | QA structural analogue |
|---|---|
| Instant-time slice `x⁰ = const` | Observer-projection cut at `k = k₀` |
| Light-front slice `x⁺ = const` | Alternate observer-projection cut |
| ETCR eqs (1.2), (1.4) | Slice-dependent CCR at a chosen `k` |
| Unequal-time `i∆(x−y)` eq (2.2) | Path invariant over orbit trajectory `k₁ → k₂` |
| Lehmann spectral density `ρ(σ²)` eq (8.9) | Orbit-class spectral decomposition |
| Unitary `U` eq (8.26) | Observer-reprojection between cuts |
| Operator identity eq (9.12) | Orbit-invariant structural identity |

**Mapping claim, softened.** Mannheim provides a QFT-side structural analogue of the NT distinction: equal-time relations depend on the chosen slicing, while the invariant object is the unequal-time dynamical commutator. The analogue is not identity. Mannheim works in canonical QFT with continuous time; QA's path time `k` is integer. What carries across is the *shape* of the distinction: slice-dependent observer object vs. path-invariant dynamical object.

**Precision guardrail on the QA side.** If we posit a QA bosonic CCR of the form

```
[â_{b,e}, â†_{b',e'}]_{k=k₀} = δ_{b,b'} δ_{e,e'}
```

at a fixed path-time slice, this is a **proposed observer-side canonical encoding**, not a theorem derived from QA orbit dynamics. The path-invariant analogue of Mannheim's `i∆(IT; x−y)` is the unequal-`k` propagator

```
i∆_QA(k₁, k₂; (b,e) ↦ (b',e')) = ⟨Ω | â_{b',e'}(k₂) â†_{b,e}(k₁) | Ω⟩ − (reverse ordering)
```

traced over the T-operator trajectory. This propagator is the primary physical object in the QA reading; the equal-`k` CCR is its zero-separation limit, as (1.2) is to (2.2).

---

## 4. Blaschke-Gieres: Poisson Bracket Inadequate, Dirac Bracket Corrects

### 4.1 The inadequacy

Blaschke and Gieres [arXiv:2004.14406, §5] show that in free Maxwell theory (and more generally in gauge field theories), the canonical energy-momentum vector `P^μ` does not generate spacetime translations via the Poisson bracket in a general gauge. Their §5.6 gives the correction.

### 4.2 Constraints + gauge fixing

In the radiation gauge [eq (5.32)]:

```
FCC₁:  φ₁ ≡ π⁰ ≈ 0                    (primary, from A⁰ non-dynamical)
FCC₂:  φ₂ ≡ ∂ᵢπⁱ ≈ 0                  (secondary, Gauss law)
Gauge fix 1:  f₁ ≡ A⁰ ≈ 0
Gauge fix 2:  f₂ ≡ div A⃗ ≈ 0
```

Collecting `(ϕ_a)_{a=1..4} = (φ₁, φ₂, f₁, f₂)`, the first-class constraints (FCCs) plus gauge-fixing conditions form a second-class system.

### 4.3 Dirac-bracket construction

The central formula [eq (5.37)]:

```
{F, G}_D ≡ {F, G} − {F, ϕ_a} (X⁻¹)^{ab} {ϕ_b, G},     X_{ab} ≡ {ϕ_a, ϕ_b}.
```

Properties preserved: bilinearity, antisymmetry, Jacobi, derivation. Strong-zero on constraints [eq (5.38)]:

```
{ϕ_a, F}_D = 0     for any F.
```

Translation recovery [eq (5.39)]:

```
δ_a ϕ(x) ≡ {ϕ(x), a_μ P^μ}_D = a_μ ∂^μ ϕ(x).
```

### 4.4 QA reading

| Blaschke-Gieres object | QA structural analogue |
|---|---|
| Poisson bracket `{F, G}` on unconstrained phase space | Raw bracket on unquotiented pair space `ℋ_raw` |
| Constraint surface `ϕ_a ≈ 0` | Orbit-invariant subspace |
| Physical subspace Γ_r | Orbit-quotiented state space |
| Matrix `X_{ab} = {ϕ_a, ϕ_b}` | Raw-bracket matrix on QA constraint functions |
| Dirac bracket `{F, G}_D` | Orbit-Dirac bracket `[F, G]_orbit` |
| Strong zero `{ϕ_a, F}_D = 0` | Constraint functions as strong invariants |
| Translation recovery eq (5.39) | T-operator acts correctly on orbit-quotiented observables |

**Critical precision — constraint functions, not labels.** In Dirac's original setting, a constraint is a function `ϕ(p, q)` on phase space whose vanishing defines the constraint surface. Orbit-class *names* ("Singularity", "Satellite", "Cosmos") are strings, not functions; they cannot be plugged into eq (5.37). The QA Dirac-bracket construction requires explicit **constraint functions** whose vanishing cuts out the chosen orbit-invariant subspace.

Candidate constraint functions on `ℋ_raw = {1,...,m}²`:

- **Fixed-point constraint** for the Singularity: `ϕ_T(b, e) := T(b, e) − (b, e)` (vector-valued); vanishes iff `(b, e)` is a T-fixed point.
- **Period constraint**: `ϕ_n(b, e) := T^n(b, e) − (b, e)`; vanishes iff the orbit period of `(b, e)` divides `n`.
- **Orbit-invariant level set**: if a non-trivial T-invariant scalar `I : ℋ_raw → ℤ/mℤ` exists, then `ϕ_I(b, e) := I(b, e) − I₀` is a proper constraint function selecting the level set `I = I₀`.

The last candidate is the cleanest and is the primary target for the orbit-Dirac-bracket cert (§6).

A preliminary computational search over `{1,...,m}²` for m = 9 and m = 24, under the Fibonacci-like step `T_F(b, e) = (a1(b+e), b)` with `a1(x) = ((x−1) mod m) + 1`, identifies a non-trivial T-invariant scalar

```
I(b, e) := (b² − be − e²)²  mod m,
```

whose invariance follows from the Cassini sign-flip identity under `T_F`. For m = 9, `I` separates the three Cosmos orbits by value (I ∈ {1, 4, 7}); Singularity and Satellite share `I = 0` and would require an auxiliary constraint (e.g., the fixed-point constraint above). For m = 24, `I` is coarser — four distinct values across 30 orbits — so a full formulation at that modulus will combine `I`-level constraints with period constraints `ϕ_n(b, e) = T^n(b, e) − (b, e)` inside each level set.

This computational evidence suggested the required constraint-function family could be made explicit for the chosen discrete dynamics, and cert [260] `qa_orbit_dirac_bracket_cert_v1` has since delivered the construction on the Cassini +1 branch of the I=1 Cosmos orbit at m=9. The cert verifies, at witnesses (1, 8) and (1, 9), that the X-matrix is invertible (`det X ≡ 1 mod 9`, a unit), that `[b, e]_orbit = 0` on the physical subspace, and that the strong-zero property `[ϕ_a, F]_orbit = 0` holds for `F ∈ {b, e}` and `a ∈ {1, 2}` — Blaschke-Gieres eq (5.38) in its QA-native form. The result is scoped to `T_F` dynamics and one branch of one orbit at m=9; m=24 and the remaining branches / orbits are open extensions.

Given a constraint-function family `{ϕ_a}`, the **QA Dirac bracket** is

```
[F, G]_orbit ≡ [F, G] − [F, ϕ_a] (X⁻¹)^{ab} [ϕ_b, G].
```

The candidate base bracket is the QA tuple wedge, `{(b₁, e₁), (b₂, e₂)} := b₁ e₂ − b₂ e₁` — the antisymmetric bilinear form on pair space, which plays the role of the Poisson bracket here.

**Mapping claim, softened.** Blaschke-Gieres provides a QFT-side structural analogue of the orbit quotient that QA carries via its orbit machinery plus [191]. The analogue is not identity: Blaschke-Gieres's constraints are functional on Maxwell phase space; QA's constraints are functional on a finite modular pair space. What carries across is the *shape* of the correction: raw bracket is gauge-redundant, quotient-corrected bracket is gauge-invariant.

---

## 5. Unified Structural Claim

Theorem NT predicts a specific failure mode and a specific fix. The failure mode: anything constructed at a fixed observer slice, or on an unquotiented state space, is slice-dependent or redundancy-contaminated — not the physical invariant. The fix: move to the path (unequal-`k`) structure and to the quotient-corrected bracket.

Mannheim and Blaschke-Gieres exhibit this exact pair of failure-and-fix in two different QFT contexts, using entirely canonical-quantization and gauge-theory vocabulary, without QA framing:

- **Mannheim.** Failure: equal-time commutator depends on slice choice. Fix: unequal-time commutator (and its Lehmann spectral representation) is scheme-invariant to all orders.
- **Blaschke-Gieres.** Failure: Poisson bracket with canonical `P^μ` does not generate translations in general gauge. Fix: Dirac bracket with constraints quotiented produces the correct translation algebra.

Neither paper cites QA or any similar discrete framework. Both arrive independently at the structural distinction that Theorem NT names.

The QA-side prediction — that this same failure-and-fix should appear in any canonical formulation where a boundary between discrete physical structure and continuous observer projection is being drawn implicitly — can now be tested by looking at other canonical QFT constructions and asking: *where is the slice-dependent observer object, and what is its unequal-time / quotient-corrected invariant?* BRST, path integral, covariant canonical — each is a candidate for the same map.

**Synthesis, stated carefully.** The three components of the structural correspondence are:

- **QFT-side result.** The unequal-time dynamical commutator is the invariant object; the equal-time commutator is representation-dependent across quantization schemes [Mannheim eqs (1.2)/(1.4) vs. (2.2), (8.9), (8.16)]. The canonical Poisson bracket is gauge-redundant; the Dirac-bracket construction on second-class constraints is the quotient-corrected version that produces the correct translation algebra [Blaschke-Gieres eq (5.37)].
- **QA-side result.** The unequal-`k` propagator `i∆_QA(Δk; (b, e), (b', e'))` is the certifiable object ([262]); its equal-`k` limit is the stipulated observer-side δ-function CCR. The orbit-Dirac bracket `[F, G]_orbit` is the certifiable quotient-corrected bracket ([260]); the unquotiented tuple-wedge bracket on pair space is the observer-side raw form.
- **Synthesis.** The two sides are bridged by a cert-backed structural correspondence, not an identity or a derivation. QFT chooses its path-invariant / quotient-corrected object as the physical one; QA carries both sides natively by Theorem NT and certifies them with integer arithmetic throughout.

This is the sense in which QA and canonical QFT converge: not that Mannheim or Blaschke-Gieres confirm QA (they do not cite it and are not concerned with integer modular dynamics), but that the *shape* of the observer/invariant distinction is the same on both sides, and QA's shape is now anchored in cert ledger rather than carried by words.

---

## 6. Limits and Cert Backfill

This section establishes a structural analogy, not an identity or a derivation. The two cert families that back it have shipped:

- **[260] `qa_orbit_dirac_bracket_cert_v1`** (Blaschke-Gieres side, MC-3 / MC-4). Constraint family `ϕ_1 = b² − b·e − e² − 1` (Cassini +1 branch) and `ϕ_2 = b − 1` (gauge fix) on `ℋ_raw = {1, …, 9}²`. Base bracket: the symplectic lift of the tuple wedge `{(b₁, e₁), (b₂, e₂)} := b₁ e₂ − b₂ e₁` (prior-art C3 recast). X-matrix entries computed symbolically via `ℤ[b, e]` partial derivatives:

  ```
  X = [[0, b + 2e], [−(b + 2e), 0]],     det X = (b + 2e)².
  ```

  Verified at witnesses (1, 8) and (1, 9): `det X ≡ 1 mod 9` (unit), `[b, e]_orbit = 0` on the physical subspace, and `[ϕ_a, F]_orbit = 0` for every `a ∈ {1, 2}` and `F ∈ {b, e}` — the strong-zero property from eq (5.38). Pure integer arithmetic throughout; `X⁻¹` via `pow(x, −1, m)`.

- **[262] `qa_unequal_k_ccr_invariant_cert_v1`** (Mannheim side, MC-1 / MC-2). Propagator definition and verified properties:

  ```
  i∆_QA(Δk; (b, e), (b', e')) := 1 if T^|Δk|(b, e) = (b', e') else 0.
  ```

  Integer-valued, slice-independent. Equal-`k` limit `i∆_QA(0; w, w') = δ_{w, w'}` verified exhaustively over `{1, …, 9}² × {1, …, 9}²` (all 6561 pairs). Cross-orbit propagator vanishes. Periodicity at the full-propagator level for each witness. The Lehmann-type trace formula

  ```
  Tr i∆_QA(Δk) = Σ_O |O| · 1[period(O) | Δk]
               = 72 · 1[24 | Δk] + 8 · 1[8 | Δk] + 1
  ```

  verified at `Δk ∈ {0, 1, 2, 3, 4, 6, 8, 12, 16, 24}`. The Singularity contributes 1 at every `Δk` (period 1 divides everything); the Satellite adds 8 when 8 divides `Δk`; the Cosmos adds 72 when 24 divides `Δk`. Witnesses: Fibonacci Cosmos (1, 1), Tribonacci Satellite (3, 3), Ninbonacci Singularity (9, 9) — one per orbit class.

The stipulated equal-`k` δ-function CCR is an observer-side canonical encoding (cross-map §4.1 precision guardrail); neither cert derives it from QA primitives.

**Explicit limits of the cert-backed mapping.**

- Both certs are scoped to m = 9 and the Fibonacci-like step `T_F(b, e) = (a1(b + e), b)`. m = 24 needs a mixed I-level + period-n constraint family on the cert-D side (Cassini² is invariant but too coarse at m = 24, with only four distinct values across 30 orbits), and a six-term Lehmann trace on the cert-C side (period set `{1, 3, 6, 8, 12, 24}`).
- Cert-D covers only the +1 branch of the I = 1 Cosmos orbit. The −1 branch, the I ∈ {4, 7} Cosmos orbits, the Satellite, and the Singularity each admit analogous constructions with different constraint-function choices; none is in v1.
- Alternative step operators (Lucas, Tribonacci, other QA generators) each define their own orbit structure. Slice-independence *across* the choice of step operator is a deeper invariant and not addressed here.
- The interacting Lehmann representation — Mannheim's `ρ(σ²)` integral form eqs (8.9) / (8.16) — corresponds on the QA side to a non-trivial orbit-period spectrum that is not captured by the free-case discrete formula above. A QA interacting analog is open work.

These are future-cert extensions. The mapping claims in §3 and §4 stand as certified structural analogues on the scope stated.

---

## 7. References

- Mannheim, P.D. *Equivalence of light-front quantization and instant-time quantization.* Phys. Rev. D 102, 025020 (2020). arXiv:1909.03548.
- Blaschke, D.N., Gieres, F. *On the canonical formulation of gauge field theories and Poincaré transformations.* Nucl. Phys. B 965 (2021). arXiv:2004.14406.
- Dirac, P.A.M. *Lectures on Quantum Mechanics.* Belfer Graduate School of Science, Yeshiva University, New York (1964).
- Dale, W. *Quantum Arithmetic: Observer Projection Compliance Specification.* docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md.
- Dale, W. (with Briddell, D.) *A Quantum Arithmetic Completion Layer for Field Structure Theory.* Frontiers in Physics, Nuclear Physics, manuscript 1850870, submitted 2026-04-08.
- *QA Tiered Reachability Theorem.* Certificate family [191], `qa_alphageometry_ptolemy/`.
- *QA Orbit-Dirac Bracket Cert v1.* Certificate family [260], `qa_alphageometry_ptolemy/qa_orbit_dirac_bracket_cert_v1/`. Delivers MC-3 / MC-4.
- *QA Unequal-k CCR Invariant Cert v1.* Certificate family [262], `qa_alphageometry_ptolemy/qa_unequal_k_ccr_invariant_cert_v1/`. Delivers MC-1 / MC-2.
- *QA Cross-Map Notes for this section.* `docs/theory/QA_QFT_ETCR_CROSSMAP.md`; audit `docs/theory/QA_QFT_COMMUTATORS_PRIOR_ART.md`.

---

**Draft status**: cert-backed synthesis pass. Certs [260] + [262] shipped 2026-04-21. Review-ready; polish pass 2026-04-22.
