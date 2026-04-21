# QA ↔ QFT ETCR Cross-Map — Mannheim 2020 + Blaschke-Gieres 2021

**Status**: Primary-source-grounded mapping claim. Pre-cert. Supersedes §5 of `QA_QFT_COMMUTATORS_PRIOR_ART.md` with explicit equations.

**Thesis**:
- Mannheim's unequal-time-commutator-as-invariant result (Phys. Rev. D 102, 025020, 2020) provides a QFT-side **structural analogue** of the Theorem NT distinction: equal-time relations depend on the chosen slicing, while the invariant object is the unequal-time dynamical commutator. QA path-time `k` is the natural home for that invariant. This is **structural convergence, not identity** — Mannheim is a canonical-quantization result, not a derivation of the QA firewall.
- Blaschke & Gieres's Dirac-bracket-replaces-Poisson-bracket result (Nucl. Phys. B 965, 2021) provides a **structural analogue** of the orbit quotient that QA carries via orbit structure ([191]). Dirac-bracket machinery gives the precise formula; QA orbits give the precise quotient. Again: structural analogue, not derivation.

**Companion**: `QA_QFT_COMMUTATORS_PRIOR_ART.md` (audit of Aug 2025 QA scaffolding).

---

## 1. Mannheim — Verbatim Claim-Set

Source: Mannheim, "Equivalence of light-front quantization and instant-time quantization," PRD 102 025020, arXiv:1909.03548.

### 1.1 Instant-time ETCR (eq 1.2)

Free scalar field, action `I_S = ∫ d⁴x ½[(∂₀φ)² − (∂ᵢφ)² − m²φ²]`:

```
[φ(x⁰,x¹,x²,x³), ∂₀φ(x⁰,y¹,y²,y³)] = i δ(x¹−y¹) δ(x²−y²) δ(x³−y³)     (1.2)
[φ(x⁰,x¹,x²,x³),    φ(x⁰,y¹,y²,y³)] = 0                               (1.6)
```

### 1.2 Light-front ETCR (eq 1.4)

Same field, light-front coords `x± = x⁰ ± x³`, action

```
I_S = ½ ∫ dx⁺ dx¹ dx² dx⁻ ½[2∂₊φ∂₋φ + 2∂₋φ∂₊φ − (∂ᵢφ)² − m²φ²]      (1.3)
```

ETCR:

```
[φ(x⁺,x¹,x²,x⁻), 2∂₋φ(x⁺,y¹,y²,y⁻)] = i δ(x¹−y¹) δ(x²−y²) δ(x⁻−y⁻)  (1.4)
```

Mannheim: "(1.4) and (1.2) not only cannot be transformed into each other, they take completely different forms."

### 1.3 Fermion anticommutators (eqs 1.8, 1.14)

Instant-time:

```
{ψ_α(x⁰,x⃗), ψ†_β(x⁰,y⃗)} = δ_αβ δ³(x⃗−y⃗)                             (1.8)
```

Light-front (with projector `Λ± = ½γ⁰γ±`, obeying `(Λ±)² = Λ±`, `Λ+Λ− = 0`):

```
{[ψ⁽⁺⁾]_α(x⁺,x⃗), [ψ†⁽⁺⁾]_β(x⁺,y⃗)} = Λ+_αβ δ(x⁻−y⁻) δ²(x⃗⊥−y⃗⊥)      (1.14)
```

The `Λ±` are non-invertible projection operators. Light-front bad fermions `ψ⁽⁻⁾` are constrained, not dynamical.

### 1.4 Unequal-time commutator as c-number invariant (eq 2.2)

Using Fock expansion (2.1), the unequal instant-time commutator between two free scalar fields is:

```
i∆(IT; x) = [φ(x⁰,x⃗), φ(0)]
         = ∫ d³p / [(2π)³ 2E_p] × (e^{−i(E_p x⁰+p⃗·x⃗)} − e^{i(E_p x⁰+p⃗·x⃗)})   (2.2)
```

**Key**: this is a c-number (not q-number) and contains (1.2) and (1.6) as its equal-time limits.

### 1.5 Lehmann representation — all-order equivalence (eqs 8.9, 8.16)

Interacting theory, only requirements: Poincaré invariance, Hermitian `P_μ`, unique vacuum. Instant-time:

```
i∆(IT,FULL; x−y) = ∫₀^∞ dσ² ρ(σ², IT) · i∆(IT,FREE; x−y, σ²)          (8.9)
```

Light-front:

```
i∆(LF,FULL; x−y) = ∫₀^∞ dσ² ρ(σ², LF) · i∆(LF,FREE; x−y, σ²)          (8.16)
```

Mannheim constructs a unitary `U` with `U|Ω(IT)⟩ = |Ω(LF)⟩`, `Uφ(IT;0)U⁻¹ = φ(LF;0)`, establishing unitary equivalence to all orders (eq 8.26).

### 1.6 Lorentz-invariant operator identity (eq 9.12)

To all orders:

```
P₀²(IT) − P₁²(IT) − P₂²(IT) − P₃²(IT) = 4P₊(LF)P₋(LF) − P₁²(LF) − P₂²(LF)   (9.12)
```

On any `p² = m²` eigenstate, both sides evaluate to `m²`.

---

## 2. Blaschke-Gieres — Verbatim Claim-Set

Source: Blaschke & Gieres, "On the canonical formulation of gauge field theories and Poincaré transformations," NPB 965, arXiv:2004.14406.

### 2.1 Constraint taxonomy — Maxwell radiation gauge (eq 5.32)

```
First-class constraints (FCC):
  φ₁ ≡ π⁰ ≈ 0           (primary)
  φ₂ ≡ ∂ᵢπⁱ ≈ 0         (Gauss law, secondary)

Gauge-fixing conditions (make above second-class):
  f₁ ≡ A⁰ ≈ 0
  f₂ ≡ div A⃗ ≈ 0                                                       (5.32)
```

Collective: `(ϕ_a)_{a=1..4} = (φ₁, φ₂, f₁, f₂)`.

### 2.2 Dirac-bracket formula (eq 5.37) — THE CORE RESULT

Matrix `X_ab ≡ {ϕ_a, ϕ_b}` (Poisson brackets of all constraints), invertible on the physical subspace Γ_r. For any two functions F, G on phase space:

```
{F, G}_D ≡ {F, G} − {F, ϕ_a} (X⁻¹)^{ab} {ϕ_b, G}                       (5.37)
```

Algebraic properties preserved: bilinearity, antisymmetry, Jacobi, derivation.

Strong-zero property:

```
{ϕ_a, F}_D = 0     for any F                                            (5.38)
```

Consequence: constraints become strong equalities after switching to Dirac brackets. In quantum theory, `{,}_D → [,]/(iℏ)`.

### 2.3 Poisson-bracket failure + Dirac-bracket rescue (eq 5.39)

Blaschke-Gieres's central claim: canonical `P^μ` does NOT generate translations via Poisson bracket in general gauge. But with Dirac bracket:

```
δ_a ϕ(x) ≡ {ϕ(x), a_μ P^μ}_D = a_μ ∂^μ ϕ(x)                            (5.39)
```

for all `ϕ ∈ {A⁰,...,A³, π⁰,...,π³}`. Translations are recovered.

---

## 3. QA Vocabulary

Anchors used in the cross-map:

- **A1**: QA states in `{1,...,m}`, never `{0,...,m−1}`. The zero residue never labels a state.
- **A2**: `d = b+e`, `a = b+2e` — always derived.
- **T2 (Theorem NT)**: Observer projection firewall. Continuous (ℝ, ℂ) amplitudes are observer-side. Boundary is crossed exactly twice (input, output). No float → int cast inside QA logic.
- **T1**: QA time = integer path length `k`. Discrete path, never continuous.
- **Orbits**: Singularity (1-cycle at (9,9)), Satellite (8-cycle, 8 pairs), Cosmos (24-cycle, 72 pairs) for mod-9. Orbits quotient the state space into equivalence classes.
- **[191] Tiered Reachability Theorem**: on S_9, Level-I ceiling 26% — structural bound on what can be reached within the mod-9 orbit constraint.

---

## 4. The Cross-Map

### 4.1 ETCR = observer projection at fixed k; unequal-k path = QA invariant

The Mannheim structure:

| QFT object | Mannheim status | QA-layer analog |
|---|---|---|
| Instant-time slice `x⁰ = const` | choice of quantization surface | choice of observer-projection cut at `k = k₀` |
| ETCR eq (1.2) | slice-dependent, differs from LF | observer-projected CCR at fixed `k` |
| Light-front slice `x⁺ = const` | different quantization surface | alternate observer-projection cut |
| ETCR eq (1.4) | slice-dependent, differs from IT | observer-projected CCR at alternate cut |
| Unequal-time `i∆(x−y)` eq (2.2) | c-number, slice-independent | orbit-traced path invariant under QA T-operator |
| Lehmann spectral density `ρ(σ²)` eqs (8.9, 8.16) | all-order invariant | QA spectral decomposition over orbit classes |
| Unitary `U` eq (8.26) | rotates IT ↔ LF | observer-reprojection operator between cuts |
| Operator identity eq (9.12) | Lorentz-invariant to all orders | orbit-class-invariant structural identity |

**Precision guardrail (before MC-1)**: The equal-`k` CCR below is a **proposed observer-side canonical encoding** on top of QA state labels `(b,e)`, not a derived QA theorem. We have NOT derived canonical commutation from QA orbit dynamics; we have chosen to *posit* it at the observer boundary by analogy with bosonic QFT. This means cert-C (see §5) must either (a) accept this encoding as a stipulated boundary condition and only certify its unequal-`k` invariant, or (b) derive the CCR from more basic QA primitives. As of this doc, path (a) is the working plan; path (b) is open work.

**Mapping claim (MC-1)**: Given the proposed encoding, for QA creation/annihilation operators `â_{b,e}`, `â†_{b',e'}` indexed on pair `(b,e) ∈ {1,...,m}²`, the relation

```
[â_{b,e}, â†_{b',e'}]_{k=k₀} = δ_{b,b'} δ_{e,e'}        (observer projection at fixed k)
```

is a slice-dependent observer projection. The path-invariant object is the unequal-`k` propagator

```
i∆_QA(k₁, k₂; (b,e) ↦ (b',e')) = ⟨Ω | â_{b',e'}(k₂) â†_{b,e}(k₁) | Ω⟩_path  −  (reverse ordering)
```

computed by tracing the orbit trajectory from `(b,e)` at path-time `k₁` to `(b',e')` at path-time `k₂` under the T-operator. This is QA's structural counterpart to Mannheim's `i∆(IT; x−y)`.

**Mapping claim (MC-2)**: The Aug 2025 prior art (C10) gives only the slice-dependent CCR (left-hand side). The unequal-`k` invariant is missing. Cert-C for bosonic CCR is therefore incomplete as drafted; it needs the unequal-`k` propagator as the primary object. The proposed encoding flagged in the precision guardrail must also be declared explicitly in cert-C's mapping protocol.

### 4.2 Dirac bracket = commutator modulo orbit-class constraints

The Blaschke-Gieres structure:

| QFT object | Blaschke-Gieres status | QA-layer analog |
|---|---|---|
| Poisson bracket `{F, G}` | gauge-redundant on general gauge | raw commutator on unquotiented pair space |
| First-class constraint `φ_a ≈ 0` | generates gauge redundancy | orbit-class membership condition |
| Gauge-fixing `f_a ≈ 0` | selects one representative per gauge orbit | selects canonical representative `(b,e)` per QA orbit |
| Matrix `X_ab = {ϕ_a, ϕ_b}` | invertible on Γ_r | structural invertibility on reachability tier |
| Dirac bracket `{F, G}_D` eq (5.37) | physical-subspace-respecting | orbit-class-respecting commutator |
| Strong-zero `{ϕ_a, F}_D = 0` eq (5.38) | constraints vanish exactly | orbit-class labels become strong invariants |
| Translation recovery eq (5.39) | `{ϕ, a·P}_D = a·∂ϕ` | path-time T-operator acts correctly after orbit-quotient |

**Mapping claim (MC-3)**: QA orbits are the substrate for a QA Dirac bracket. Concretely, let

- `ℋ_raw = ℤ/mℤ × ℤ/mℤ` (pair space, with A1 exclusion)
- `𝒪 = set of orbit classes` (Singularity, Satellite, Cosmos for m=9)
- `q : ℋ_raw → 𝒪` the orbit-class projection

**Critical precision — constraint FUNCTIONS, not labels**: In Dirac's original setting, `ϕ_a ≈ 0` means `ϕ_a` is a function on phase space whose vanishing defines the constraint surface. Orbit-class names ("Singularity", "Satellite", "Cosmos") are string labels, not functions, and cannot be plugged into eq (5.37). Before cert-D, we must define actual **QA constraint functions** whose vanishing cuts out either a canonical orbit representative or an orbit-invariant subspace.

Candidate constraint functions (to be validated in cert-D):

| Target subspace | Candidate constraint function | Vanishes iff |
|---|---|---|
| Singularity (fixed point) for mod-9 | `ϕ_sing(b,e) = (b−9)·(b−9) + (e−9)·(e−9)` (over ℤ, not ℤ/9ℤ) | `(b,e) = (9,9)` |
| Fixed-point set of T | `ϕ_T(b,e) = T(b,e) − (b,e)` (vector-valued) | `(b,e)` is T-fixed |
| Period-dividing-n set | `ϕ_n(b,e) = T^n(b,e) − (b,e)` | orbit period divides n |
| Orbit-invariant level set | `ϕ_I(b,e) = I(b,e) − I₀` where I is a T-invariant scalar | `(b,e)` in level set `I = I₀` |

The last row is the cleanest: if we can find a non-trivial T-invariant function `I(b,e)` (e.g., a Casimir-like scalar on the pair space), its level sets foliate ℋ_raw into orbit-invariant layers, and `ϕ_I = I − I₀` is a proper constraint function in Dirac's sense.

**Preliminary computational support (2026-04-20, NOT cert-validated)**. Under the Fibonacci-like QA step `T_F(b, e) = (a1(b+e), b)` with `a1(x) = ((x−1) mod m) + 1`, the Cassini-squared scalar

```
I(b, e) := (b² − be − e²)²  mod m
```

is T-invariant. Mechanism: `b² − be − e²` flips sign under `T_F` (classical Cassini identity `F_{n+1} F_{n-1} − F_n² = (−1)^n` in pair-space form), so its square is preserved. Verification on `{1,...,m}²` for m = 9 and m = 24 — see `docs/theory/empirical/etcr_t_invariant_check.py`:

- **m = 9**. Orbit structure: 3 Cosmos (period 24) + 1 Satellite (period 8) + 1 Singularity (period 1) = 81 pairs. `I` separates the three Cosmos orbits (I ∈ {1, 4, 7}); Singularity and Satellite share `I = 0` and require an auxiliary constraint (e.g. the fixed-point constraint `ϕ_T` above) to separate.
- **m = 24**. 30 orbits across periods {1, 3, 6, 8, 12, 24}. `I` is invariant with 4 distinct values, coarser than full orbit separation; cert-D will need a mixed family — `I`-level constraints plus period constraints `ϕ_n(b, e) = T^n(b, e) − (b, e)` inside each level set.

**Scope of the result**: the invariance depends on the specific choice `T_F(b, e) = (a1(b+e), b)`. Other QA step operators may need re-checking. The computational existence of `I` upgrades MC-3 from "speculative construction" to "viable construction with identified invariant plus auxiliary constraints," but does NOT constitute cert-D — full validation requires the `X`-matrix computation, invertibility check, and strong-zero verification listed in MC-4.

Given a chosen family `{ϕ_a}` of such constraint functions, the **QA Dirac bracket** is

```
[F, G]_orbit ≡ [F, G] − [F, ϕ_a] (X⁻¹)^{ab} [ϕ_b, G]       (QA analog of 5.37)
```

where `[,]` is an unquotiented bracket on pair space (candidate: the tuple wedge `{(b₁,e₁),(b₂,e₂)} = b₁e₂ − b₂e₁`, i.e., prior-art C3 recast as Poisson structure), and the `X`-matrix is `X_{ab} = [ϕ_a, ϕ_b]` in that bracket. The Aug 2025 prior art (C2, C3) proposed commutators without the constraint-quotient term; those are `[F, G]` not `[F, G]_orbit`. Observable/physical commutators must use `[F, G]_orbit`.

**Mapping claim (MC-4)**: This is the substrate for cert-D. The orbit machinery already exists in `qa_alphageometry_ptolemy/` and [191]. What's needed is:

1. Explicit construction of a constraint-function family `{ϕ_a}` (not labels), following the candidates above. A T-invariant scalar `I(b,e)` — if one exists — is the cleanest route.
2. Choice of the unquotiented bracket `[,]` on pair space. Default: tuple wedge from C3.
3. Computation of the `X`-matrix `X_{ab} = [ϕ_a, ϕ_b]`.
4. Invertibility check of `X` on the physical subspace (parallels Blaschke-Gieres eq 5.36).
5. The Dirac-bracket formula instantiated for a simple observable (e.g. `F = b`, `G = e`).
6. Verification that `[ϕ_a, F]_orbit = 0` for any `F` — constraint functions as strong invariants (parallels eq 5.38).

Preliminary computational support for (1) is in hand for the Fibonacci-like step `T_F` — see the Cassini-squared paragraph above. That upgrades (1) from "speculative" to "viable for the chosen dynamics." Items (2)–(6) remain open and are the primary cert-D deliverables.

### 4.3 Consequences for prior-art scaffolding

Using the cross-map to re-audit `QA_QFT_COMMUTATORS_PRIOR_ART.md` §3:

- **C2 (projective commutator, ill-posed)**: Replace with orbit-Dirac bracket from MC-3. The scalar-commutator-on-ℤ/mℤ collapse disappears because the antisymmetric structure now lives in the constraint-quotient correction, not in the bare ring operation.
- **C3 (tuple wedge)**: Correctly identified as a symplectic form, not a commutator. Its role in the cross-map is as the Poisson-bracket analog feeding eq (5.37): `{(b₁,e₁), (b₂,e₂)} = b₁e₂ − b₂e₁` is exactly the Poisson structure on the `(b,e)` phase space, before orbit-quotient.
- **C4 (field strength)**: The `[A_x, A_t]` term in `F_xt` uses the wedge. The QA field strength on orbit-quotiented space should replace `[A,A]` with `[A,A]_orbit`. The Aug 2025 expression is the unquotiented version.
- **C10 (CCR)**: See MC-1 and MC-2 above. Slice-dependent; missing the unequal-k invariant.
- **C11 (Hamiltonian)**: `U = exp(iθH_φ)` acts on observer-layer amplitudes `ψ ∈ ℂ^N` (cyclotomic `ℚ(ζ_m)` in the limit). The QA-layer analog is the `k`-step T-operator on `(b,e)` pairs, with the cyclotomic phase arising only at the observer boundary. This is the T2-b resolution noted in the audit §4b.

---

## 5. Cert-Family Candidates Refined

Based on the cross-map, two candidates have primary-source backing:

### cert-D — QA Orbit-Dirac Bracket (HIGHEST READINESS)

**Generator**: Blaschke-Gieres eq (5.37).
**QA substrate**: [191] tiered reachability + orbit classifier.
**Deliverable**: explicit computation of `[F, G]_orbit` for a minimum pair, verification of strong-zero on orbit-class labels, invertibility check on `X`-matrix.
**Axiom coverage**: A1 (orbits exclude zero-state), A2 (derived coordinates preserved), T1 (path-time k as invariant), T2 (no continuous amplitude in construction).
**Paper value**: cleanly formalizes one half of the convergence.
**Risk**: low. Machinery exists. Computational path clean.

### cert-C — QA Unequal-k CCR Invariant (HIGHER VALUE, HIGHER RISK)

**Generator**: Mannheim eq (2.2) + Lehmann representation (8.9, 8.16).
**QA substrate**: QA creation/annihilation from C10 + path-time k from T1.
**Deliverable**: explicit unequal-k propagator `i∆_QA(k₁, k₂; (b,e) → (b',e'))` as orbit-trajectory sum. Proof that equal-k limit recovers C10's δ-function CCR. Sketch of QA Lehmann representation as sum over orbit classes weighted by class-density.
**Axiom coverage**: T1 (path-time central), T2 (slice = observer projection), NT (invariant lives in path structure).
**Paper value**: directly encodes Mannheim's result in QA vocabulary.
**Risk**: higher. Needs explicit construction of the unequal-k propagator, which is new computational territory. Cert-D should ship first.

**Sequencing**: cert-D first; cert-C after cert-D is stable.

---

## 6. Paper Section Outline (FST Part 3 candidate)

Target venue: continuation of the Briddell-Dale FST lineage (Frontiers manuscript 1850870 submitted 2026-04-08). Single section, ~5–8 pages.

Working title: **"ETCR as Observer Projection, Dirac Bracket as Orbit Quotient: Two QFT-Literature Confirmations of QA's Structural Firewall"**

Draft structure:

1. **Introduction** (0.5p). Frame: QFT has two known facts — (a) ETCRs are quantization-scheme dependent; unequal-time is invariant [Mannheim 2020], (b) canonical gauge bracket is Poisson-inadequate; Dirac-bracket construction is required [Blaschke-Gieres 2021]. Both are known but not connected. QA predicts the connection: both are instances of the same principle — observer-projection at a fixed slice is not the physical object; the path / orbit-quotient structure is.
2. **QA primitives** (0.5p). A1, A2, T1, T2, orbits, [191]. One paragraph each.
3. **Mannheim mapping** (1.5p). §1 of this doc, condensed. ETCR as observer-projection at fixed `k`; unequal-k propagator as path invariant; Lehmann representation as orbit-class decomposition.
4. **Blaschke-Gieres mapping** (1.5p). §2 of this doc, condensed. Dirac bracket as orbit-Dirac bracket; strong-zero property via orbit-label invariance; translation recovery via T-operator on quotiented pair space.
5. **Unified claim** (1p). Theorem NT (observer projection firewall) predicts both results. Equal-time slices and ordinary Poisson brackets are both "observer-layer" constructs that violate the firewall when treated as physical. Unequal-time path structure and Dirac brackets are the boundary-respecting versions.
6. **Discussion** (0.5p). Limits: mapping is claim, not theorem (until certs D and C ship). Relation to FST Part 2 (Briddell-Dale). Prediction: gauge-field light-front zero-mode singularities (Mannheim §3) should disappear under orbit-class treatment — testable.
7. **References**: Mannheim 2020, Blaschke-Gieres 2021, Dirac (1964 lectures), QA axioms spec, [191].

Scope is tight. No new math beyond the mapping; no cert dependency (certs backfill the paper later). Writable in a session.

---

## 7. Next Moves

1. Will review: is this cross-map sound? Any mapping claim over-stated?
2. If approved: draft the paper section at `papers/in-progress/qft-etcr-orbit-quotient/section.md` or similar. FST Part 3 framing.
3. After paper draft: scaffold cert-D under `qa_alphageometry_ptolemy/qa_orbit_dirac_bracket_cert_v1/`.
4. After cert-D stable: scaffold cert-C (unequal-k CCR invariant).

No code run, no cert registered, no files beyond this doc changed. Session `exp-etcr-qa` remains open.
