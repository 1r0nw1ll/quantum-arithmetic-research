# QA-QFT Commutators — Prior-Art Audit

**Status**: PRIOR ART, pre-cert, uncertified. Material below is exploratory scaffolding from Aug 2025 chat exports — cited verbatim for audit. Nothing here has been cert-gated, axiom-linted, or grounded against primary-source QFT literature. This doc exists to recover what is already drafted before writing anything new.

**Trigger**: 2026-04-20 research thread on equal-time commutation relations (ETCRs). Before drafting a cert + paper, we audited OB and QAnotes and found ~six Aug 2025 chat entries already proposing QA commutator algebra, field strength tensor, Hilbert analog, and CCR structure. Those drafts need to be consolidated and grounded before any certification.

**Scope**: Canonical commutation / anticommutation relations in the QA layer and at the QA → observer boundary. Out of scope here: E8 / Hopf / TUFT mappings (separate families), SU(3) Yang-Mills mass gap exploration (covered in `private/QAnotes/Nexus AI Chat Imports/2025/05/Quantum Arithmetic Gauge Theory.md`, referenced §2).

---

## 1. Provenance

Primary sources for the prior-art (chat exports, uncertified):

- `private/QAnotes/Nexus AI Chat Imports/2025/08/Quantum arithmetic overview.md` (2975 lines). Chat session 08/17/2025 14:54 – 08/18/2025 16:23. Key line ranges: 509–630 (QA gauge symmetries), 699–720 (commutator visualization), 1020–1102 (field strength tensor), 1128–1160 (F_xt tensor definition), 1176–1217 (SU(2) triplet extension), 1481–1544 (Noether + Hilbert + CCR), 1560–1598 (Fock ladder).
- `private/QAnotes/Nexus AI Chat Imports/2025/05/Quantum Arithmetic Gauge Theory.md` (397 lines). Full Yang-Mills paper draft: SU(3), mass gap, θ-vacuum, lattice comparison. Referenced, not re-audited here.
- `private/QAnotes/Nexus AI Chat Imports/2025/05/Gizmo Naming Explanation.md` (SU(2)/SU(3) Gell-Mann/Pauli tuple overlays; has `qa_commutator` function emulating Lie bracket).
- Open Brain entries 2026-03-08 (block from Obsidian chat import): 4601890e, 28304653, e588436d, e97cdee8, 01b0f309, ab871234 — six thoughts captured from the same Aug 2025 session.

Secondary (for cross-reference, not audited here):

- `private/QAnotes/Nexus AI Chat Imports/2025/09/QA TUFT Hopf Mapping.md`
- `private/QAnotes/Nexus AI Chat Imports/2025/09/QA Hopf fibration exploration.md`
- `private/QAnotes/Nexus AI Chat Imports/2025/09/SU(3) options layer.md`

---

## 2. Claims Inventory

Each claim quoted verbatim with source line. No re-interpretation; red-team notes are deferred to §4.

### C1. QA gauge symmetry set (four types)

Source: overview.md:619–625. Transcribed:

| Type | Transform | Preserves |
|---|---|---|
| Scaling | `(b,e) → λ(b,e)` | Vector-space laws |
| Modular Shift | `(b,e) → (b+m, e+n) mod 24` | Canonical tuple laws mod-24 |
| Projective Class | `(b,e,d,a) ~ (λb, λe, λd, λa)` | Geometry on icositetragon |
| Commutator Algebra | Nested conjugations | QA gauge group candidate |

Claimed target group: subset of `GL_2(ℤ/24ℤ)` (overview.md:615).

### C2. Projective conjugation commutator

Source: overview.md:607–613. Verbatim:

```
[C_{λ₁}, C_{λ₂}] := C_{λ₁λ₂} − C_{λ₂λ₁}
```

"If we allow λ ∈ ℤ/24ℤ with modular inversion, we can define a non-trivial commutator algebra: closed under composition."

### C3. Tuple commutator (wedge/determinant form)

Source: overview.md:1049–1052, repeated 1141. Verbatim:

```
[(b₁, e₁), (b₂, e₂)] := (b₁ e₂ − b₂ e₁) mod 24
```

Used as `[A_x, A_t] = b_x e_t − b_t e_x`. This is the 2×2 determinant / wedge product of two tuples, returning a scalar in ℤ/24ℤ.

### C4. QA field strength tensor (non-Abelian form)

Source: overview.md:1042–1063, 1128–1142. Verbatim:

```
F_xt^(QA) = Δ_x A_t − Δ_t A_x + [A_x, A_t]
```

Components emit a tuple:

```
F_xt^tuple = (δb, δe, δd = δb + δe, δa = δb + 2δe)
```

with the commutator feeding the `δa` channel (1142): `δa = δb + 2δe + [A_x, A_t] mod 24`.

Discrete directional operators (1036):

```
Δ_x ψ = ψ(x+1, t) − ψ(x, t)
Δ_t ψ = ψ(x, t+1) − ψ(x, t)
```

### C5. Symbolic Yang-Mills action

Source: overview.md:1083–1088. Verbatim:

```
S = Σ_{x,t} Tr(F_xt²) = Σ_{x,t} (δa)² mod 24
```

Rationale: "symbolic curvature energy; lattice-based gauge tension; source of QA field self-interactions."

### C6. SU(2) triplet extension

Source: overview.md:1183–1201. Verbatim:

```
F_SU(2) = Σ_i F^(i) + Σ_{i<j} [ψ_i, ψ_j]
[ψ_i, ψ_j] = b_i e_j − b_j e_i mod 24
```

Three QA tuple fields ψ_1, ψ_2, ψ_3 at each lattice point; triplet basis.

### C7. QA Noether conservation (empirical)

Source: overview.md:1457–1493. Claim:

```
Δ_t J^0 + Δ_x J^1 ≡ 0 (mod 24)   at nearly all grid points
```

"Verifies a QA analog of local charge conservation and Noether's theorem in a discrete symbolic modular field." Empirical status only — "nearly all" is not "all" and no proof is attempted.

### C8. QA Hilbert space analog

Source: overview.md:1503–1509. Verbatim:

```
ℋ_QA := { ψ : ℤ_N^d → ℚ^(b,e,d,a) }
```

"Functions from lattice points to QA tuples. No continuous amplitudes — all symbolic."

### C9. Modular inner product

Source: overview.md:1516–1522. Verbatim:

```
⟨ψ, φ⟩ = Σ_{x ∈ Λ} ψ*(x) · φ(x) mod 24
ψ* = (b, −e, d, −a) mod 24
```

### C10. Creation / annihilation CCR

Source: overview.md:1526–1534. Verbatim:

```
â†_{b,e} : adds tuple (b,e)
â_{b,e}  : removes tuple (b,e)

[â_{b,e}, â†_{b',e'}] = δ_{b,b'} δ_{e,e'} mod 24
```

This is the direct bosonic CCR, lifted to ℤ/24ℤ indexing, with Kronecker deltas on both b and e.

### C11. Hamiltonian with cyclotomic phase

Source: later session in same file (OB e97cdee8, from overview.md line ~1850+). Verbatim structure:

```
A_φ(u,v) = e^{2πi r/24},    r = (x' + 2t' − (x + 2t)) mod 24
H_φ = D^{−1/2} A_φ D^{−1/2}
U = exp(i θ H_φ)
```

Phases restricted to 24th roots of unity — i.e. cyclotomic integers ℚ(ζ_24). Tests prescribed: `‖U†U − I‖_F < 10^{-12}`; norm conservation over 100 steps `|‖ψ_t‖_2 − 1| < 10^{-12}`.

---

## 3. Relation to Already-Certified QA Infrastructure

None of C1–C11 are presently cert-gated under `qa_alphageometry_ptolemy/`. Grep on the meta-validator returns no commutator / ETCR family. Closest adjacencies:

- **[234] Pythagorean Identity** (`C² + F² = G²` with `C = 2de`, `F = ba`, `G = e² + d²` RAW): the bilinear cross-pair `C_i C_j + F_i F_j ≡ G_i G_j mod m` surfaced in [256] QA Orbit-Resonance Attention is structurally identical to a symmetric form on tuple pairs. C3's antisymmetric form (`b_1 e_2 − b_2 e_1`) is the complementary wedge.
- **[191] Tiered Reachability Theorem** on S_9: if orbits are the physical state space, C1's "projective class" row is the Level-I tier quotient.
- **[214] QA Generator Patterns** (Fibonacci/Lucas/Tribonacci/Phibonacci/Ninbonacci): these are generator-family labels on (b,e) pairs. C10's â†_{b,e} creation indexing would need to be compatible with orbit-class invariants to avoid creating states outside the admissible orbit.

---

## 4. Axiom Check (A1, A2, T2, S1, S2, T1)

Per CLAUDE.md HARD GATE.

| Claim | A1 (no 0) | A2 (d,a derived) | T2 (firewall) | S1 (no **2) | S2 (int/Frac state) | T1 (path-time k) |
|---|---|---|---|---|---|---|
| C1 scaling | ℤ/24ℤ includes 0 — violation if states are ℤ/24ℤ; safe if states {1..24} | d,a derived OK | N/A | N/A | OK | N/A |
| C1 mod-shift | same 0-concern | OK | N/A | N/A | OK | N/A |
| C1 projective | same | OK | N/A | N/A | OK | N/A |
| C1 commutator | **ill-posed, see §4a** | — | — | — | — | — |
| C2 conj. comm. | **ill-posed, see §4a** | — | — | — | — | — |
| C3 tuple wedge | determinant can be 0 (legitimate, not state) | uses b,e only | N/A | N/A | OK | N/A |
| C4 F_xt | depends on above | uses b,e,d,a correctly | **uses Δ on ψ — Δ is discrete OK** | no **2 used | OK (integers) | lattice (x,t) — `t` here is lattice coord not continuous time, acceptable if classified as QA path-time |
| C5 S = Σ(δa)² | uses (δa)² = δa·δa — **S1 flag** | derived OK | ambiguous Tr | written as `(δa)²` — must be `δa·δa` | OK | same |
| C6 SU(2) triplet | inherits | OK | N/A | N/A | OK | same |
| C7 Noether | empirical mod 24 | OK | N/A | N/A | OK | lattice time OK |
| C8 ℋ_QA | codomain is ℚ^(b,e,d,a) — zero allowed as a coefficient is fine; zero as a (b,e) state is an A1 question | — | **C amplitudes absent — pure QA — OK** | N/A | ℚ-valued, Fractions OK | N/A |
| C9 inner product | `−e` mod 24: if e=0 was admissible it'd pin, but A1 excludes e=0, so −e ∈ {23,…,1}, OK | — | dot product lives in ℤ/24ℤ, pure QA | N/A | OK | N/A |
| C10 CCR | Kronecker δ on (b,e) indexing — if indices run {1..24}² this is clean; if ℤ/24ℤ×ℤ/24ℤ including (0,·), A1 flag | — | **â, â† are operators on ℋ_QA → remains QA interior; no continuous amplitude** | N/A | OK | N/A |
| C11 Hamiltonian | phase e^{2πi r/24} — complex amplitude — **T2-b risk**: if ψ is the QA state, H_φ acting via complex phase puts QA state into C. This crosses the firewall. Must be reclassified: ψ is OBSERVER-layer amplitude; QA state is (b,e,d,a). Then H_φ is observer-layer dynamics, and the QA-layer analog is the integer path k that parameterizes U = exp(i θ H_φ) stepwise. | — | **flag — see §4b** | no `**` but e^{…} is an exp — continuous function used for complex phase OK at observer layer, NOT at QA layer | ψ is complex in code, Q(ζ_24) conceptually — boundary case | **T1 OK if θ is ℕ step index, not continuous** |

### 4a. Projective conjugation commutator is ill-posed

C2 defines `[C_{λ₁}, C_{λ₂}] := C_{λ₁λ₂} − C_{λ₂λ₁}`. For scalar λ ∈ ℤ/24ℤ, multiplication is commutative: `λ₁ λ₂ = λ₂ λ₁` always. So `C_{λ₁ λ₂} = C_{λ₂ λ₁}` and the commutator is identically zero. The formulation as written cannot yield a non-trivial commutator algebra.

Two repairs available:

1. Replace scalars with 2×2 matrices in `GL_2(ℤ/24ℤ)` acting on `(b, e)^T`. Matrix multiplication is non-commutative and yields a genuine Lie-bracket analog after subtraction.
2. Interpret `C_λ` as conjugation by a ring element and take the group-theoretic commutator `C_{λ₁} C_{λ₂} C_{λ₁}^{-1} C_{λ₂}^{-1}`, not difference of compositions.

The "heatmap showing bright/dark zones of commutator disagreement" (overview.md:699–716) computes something nonzero only because the modular shift `C_{m,n}` is applied alongside the scaling `C_λ`, and these do not commute when m,n are not proportional to λ-scaled shifts. The non-commutativity is real; the *formulation* is just stated wrong.

### 4b. Hamiltonian T2-b boundary

C11's `ψ` is a complex-valued vector, evolved by unitary `U = exp(iθ H_φ)` over phases in `ℚ(ζ_24)`. This is QA-admissible ONLY if ψ is explicitly classified as an **observer-layer amplitude** and the QA state remains the integer tuple `(b,e,d,a)` at each lattice site. Then:

- QA-layer dynamics: integer path `k = 0, 1, 2, …` stepping through orbit states
- Observer-layer dynamics: unitary `U` acts on complex amplitudes
- Boundary crossing occurs twice: input prep (integer → complex amplitude), measurement (complex → integer residue)

As currently written in the chat export, the firewall is implicit and not declared. A cert-gated version MUST declare `ψ ∈ ℂ^N` as observer-side and specify the boundary map.

---

## 5. Gaps vs Primary Sources (Mannheim 2020; Blaschke-Gieres 2021)

### 5a. Mannheim, PRD 102 025020 (arXiv:1909.03548) — ETCRs differ across quantization schemes; unequal-time commutators are the invariant

Mannheim's result, in his abstract: "commutation or anticommutation relations quantized at equal instant time and commutation or anticommutation relations quantized at equal light-front time not only cannot be transformed into each other, they take completely different forms." Equivalence is restored by unequal-time commutators via the Lehmann representation.

What the prior art has:
- C10 is an equal-time CCR `[â, â†] = δ`, written without specifying which t-slice.
- No statement of what the unequal-k (unequal path-time) structure looks like.

What the QA mapping naturally provides (but is not yet in the chat export):
- QA time is integer path length `k` (T1). A t-slice is a fixed `k = k_0` cut; all equal-k commutators are a *choice of projection*.
- The physical invariant should be the **path structure** over unequal k — a function of the orbit traversal, not of any single slice.
- Mannheim's claim maps: "instant-time ETCRs differ from light-front ETCRs" → "different observer-projection choices at the QA→ℂ boundary produce different slice-CCRs; the QA-layer orbit is the invariant object." This is literal Theorem NT at the CCR level.

### 5b. Blaschke & Gieres, NPB 965 (arXiv:2004.14406) — Dirac brackets replace Poisson brackets under gauge constraints

Blaschke-Gieres's point: canonical energy-momentum `P^μ` does not generate spacetime translations via Poisson brackets in general gauges. Dirac brackets — constructed from first-class (gauge) and second-class (gauge-fixing) constraints — are required to produce the correct translation algebra.

What the prior art has:
- C1–C4 propose gauge symmetries and a field strength tensor.
- No constraint handling. No distinction between first-class and second-class. No quotient of redundant states.

What the QA mapping naturally provides:
- Orbits (Singularity, Satellite, Cosmos) are already a physical-state quotient of ℤ/24ℤ^2. The tiered reachability theorem [191] further decomposes S_9 (26% Level-I ceiling).
- Dirac bracket ≈ commutator-mod-constraints. In QA: commutator-mod-orbit-class.
- A QA Dirac bracket candidate: `{f, g}_D := {f, g}_P − {f, φ_i} (C^{-1})^{ij} {φ_j, g}` where `φ_i` are orbit-boundary constraints. This is a structural analogy, not yet a computation.

### 5c. Net gap

The Aug 2025 scaffolding has:
- A QA gauge group candidate (half-formulated)
- A tuple wedge product called a "commutator"
- A non-Abelian field strength written in QA variables
- Equal-time CCR for creation/annihilation
- An untested empirical Noether result
- A cyclotomic Hamiltonian with unclear QA/observer boundary

It does NOT have:
- Distinction between equal-k and unequal-k commutators (Mannheim gap)
- Gauge-constraint quotient via Dirac brackets (Blaschke-Gieres gap)
- Declared observer-projection boundary for the Hamiltonian (T2 gap)
- Repair of the ill-posed projective commutator (§4a)
- Any primary-source citation

The Mannheim and Blaschke-Gieres results BOTH land structurally inside QA when the boundary is drawn correctly: equal-time slices are observer projections (NT), and gauge constraints quotient the QA state space into orbit classes.

---

## 6. Red-Team Summary

1. **C2 is wrong as written.** Scalar λ in ℤ/24ℤ commutes; the proposed commutator is identically zero. Replace with matrix group GL_2(ℤ/24ℤ) or group-theoretic commutator.
2. **C3's "commutator" is actually a wedge product.** `b_1 e_2 − b_2 e_1` is the determinant of the 2×2 matrix `[(b_1,e_1); (b_2,e_2)]`. Useful object, wrong name.
3. **C5 uses `(δa)²`**, which violates S1 in code. Must be written as `δa · δa` when cert-gated.
4. **C7 is empirical, unproven.** "Nearly all grid points" is not a conservation law. A proof would follow from writing `J^μ` explicitly in terms of (b, e, d, a) and applying the QA derivative identities.
5. **C10 omits t-slice declaration.** Mannheim's result says this matters. The ETCR as written is not the invariant.
6. **C11 omits firewall.** The complex amplitude ψ must be declared observer-side; as written, the firewall is crossable in both directions.
7. **All claims lack primary-source grounding.** Not one citation to Mannheim, Blaschke-Gieres, Dirac, Haag-Kastler, Wightman, or any canonical AQFT / lattice-gauge text. The chat export invents vocabulary where established vocabulary exists and differs in meaning.

---

## 7. Proposed Next Moves

This is the end of step 1 (recover + audit). Step 2 is the cross-map to Mannheim + Blaschke-Gieres. Step 3 is cert + paper scope decision.

Candidate cert triples (ranked by axiom-completeness readiness):

- **[cert-A] QA Tuple Wedge** — formalize C3 as a signed bilinear `ω : (b,e)×(b,e) → ℤ/mℤ`, antisymmetric, distributive over addition, zero on the diagonal. Low-risk; purely algebraic; no ETCR claim.
- **[cert-B] QA Gauge Group on GL_2(ℤ/mℤ)** — repair C1+C2 by restricting to 2×2 matrix action. Requires §4a resolution.
- **[cert-C] QA Bosonic CCR** — cert C10 with explicit equal-k declaration as observer-projection and path-time k as invariant. Directly encodes Mannheim's result as a QA structural statement.
- **[cert-D] QA Orbit-Constraint Quotient (Dirac bracket analog)** — cert Blaschke-Gieres mapping: orbit classes as first-class-constraint quotient of ℤ/mℤ². Requires [191] as dependency.

Paper angle: FST Part 3 candidate. Briddell Part 2 submission (2026-04-08, Frontiers ms 1850870) is the precedent. An "ETCR as observer projection; Dirac bracket as orbit quotient" thread reads as a natural Part 3 — the Mannheim + Blaschke-Gieres results are a *physics-literature confession* of two QA axioms (NT + orbit-as-constraint) in their own canonical vocabulary.

Decision pending Will's review: cert triple first (cert-A simplest, cert-C highest-value) or paper draft first.

---

**End of prior-art audit.** No changes to cert registry. No new cert directories created. This doc is repo-anchored at `docs/theory/QA_QFT_COMMUTATORS_PRIOR_ART.md` and supersedes scattered chat-export references for future sessions.
