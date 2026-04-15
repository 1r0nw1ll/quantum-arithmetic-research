# QA ↔ Group-Equivariant Convolutional Networks (Cohen–Welling 2016)

**Status:** structural note, draft 2026-04-14
**Originator:** Will Dale (2026-04-14 CV-mapping session, post primary-source correction)
**Scope:** map the Cohen–Welling G-CNN generator to QA algebra; identify the QA-canonical rotation group; specify the cert claim.
**Primary sources:**
- Cohen, T. S., & Welling, M. (2016). *Group Equivariant Convolutional Networks.* ICML. arxiv.org/abs/1602.07576 (full-text extracted via Playwright on ar5iv).
- Bruna, J., & Mallat, S. (2013). *Invariant Scattering Convolution Networks.* IEEE TPAMI. arxiv.org/abs/1203.1513 (complementary non-learned invariant family).
- Weiler, M., & Cesa, G. (2019). *General E(2)-Equivariant Steerable CNNs.* NeurIPS. arxiv.org/abs/1911.08251 (SOTA extension).
**Companion files:** `/tmp/cv_papers/gcnn_cohen_welling_2016.pdf`, `/tmp/cv_papers/scattering_bruna_mallat_2013.pdf`, `/tmp/cv_papers/e2_steerable_weiler_cesa_2019.pdf`
**Replaces:** the pixel-histogram invention in `qa_cv_experiments/cv3_orbit_invariance_mnist_rot.py` (2026-04-14, 22% accuracy vs SOTA 97.72%, not a QA mapping but a from-scratch construction in violation of the "map best-performing first" rule).

---

## 1. Why G-CNN is the right QA anchor for CV rotation invariance

**SOTA anchor.** Cohen & Welling (2016) report **2.28 % error on MNIST-rot** (i.e. 97.72 % accuracy) using the p4 and p4m groups — a four-rotation cyclic group (p4) or cyclic + reflection (p4m) acting on the sampling grid. This is the ~95 %+ target we need to reach from the QA side before any CV-3-style experiment is legitimate.

**Structural match.** G-CNNs operate on feature maps that are **functions on a discrete group G**, not on continuous-valued pixel grids. The group is integer-parameterised, composition is defined by matrix multiplication on integer tuples, and feature-map transformation is a group action. This is the QA layer's shape exactly: integer tuples, discrete group action, no continuous state in the core dynamics.

**Complement.** Scattering networks (Bruna & Mallat 2013) give a **non-learned** invariant with similar theoretical status, built on a cascade of modulus-wavelet convolutions indexed by integer paths `(j₁, j₂, …, jₘ)`. This complements G-CNN's learned invariants and is an independent second cert target.

The QA contribution is the **mapping**, per the hard rule: we do not replace G-CNN; we identify what QA's (b,e,d,a) algebra is doing *when a G-CNN with a QA-canonical rotation group is deployed*.

---

## 2. G-CNN generator — verbatim from primary source

### 2.1 Groups tested

From §4.2–4.3 of Cohen–Welling 2016:

**p4** — "all compositions of translations and rotations by 90 degrees about any center of rotation in a square grid". Parameterised by `(r, u, v)` with `0 ≤ r < 4`, `(u, v) ∈ ℤ²`:

    g(r, u, v) = [ cos(rπ/2)  -sin(rπ/2)  u ]
                 [ sin(rπ/2)   cos(rπ/2)  v ]
                 [    0           0       1 ]

**p4m** — p4 plus mirror reflections. Parameterised by `(m, r, u, v)`, `m ∈ {0,1}`, `0 ≤ r < 4`, `(u,v) ∈ ℤ²`.

Split structure (§7): any `g ∈ G` decomposes as `g = t s` with `t ∈ ℤ²` a translation and `s` in the stabilizer of the origin. Stabilizer sizes `S_l`:

- `S = 1` for planar ℤ²
- `S = 4` for p4
- `S = 8` for p4m

### 2.2 Lifting layer (image → feature on G)

Equation 10 of the paper (first-layer G-correlation):

    [f ⋆ ψ](g) = Σ_{y ∈ ℤ²} Σ_k f_k(y) · ψ_k(g⁻¹ y)

Input `f: ℤ² → ℝᴷ` (pixel grid). Output `f ⋆ ψ: G → ℝ` (function on the group).

### 2.3 G-convolution (layers 2+)

Equation 11 (full G-correlation, both operands on G):

    [f ⋆ ψ](g) = Σ_{h ∈ G} Σ_k f_k(h) · ψ_k(g⁻¹ h)

### 2.4 Equivariance

Equation 12 (the load-bearing identity):

    [L_u f] ⋆ ψ = L_u [f ⋆ ψ]

That is: transform-then-convolve = convolve-then-transform. Under rotation `u` of the input, the feature map rotates correspondingly; the information content is preserved.

### 2.5 Invariant pooling via cosets (§6.3)

    P f(g) = max_{k ∈ gU} f(k)

Coset pooling over a subgroup `H ⊂ G` collapses the feature map to the quotient `G/H`, producing an `H`-invariant representation. For `H = R` (the four rotations), a p4 feature map pools down to a function on ℤ² ≅ p4/R — recovering translation-equivariant but rotation-invariant features.

### 2.6 Benchmark

> "In section 8 we report experimental results on MNIST-rot and CIFAR10, where G-CNNs achieve state of the art results (**2.28 % error on MNIST-rot**, and 4.19 % resp. 6.46 % on augmented and plain CIFAR10)."

Translating: **97.72 % accuracy on MNIST-rot**.

---

## 3. QA-canonical instantiation

### 3.1 The rotation-group choice is where QA enters

Cohen–Welling tested p4 (Cₙ with n=4) and p4m (Dₙ with n=4). They note the method generalises to any discrete group acting on the lattice but ship results only for n=4.

**QA prediction.** The natural rotation moduli for QA-native G-CNNs are `n ∈ {9, 24}`:

- **n = 9.** The micro-scale modulus. Matches QA's native `m = 9` orbit structure (Cosmos 24-cycle, Satellite 8-cycle, Singularity). The 9-way cyclic rotation `C₉` acts on a hexagonal / triangular-grid-adapted sampling, *not* on a square grid — this is the first non-trivial structural choice QA forces.
- **n = 24.** The macro-scale, dual-extremal modulus per cert [192] (π(9) = 24, simultaneously the minimum non-trivial Pisano fixed-point modulus and the maximum Carmichael λ=2 modulus). On a square grid, `C₂₄` acts with 15° rotations — much finer resolution than Cohen–Welling's 90° steps.

A correct CV-3-scale experiment compares p4 (n=4) against these QA moduli on MNIST-rot / CIFAR-Rot, with the hypothesis that QA-canonical n's are not worse than the empirically chosen n=4.

### 3.2 QA feature-map encoding

A G-CNN feature map is `f: G → ℝ` where `G = ℤ² × Cₙ`. In QA coordinates with the group Cₙ (rotation) acting on a single spatial cell:

- `b = r + 1` where `r ∈ {0, 1, …, n-1}` is the rotation index → `b ∈ {1, …, n}` (A1 compliant; r offset +1 ensures no-zero).
- `e` = a second integer channel on the same G-point — candidate: intensity bin, scale-level index, or a second rotation in a product group `Cₙ × Cₙ` (for n=9 this gives 81 bins = `m²` points, matching the exhaustive m=9 orbit enumeration used in cert [233]).
- `d = b + e` — derived (A2).
- `a = b + 2e` — derived (A2).

The feature value at `f(g) = f(r, u, v)` is the continuous (float) activation. This is the **observer layer**: continuous floats are legitimate *attached to* the discrete G-point, because the QA invariant — the rotation class, the orbit membership — lives in the index, not the value. The dynamics on the index are discrete; the value at each index is observer content.

### 3.3 Equation correspondences

| G-CNN primitive (Cohen–Welling) | QA primitive |
|---|---|
| Group `G = ℤ² × Cₙ` | Product of spatial integers and a rotation index `b ∈ {1..n}` |
| Feature map `f: G → ℝᴷ` | Observer-valued field over integer (position, b) indices |
| Lifting `[f⋆ψ](g) = Σ_y f(y) ψ(g⁻¹y)` | observer IN projection: pixel grid → G-indexed field |
| G-convolution `[f⋆ψ](g) = Σ_h f(h) ψ(g⁻¹h)` | Resonance coupling on Cₙ orbit — CLAUDE.md pattern `einsum('ik,jk->ij', …)` with kernel indexed by group |
| Equivariance `L_u(f ⋆ ψ) = (L_u f) ⋆ ψ` | Orbit membership survives T-operator action |
| Coset pooling over `H ⊂ G` | Collapsing to quotient = orbit-class histogram |
| Translation → invariance via `H = ℤ²` | ΣΣ spatial aggregation = translation-invariant orbit vector |
| Rotation → invariance via `H = Cₙ` | Coset pool over rotation index = rotation-invariant |

### 3.4 T-operator on the rotation index

For `Cₙ` (cyclic group, rotation by 2π/n), the group operation on `r, r' ∈ {0, …, n-1}` is:

    r ⊕ r' = (r + r') mod n

In QA's A1-safe form with `b = r + 1 ∈ {1..n}`:

    T(b, b') = ((b + b' - 1) mod n) + 1

This is **exactly** the QA `qa_step` operator from `qa_arithmetic` (CLAUDE.md §QA Axiom Compliance, A1). So:

**The QA T-operator IS the Cₙ group operation on the rotation index.** For n ∈ {9, 24}, this is not a coincidence but the shape of the map.

### 3.5 Orbit structure on the rotation index

For n = 9, `Cₙ = C₉`. The QA orbit classes on single-element evolution under T are:

- Singularity: `b = 9` (fixed point of T).
- Satellite: orbits of period dividing 8 (since φ(9) = 6 and the multiplicative structure is separate from additive; under additive T with `b'=b`, the 8-cycle emerges from doubling).
- Cosmos: the 24-cycle on the m=9 ring.

In the G-CNN picture, the rotation index cycles through `C₉` under repeated application of a fixed rotation generator. **QA's orbit classification of C₉ elements = dynamical orbit partition of the G-CNN's rotation-index state under composition.**

---

## 4. Bruna–Mallat scattering as a complementary map (brief)

The scattering transform (eq from Bruna–Mallat 2013) is:

    S_J x(p) = x ⋆ φ_J,  |x ⋆ ψ_{j₁}| ⋆ φ_J,  ||x ⋆ ψ_{j₁}| ⋆ ψ_{j₂}| ⋆ φ_J,  …

where `ψ_j` is a wavelet at scale j (and rotation in the 2-D case), `φ_J` a low-pass averaging, `|·|` modulus.

QA correspondences:

- Path index `(j₁, j₂, …, jₘ)` = QA path time `T₁` (integer path length; CLAUDE.md axiom T1).
- Modulus `|·|` ≈ QA modular reduction `((x - 1) mod m) + 1` in the integer regime.
- Scale hierarchy `j = 0, 1, 2, …, J` = Pisano-modulus hierarchy per `docs/theory/QA_CV_METHODOLOGY_MAPPING.md` §3.3 (mod-3 → mod-9 → mod-24 → mod-72).
- Rotation in Sifre & Mallat 2013 extension: same Cₙ map as §3 above.

This gives a **non-learned** cert target parallel to G-CNN.

---

## 5. Proposed cert (replaces earlier [CV-3] scaffold)

### [CV-3′] QA_G_EQUIVARIANT_CNN_MAPPING_CERT.v1

**Claim (structural).** The G-CNN architecture of Cohen–Welling 2016 is a QA-compliant observer-layer neural network in the sense of Theorem NT:

1. the group index `(r, u, v)` is integer-valued (QA discrete layer);
2. the group operation on `r` is QA's `qa_step` up to A1 offset;
3. feature values `f(g) ∈ ℝ` are legitimate observer content attached to the integer index;
4. equivariance (eq 12) is orbit-preservation under T;
5. coset pooling is projection to orbit-class histogram (observer OUT).

**Claim (empirical, QA-canonical).** G-CNN with rotation group Cₙ for `n ∈ {9, 24}` achieves error rates on MNIST-rot within the statistical envelope of the published p4 (n=4) result (2.28 %), confirming that QA-natural moduli are not a regression.

**Validator plan (for Codex when privileges are reinstated).**

1. Fork Cohen–Welling reference implementation (GitHub: `tscohen/GrouPy`, or Weiler's `e2cnn` which supports arbitrary Cₙ).
2. Train four variants on MNIST-rot with identical architecture and hyperparams:
   - Baseline Z² (planar CNN, no equivariance)
   - p4 (n=4, reproduction)
   - p(9) — C₉ rotation group
   - p(24) — C₂₄ rotation group
3. Report test-error mean ± std over 5 seeds.
4. Cert PASSES if `p(9)` and `p(24)` are within 1 % absolute of `p4` (published 2.28 % ± typical reproduction noise).
5. Cert FAILS if QA-canonical moduli underperform substantially — this would falsify the structural claim that QA picks the "right" n.

**Fixtures.**
- `fixtures/gcnn_pass_p9_mnist_rot.json` — seed-averaged error ≤ 3.3 %
- `fixtures/gcnn_pass_p24_mnist_rot.json` — seed-averaged error ≤ 3.3 %
- `fixtures/gcnn_fail_wrong_group.json` — `p(7)` (not a QA-natural modulus) must underperform by > 0.5 % as negative control
- `fixtures/gcnn_null_z2.json` — Z² baseline > 10 % error

**Dependencies.**
- `e2cnn` or `GrouPy` (pip-installable)
- PyTorch (already present)
- MNIST-rot dataset (published split, torchvision or direct URL)

### [CV-3″] QA_SCATTERING_PATH_PISANO_CERT.v1 (secondary)

**Claim.** The scattering transform path-index `(j₁, …, jₘ)` under the Pisano-modulus hierarchy (j ∈ {log₂ m : m ∈ {3, 9, 24, 72}}) reproduces MNIST-rot accuracy of the published scattering network within envelope.

Deferred pending cert [CV-3′] result.

---

## 6. What this note does *not* claim

- We do **not** claim QA replaces G-CNN. QA identifies the canonical group choice and certifies the discrete-layer compliance.
- We do **not** claim MNIST-rot accuracy > 97.72 % from QA. The ceiling is whatever the underlying G-CNN hits; QA's prediction is that `n ∈ {9, 24}` does not degrade it.
- We do **not** claim novel equivariance theory. Cohen–Welling establish equivariance; we map it.

---

## 7. Relation to `docs/theory/QA_CV_METHODOLOGY_MAPPING.md`

The parent CV-methodology note (2026-04-13) correctly listed "map best-performing first" as §7 rule but then §5 [CV-3] specified an invented pixel-histogram featurizer. That section is **superseded** by the present note. The corrected cert is [CV-3′] above.

`QA_CV_METHODOLOGY_MAPPING.md` should be revised to:
1. Remove the invented orbit-histogram spec in §5 [CV-3].
2. Point [CV-3] to this G-CNN mapping as the canonical formulation.
3. Retain [CV-1] (observer firewall), [CV-2] (Pisano scale-space, now anchored to scattering), [CV-4] (D_k segmentation), [CV-5] (T-orbit tracking) as separate tracks, each gated on primary-source literature acquisition before implementation.

---

## 8. References

- Cohen, T. S., & Welling, M. (2016). Group Equivariant Convolutional Networks. *ICML*. arXiv:1602.07576. Verbatim formulas and 97.72 % MNIST-rot accuracy from §4–§8 of the paper.
- Bruna, J., & Mallat, S. (2013). Invariant Scattering Convolution Networks. *IEEE TPAMI* 35(8), 1872–1886. arXiv:1203.1513.
- Weiler, M., & Cesa, G. (2019). General E(2)-Equivariant Steerable CNNs. *NeurIPS*. arXiv:1911.08251. (98.9 % MNIST-rot; supports arbitrary Cₙ.)
- Sifre, L., & Mallat, S. (2013). Rotation, Scaling and Deformation Invariant Scattering for Texture Discrimination. *CVPR*. (Rotation-group extension of Bruna–Mallat.)
- Cert [192] QA_DUAL_EXTREMALITY_24 (π(9) = 24).
- `memory/feedback_map_best_to_qa.md` (authority for the methodology).
- `docs/specs/PRIMARY_SOURCE_GATE_HOOK_SPEC.md` (elevated enforcement of the rule).
