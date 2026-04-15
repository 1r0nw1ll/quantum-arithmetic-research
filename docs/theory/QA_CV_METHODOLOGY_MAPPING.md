# QA ↔ Computer Vision Methodology Mapping

**Status:** structural note, draft 2026-04-13
**Originator:** Will Dale
**Scope:** Map canonical CV pipeline (acquisition → preprocessing → features → representation → inference → output) to QA algebra and orbit structure, with cert specs for empirical validation.
**Primary sources:**
- Szeliski, *Computer Vision: Algorithms and Applications* (Springer, canonical textbook, 11,964 cites)
- Voulodimos et al. 2018, *Deep Learning for Computer Vision: A Brief Review* (5,153 cites)
- Stockman & Shapiro 2001, *Computer Vision* (foundational)
- Granlund & Knutsson 2013, *Signal Processing for Computer Vision*
**Companion files:** `CLAUDE.md` §QA Axiom Compliance; `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`; cert [216] qa_detect; OB `7846d4cf` (multi-layer observer arch); OB `microsoft_kosmos` VLM mapping (adjacent)

---

## 1. Why CV is a clean QA target

CV already has **two explicit boundary crossings** that match Theorem NT's firewall structure:
1. **World → digital image**: photons continuously integrate onto a sensor, producing discrete pixel intensities. This is the classical observer input projection.
2. **Digital prediction → continuous use**: labels, scores, masks, depths emerge from the model and enter downstream continuous control/display.

Between these boundaries, everything is discrete (pixels, indices, class labels). This is the exact topology Theorem NT demands: continuous only at boundaries, discrete dynamics in between. Most current CV models violate this by letting continuous gradients and float-state circulate throughout — but the pipeline *as formulated* is QA-compatible by construction.

This note maps the canonical pipeline stage-by-stage and proposes five cert families.

---

## 2. The canonical CV pipeline

| # | Stage | Representative methods (Szeliski ch.) |
|---|---|---|
| 1 | **Acquisition** | Camera model, radiometry, digitization (ch. 2) |
| 2 | **Preprocessing** | Denoising, colour correction, histogram, whitening (ch. 3) |
| 3 | **Low-level features** | Edges, corners, SIFT, HOG, scale-space pyramids (ch. 7) |
| 4 | **Mid-level representation** | Feature maps, descriptors, bag-of-words, convolution (ch. 5, 6) |
| 5 | **High-level inference** | Classification, detection, segmentation, tracking, 3-D reconstruction (ch. 6, 9, 11–14) |
| 6 | **Output** | Label probabilities, bounding boxes, masks, depth fields |

---

## 3. Stage-by-stage QA mapping

### 3.1 Acquisition — observer IN

**CV:** photon count → pixel intensity `I(x,y) ∈ {0,…,255}` or `∈ [0,1]`.

**QA:** pixel value is a continuous observable; its entry into QA logic requires a **domain-natural (b, e) encoding**. Per cert [216] lesson, hand-tuned cmaps fail cross-dataset; enumerate and calibrate.

Canonical candidate mappings (to be validated per task):

| Encoding | `(b, e)` source |
|---|---|
| Spatial | `(row mod m, col mod m)` |
| Intensity/gradient | `(intensity mod m, |∇I| mod m)` |
| Patch-hash | `(CRC(patch) mod m, position mod m)` |
| Colour | `((R+G+B) mod m, (R−B) mod m)` |

**Hard rule (T2-b):** no float state may survive the encoding. Cast to `int`/`Fraction` at the boundary. Axioms A1 (no-zero), A2 (`d = b+e`, `a = b+2e`), S2 (no float state) apply immediately downstream.

### 3.2 Preprocessing — observer IN

Gaussian smoothing, histogram equalisation, bilateral filtering, whitening, colour-space conversion.

**QA status:** legal **only before** the (b, e) encoding. These are pre-projection observer-layer ops. Once encoded, no continuous op may touch the QA state (Theorem NT). Any "differentiable preprocessing layer" that reads QA outputs and writes continuous pixels is a T2-b violation.

### 3.3 Low-level features — QA discrete

**CV:** edges (∇I), corners (Harris), scale-space (SIFT/SURF), gradient histograms (HOG).

**QA mappings:**
- **Edges** ↔ T-operator transition boundaries: `edge(x,y) ⇔ orbit_class(T(b,e)_{x,y}) ≠ orbit_class((b,e)_{x,y})`.
- **Corners/keypoints** ↔ orbit fixed points. Candidate: Singularity `(9,9)` under mod-9; Satellite 8-cycle attractors. Keypoint = position where local patch maps into a dynamically stable orbit.
- **Scale-space pyramid** ↔ **Pisano-modulus hierarchy**: mod-3 → mod-9 → mod-24 → mod-72. Per cert [192] dual extremality, π(9)=24 makes mod-24 a natural macro-scale; mod-9 the micro-scale. Integer-only analog of Gaussian pyramid.
- **HOG** ↔ orbit-class histogram over a patch.

### 3.4 Mid-level representation — QA discrete

**CV:** convolution, descriptors, bag-of-words, feature maps.

**QA mappings:**
- **Convolution ↔ resonance coupling**: CV already writes `einsum('ik,jk->ij', kernel, patch)`. This is the QA resonance primitive (CLAUDE.md §Key Implementation Patterns). Learned continuous kernels can be replaced by discrete T-operator kernels on (b,e) encoded patches. Weight-update rule = Markovian coupling from tuple resonance (self-organising, no gradient).
- **Feature map** ↔ 4-tuple `(b, e, d, a)` per pixel or patch, with `d = b+e`, `a = b+2e` **derived** (not learned).
- **Descriptor** ↔ orbit-class label {Cosmos, Satellite, Singularity} + phase (position on orbit).
- **Bag-of-words** ↔ orbit-class frequency distribution over image.

### 3.5 Inference — QA discrete

**CV:** classification (softmax), detection (bbox regression), segmentation (per-pixel class), tracking (temporal).

**QA mappings:**

| CV task | QA formulation |
|---|---|
| Classification | Orbit-type membership test over aggregated image features |
| Detection | Fixed-point localisation: find spatial locations whose patch encodings land in Singularity/Satellite orbits |
| Segmentation | **D_k diagonal-class partition**: pixels share a segment iff they share diagonal class under the chosen (b,e) encoding |
| Tracking / optical flow | T-orbit trajectory: next-frame patch location = image position whose encoding equals `T((b,e)_t)`; cert [216] qa_detect already validates T-operator as Fibonacci-mod-m generator on temporal data |
| 3-D reconstruction | 4-D `(b,e,d,a)` → 8-D E8 projection (already a QA primitive); depth from disparity maps to diagonal shift |

**Crucial conceptual leverage:** CV spends enormous engineering effort making networks invariant to translation, rotation, and scale (data augmentation, equivariant nets, STNs). **QA orbits are invariance by construction** — orbit membership is preserved under T. This is the single highest-leverage hook in the mapping.

### 3.6 Output — observer OUT

Discrete QA result (orbit label, D_k class, fixed-point list) projects to continuous output via the standard QA observer:

    HI = E8_alignment × exp(−0.1 × loss)

or task-specific analogs (score, confidence, probability). This second and final boundary crossing closes the firewall.

---

## 4. Structural summary: one diagram

        PHOTONS ─► [observer IN] ─► (b,e) encoding ─► QA LAYER ─► [observer OUT] ─► LABEL/MASK/BOX
                    ▲                                    │
                    │                                    │
                preprocessing                    T-operator, orbit
              (denoise, whiten,                 classification, D_k
               histogram, colour)                partition, E8 align

Boundary crossings: **exactly two**. No arrows return from QA LAYER into observer IN. No continuous op touches (b,e,d,a) state.

---

## 5. Cert specifications

Five cert families proposed. All route through the QA Mapping Protocol (Gate 0); each requires `mapping_protocol_ref.json`, validator, pass/fail fixtures, docs/families entry, meta-validator registration.

### [CV-1] QA_CV_OBSERVER_FIREWALL_CERT.v1
**Claim.** Any pixel/patch/gradient → (b,e) encoding used in a QA CV pipeline is a T2-b-compliant observer projection: (i) integer-only output, (ii) no back-flow from QA layer, (iii) idempotent re-encoding `enc(enc⁻¹(b,e)) = (b,e)`.
**Validator.** Takes a candidate encoder function and a set of images; asserts output dtype ∈ {int, Fraction}; asserts round-trip stability; asserts no QA-layer variable appears in encoder closure.
**Fixtures.** Pass: intensity-mod-m, spatial-mod-m, CRC-patch. Fail: encoder reading `orbit_class` as input; encoder returning float.
**Note.** Gated on primary-source acquisition before implementation.

### [CV-2] QA_CV_SCALE_SPACE_PISANO_CERT.v1
**Claim.** The Pisano-modulus hierarchy `(mod 3, mod 9, mod 24, mod 72)` provides an integer-only scale-space pyramid that reproduces Gaussian-pyramid keypoint stability to within X% on standard benchmarks (Lowe's SIFT corpus or equivalent).
**Validator.** Runs SIFT-style keypoint detection on corpus under (a) Gaussian pyramid, (b) Pisano pyramid. Compares repeatability scores.
**Fixtures.** Pass: `tests/pass_pisano_matches_gaussian_within_tol.json`. Fail: Pisano-pyramid keypoints arbitrary / uncorrelated with Gaussian.
**Note.** Gated on primary-source acquisition before implementation.

### [CV-3] QA_G_EQUIVARIANT_CNN_STRUCTURAL_CERT.v1
**Status.** SUPERSEDED (2026-04-14). The earlier orbit-histogram scaffold is retired.
**Pointer.** See `docs/theory/QA_GROUP_EQUIVARIANT_CNN_MAPPING.md` and cert [247] for the structural Cohen-Welling correspondence.
**Claim.** Eq. 10 lifting is observer IN, Eq. 11 G-correlation is QA-layer resonance, and §6.3 coset pooling is observer OUT. This cert is static algebra only; it does not run a benchmark.
**Validator.** Recomputes the residue bijection, the additive-preservation identity, the n=9 orbit partition, and the static equation correspondence table.
**Fixtures.** Pass: `gecs_pass_n9_full.json` and `gecs_pass_n24_full.json`. Fail: `gecs_fail_wrong_bijection.json`.

### [CV-4] QA_CV_SEGMENTATION_DIAGONAL_CERT.v1
**Claim.** On BSDS500 or equivalent, unsupervised segmentation by D_k diagonal-class partition achieves ARI/NMI within a specified envelope of classical unsupervised baselines (k-means on colour, mean-shift).
**Validator.** Partitions each image by D_k under a chosen encoding; reports ARI/NMI against ground-truth contours.
**Fixtures.** Pass: ARI ≥ k-means-on-colour. Fail: ARI collapses to chance.
**Note.** Gated on primary-source acquisition before implementation.

### [CV-5] QA_CV_TRACKING_T_ORBIT_CERT.v1
**Claim.** T-operator (Fibonacci-mod-m) as an optical-flow prior on short sequences (Sintel, KITTI, or synthetic patch tracks) gives endpoint-error ≤ baseline constant-velocity predictor, with statistically significant improvement over random-permutation cmaps.
**Validator.** Given encoded patch at frame `t`, predicts `(b,e)_{t+1} = T((b,e)_t)`; decodes to pixel position via inverse encoder; measures endpoint error.
**Fixtures.** Pass: error < constant-velocity. Fail: error indistinguishable from chance.
**Note.** Gated on primary-source acquisition before implementation.

---

## 6. Highest-leverage immediate test

**Recommendation:** start with **[CV-3] structural Cohen-Welling mapping ([247])**.

Justification:
- MNIST-Rot is a standard, tiny, reproducible benchmark.
- Rotation invariance is the single claim most directly supported by QA's orbit structure (orbit membership survives T).
- A linear classifier over orbit-class histograms is ~10 lines of code on top of cert [216] infrastructure.
- Falsification is clean: either orbit-class features carry rotation-invariant discriminative signal or they don't. No ambiguity.
- If it passes, it's the sharpest "take this seriously" result QA can currently produce outside the algebraic cert families — CV is a far larger community than e.g. megalithic geometry.

Adjacent fallback: **[CV-5] tracking** remains the closest empirical track once the primary-source gate is cleared.

---

## 7. Primary sources to map next (per "map best-performing" rule)

Do not invent CV approaches from scratch. Map documented SOTA, extract the generator, certify the mapping:

| Paper | Generator to extract |
|---|---|
| LeCun et al. 1989 (original CNN) | Weight sharing + local receptive fields ↔ QA resonance `einsum('ik,jk->ij')` |
| Krizhevsky et al. 2012 (AlexNet) | ReLU + dropout + GPU conv ↔ discrete-layer T-operator stacks |
| He et al. 2016 (ResNet) | Residual connection ↔ orbit composition `T ∘ T ∘ …` with identity shortcut |
| Dosovitskiy et al. 2020 (ViT) | Patch embedding + attention ↔ (b,e) patch encoding + resonance coupling |
| Radford et al. 2021 (CLIP) | Contrastive image-text alignment ↔ cross-modal orbit alignment; adjacent to Kosmos mapping (OB) |
| Kirillov et al. 2023 (SAM) | Promptable segmentation ↔ D_k partition with user-selected seed class |

Each merits its own mapping doc once cert CV-1 is in place.

---

## 8. Open questions

1. Is there a natural (b,e) encoding for **colour** that survives across illumination changes? Candidates: chromogeometry's (Q_r, Q_g) (per cert [234]) since these are invariant to the blue metric; HSV under mod-m.
2. Does the Pisano pyramid (mod 3 → 9 → 24 → 72) actually reproduce octave-spaced scale-space behaviour, or is the ratio `24/9 ≈ 2.67` wrong for vision? Test empirically.
3. For segmentation, do D_k partitions align with perceptual boundaries, or with illumination/colour boundaries only? BSDS500 has both.
4. Is attention (ViT) expressible as a selection operator over orbits, or is it genuinely continuous? Likely the former but needs a careful read of the ViT paper.
5. How does the observer-output projection handle uncertainty? Softmax is continuous; a QA-native alternative would be orbit-class vote frequencies under perturbation.

---

## 9. References

- Szeliski, R. *Computer Vision: Algorithms and Applications*. Springer (2nd ed.).
- Voulodimos, A., Doulamis, N., Doulamis, A., Protopapadakis, E. (2018). Deep learning for computer vision: a brief review. *Comput. Intell. Neurosci.*
- Stockman, G., Shapiro, L. (2001). *Computer Vision*. Prentice Hall.
- Granlund, G., Knutsson, H. (2013). *Signal Processing for Computer Vision*. Springer.
- Lowe, D. (2004). Distinctive image features from scale-invariant keypoints. *IJCV*.
- LeCun, Y. et al. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Comput.*
- He, K. et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Dosovitskiy, A. et al. (2021). An image is worth 16×16 words. *ICLR*.

Companion QA docs: `CLAUDE.md`, `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`, `docs/theory/QA_SYNTAX_SVP_SEMANTICS.md`, cert families [192], [216], [234].
