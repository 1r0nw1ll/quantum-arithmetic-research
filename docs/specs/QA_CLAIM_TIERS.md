# QA Claim Tier System

**Version**: 2.0
**Date**: 2026-04-01
**Authors**: Will Dale, Claude (synthesis), ChatGPT (adversarial review)

## Purpose

Every claim made in the QA project must be assigned to exactly one tier.
This prevents the common failure mode of conflating reformulation with prediction,
or prediction with ontological truth.

## The Four Tiers

### Tier 1 — EXACT

**Definition**: Mathematical identity or exact reformulation. No empirical claim.

**Examples**:
- Bragg's law nλ = 2d·sinθ ↔ n²Q_λ = 4Q_d·s
- Tetrahedral angle: cos(109.47°) = -1/3, therefore spread = 8/9
- Periodic table: shell capacity = 2n² (known formula restated)
- Rydberg formula as quadrance ratio
- WGS84 eccentricity ≈ 9/110 (numerical approximation)

**Certification criterion**: The rewrite is mathematically equivalent to the standard form.

**Failure condition**: The rewrite is NOT mathematically equivalent, or introduces approximations not present in the original.

**What this tier does NOT establish**: Any causal or explanatory role for QA. These are translations, not discoveries.

---

### Tier 2 — STRUCTURAL

**Definition**: Nontrivial correspondence, compression, shared formalism, encoding regularity, or invariant pattern across domains.

**Examples**:
- Tetrahedral spread 8/9 matching QA satellite orbit fraction 8/9
- Hexagonal lattice spread 3/4 matching Wildberger's equilateral bound
- Miller index quadrance Q_hkl = h² + k² + l² being a natural QA object
- Solar system prime-sharing network between QN tuples
- Musical intervals appearing as torus R/r ratios
- 7 crystal systems classifiable by spread conditions

**Certification criterion**: The pattern is nontrivial (not recoverable from trivial encoding choices) and persists under reasonable alternative encodings.

**Failure condition**: The pattern vanishes under null comparison, alternative encoding, or reasonable permutation test. Or it is shown to be an artifact of the body-to-tuple assignment procedure rather than a property of the physical system.

**Mandatory check**: For any Tier 2 claim involving assigned QN tuples, test against null model: do randomly assigned tuples with the same eccentricity-matching constraint produce similar structure?

**What this tier does NOT establish**: Predictive power or physical causation. These are patterns, not mechanisms.

---

### Tier 3 — PREDICTIVE

**Definition**: Out-of-sample lift beyond stated baselines, with null models, ablations, and robustness checks.

**Examples**:
- Finance QCI: partial r = -0.22 beyond lagged RV (permutation validated, 84% robust)
- EEG topographic QA: ΔR² = +0.21 beyond delta baseline (10/10 patients, Fisher p < 10⁻³³)
- Seismology QCI: r = +0.225 OOS (χ² = 61.3, p < 10⁻⁶)
- Audio orbit structure: partial r = +0.752 beyond lag-1 AC (p = 0.020)

**Certification criterion**: ALL of the following must hold:
1. Mapping (signal → QA states) is fixed before scoring
2. Observable (QCI or equivalent) is defined before scoring
3. Evaluation window is held out (OOS)
4. Compared against explicit baselines (not just chance)
5. Robustness: signal persists across parameter variations
6. **Process-level null models** (NOT just event-label permutation):
   - IID noise (random walk)
   - Autocorrelated noise (AR(1), ARMA, GARCH)
   - Phase-randomized surrogate (preserves spectrum, destroys structure)
   - Block-shuffled series (preserves local clustering)
   QCI must beat ALL of these. Simple p-values are insufficient due to
   known pipeline bias toward finding coherence in temporally smooth data.
   (Identified 2026-04-01 via null control failure.)

**Failure condition**: Signal disappears OOS, under permutation, or against stronger baselines. Or the mapping is shown to be post-hoc optimized.

**What this tier does NOT establish**: That QA is the correct ontological description. Predictive power can come from useful formalisms that don't reflect underlying reality (cf. epicycles predicted planetary positions well).

**Critical risk — Encoding Dependence**: Prime sharing networks, QN assignments,
and orbit classifications are all derived representations. For any Tier 3 claim
being considered for Tier 4 escalation, encoding perturbation tests are
NON-NEGOTIABLE:
- Alternate QN mappings (different assignment procedure)
- Randomized but constraint-preserving mappings
- Shuffled prime assignments
- Alternative orbit definitions (different K, different clustering)

If signal disappears -> encoding artifact.
If signal persists -> structural (candidate for Tier 4).

---

### Tier 4 — ONTOLOGICAL

**Definition**: Claim that QA is not merely useful but physically constitutive — that QA dynamics are the actual generative process behind observed phenomena.

**Examples** (hypothetical — none yet certified):
- A cross-domain invariant derivable from QA orbit structure but not from any domain-specific model
- A structural prohibition uniquely predicted by QA (e.g., "no stable configuration has spread equal to a QA singularity value") that nature obeys
- A QA-derived constant of nature that matches experiment to precision not achievable by parameter fitting
- Unification: two apparently unrelated phenomena shown to be the same QA orbit, with this unification predicting a previously unknown third observable

**Certification criterion**: ALL FIVE of the following must hold simultaneously:

1. **Uniqueness (Non-recoverability)**
   The result cannot be reproduced by any model within a reasonable closure
   of standard methods, including: nonlinear time series (ARIMA/GARCH),
   spectral/wavelet methods, graph-based dynamics, state-space/control systems,
   CNNs, Poisson/ETAS models. If ANY of these reproduce the invariant,
   QA is not uniquely required.

2. **Parameter Invariance (Zero Retuning)**
   A single fixed operator T_QA with fixed hyperparameters theta must produce
   statistically significant lift across >= N unrelated domains. No per-domain
   normalization tricks, hidden scaling constants, or re-thresholding allowed.

3. **Constraint Form (Not Just Prediction)**
   The claim must be expressible as an invariant, prohibition, or conservation
   law — not merely a correlation. Examples: "No stable system occupies QA
   singular spread values." "All stress transitions pass through orbit ratio X."
   This is what upgrades model -> theory, correlation -> law.

4. **Encoding Robustness**
   The signal must survive mapping perturbations: alternate QN mappings,
   randomized but constraint-preserving mappings, shuffled assignments,
   alternative orbit definitions. If the signal disappears under alternative
   encodings, it is an encoding artifact, not structure.

5. **Pre-registered Test**
   The test must be defined completely before evaluation. No post-hoc
   adjustment of thresholds, windows, or success criteria.

**Canonical Tier 4 Test Template**:
```
Given:
  Fixed QA operator T
  Fixed parameters theta
  Domains D1...Dn

Claim:
  Invariant I holds across all domains

Test:
  Evaluate I(Di) for each domain OOS

Failure condition:
  Exists Di such that I(Di) is violated beyond epsilon
```

If it survives -> law candidate.
If it fails -> ontology rejected (or refined to smaller scope).

**Failure condition**: ANY of the five criteria above fails.

**Current status**: No Tier 4 claims are certified. This is the hardest tier and should almost never be asserted casually. The project is correctly empty at this tier.

---

## Rules of Application

1. **Every cert family must declare its tier.** No cert may span multiple tiers without explicit justification.

2. **Tier escalation requires new evidence.** A Tier 1 claim cannot become Tier 3 without out-of-sample prediction. A Tier 3 claim cannot become Tier 4 without uniqueness demonstration.

3. **No predictive claim counts unless** the mapping, observable, and evaluation window are fixed before scoring, and compared against explicit null and baseline families.

4. **Honest failure reporting is mandatory.** If a Tier 3 claim fails its robustness check, it must be downgraded or retracted, not buried.

5. **The honest summary** (as of 2026-04-01):

> QA and rational trigonometry provide a common algebraic language for many
> square-and-ratio governed phenomena. Some of these correspondences are exact
> reformulations; others are structural analogies. Separately, in several
> empirical domains, QA-derived observables appear to carry out-of-sample
> predictive information beyond selected baselines. Whether these successes
> indicate a deeper physical ontology, a useful computational formalism, or a
> family of domain-specific encodings remains open.

---

## Appendix: Current Claim Inventory

### Tier 1 (Exact)
- Bragg's law ↔ RT form
- Tetrahedral spread = 8/9
- Shell capacity = 2n²
- Rydberg = Q ratio
- WGS84 ≈ QN (101,9,110,119)
- ECEF via spreads and crosses

### Tier 2 (Structural)
- Tetrahedral 8/9 ↔ satellite fraction
- Crystal systems ↔ spread classification
- Solar system prime harmonic network
- Torus R/r ↔ musical intervals
- Volk E⊥M ↔ C²+F²=G²
- Grant sum-product ↔ QA right triangle

### Tier 3 (Predictive)
- Finance QCI (partial r=-0.22, permutation validated)
- EEG topographic QA (ΔR²=+0.21, 10/10 patients)
- Seismology QCI (r=+0.225, OOS)
- Audio orbit structure (partial r=+0.752)

### Tier 4 (Ontological)
- None certified. Correctly empty.

---

## The Honest State (2026-04-01)

- Tier 1: **Solid.** Clean reformulations, mathematically equivalent.
- Tier 2: **Rich and interesting.** Cross-domain structural patterns worth investigating.
- Tier 3: **The real asset.** Four domains with OOS predictive signal. Don't undersell this.
- Tier 4: **Correctly empty.** This is exactly where a serious theory should be.

The question is no longer "Is QA true?" but:

> "Can QA produce invariant laws that survive adversarial testing across domains?"

That is a real scientific question.

---

## Next Experiment: Cross-Domain QCI Invariance (Tier 4 Candidate)

**Cert family**: QA_CROSS_DOMAIN_INVARIANT_CERT.v1

**Design**:
- Fix T-operator definition globally (Fibonacci shift mod m)
- Fix all hyperparameters (K, window, threshold) identically across domains
- No per-domain normalization or tuning

**Domains**: Finance, EEG, Seismology, Audio

**Measure** (not just "it predicts" — but structural invariants):
1. Same orbit transition ratios at stress events
2. Same temporal lead/lag profile shape
3. Same distributional invariant (entropy, orbit frequency)

**Compare against**: ARIMA/GARCH (finance), CNN/spectral (EEG/audio), Poisson/ETAS (seismology)

**Success criterion**: The invariant holds across all four domains with zero retuning, and no standard model reproduces it without re-derivation.

**Failure criterion**: The invariant fails in any domain, or any standard model reproduces it.

**Status**: Not yet run. This is the kill-or-prove test.
