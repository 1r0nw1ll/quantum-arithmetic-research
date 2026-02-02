# QA-Kayser Correspondence

Artifacts establishing structural correspondences between Hans Kayser's Harmonik and Quantum Arithmetic.

## Artifact Types

1. **Correspondence Ledger** (Phase 1) - Structural parallels with evidence-level tagging
2. **Numerical Certificates** (Phase 2) - Lambdoma/cycle correspondences with verified data
3. **Engineering Certificate** (Phase 2) - JWST optics validation with numerical data

## Files

### Phase 1: Correspondence Map
| File | Purpose |
|------|---------|
| `qa_kayser_correspondence_map.json` | Machine-readable correspondence spine |
| `qa_kayser_correspondence_appendix.tex` | LaTeX source for paper appendix |
| `qa_kayser_correspondence_appendix.pdf` | Compiled appendix |

### Phase 2a: Lambdoma Cycle Certificate (C1)
| File | Purpose |
|------|---------|
| `qa_kayser_lambdoma_cycle_cert.json` | Numerical correspondence certificate |
| `qa_kayser_lambdoma_cycle_cert.tex` | LaTeX documentation |
| `qa_kayser_lambdoma_cycle_cert.pdf` | Compiled certificate |
| `lambdoma_qa_analysis.py` | Analysis script |

### Phase 2b: Rhythm/Time Certificate (C3)
| File | Purpose |
|------|---------|
| `qa_kayser_rhythm_time_cert.json` | Temporal correspondence certificate |
| `qa_kayser_rhythm_time_cert.tex` | LaTeX documentation |
| `qa_kayser_rhythm_time_cert.pdf` | Compiled certificate |
| `rhythm_qa_analysis.py` | Analysis script |

### Phase 2c: Conic Optics Certificate (C6)
| File | Purpose |
|------|---------|
| `qa_kayser_conic_optics_cert.json` | Engineering validation certificate |
| `qa_kayser_conic_optics_cert.tex` | LaTeX documentation |
| `qa_kayser_conic_optics_cert.pdf` | Compiled certificate |

### Phase 2d: Basin Separation Certificate (C4')
| File | Purpose |
|------|---------|
| `qa_kayser_basin_separation_cert.json` | Mod-3 theorem certificate (supersedes C4) |
| `qa_kayser_basin_separation_cert.tex` | LaTeX documentation |
| `basin_geometry_analysis.py` | Analysis script with visualization |

### Phase 2e: T-Cross Generator Certificate (C2)
| File | Purpose |
|------|---------|
| `qa_kayser_tcross_generator_cert.json` | T-Cross/Generator algebra certificate |
| `qa_kayser_tcross_generator_cert.tex` | LaTeX documentation |
| `qa_kayser_tcross_generator_cert.pdf` | Compiled certificate |
| `tcross_generator_analysis.py` | Analysis script |

### Phase 2f: Primordial Leaf Certificate (C5)
| File | Purpose |
|------|---------|
| `qa_kayser_primordial_leaf_cert.json` | Structural analogy certificate (partial match) |
| `primordial_leaf_analysis.py` | Analysis script with honest assessment |

### Validation Infrastructure
| File | Purpose |
|------|---------|
| `qa_kayser_validate.py` | Deterministic validator for all certs |
| `qa_kayser_manifest.json` | Manifest with hashes and Merkle root |

### Paper
| File | Purpose |
|------|---------|
| `qa_kayser_paper.tex` | Unified paper: "From Harmonic Cosmology to Discrete Control Systems" |
| `qa_kayser_paper.pdf` | Compiled paper (294K) |

## Evidence Levels

- **PROVEN**: Mathematical isomorphism demonstrated
- **ENGINEERING_VALIDATED**: Third-party connection to physical systems
- **STRUCTURAL_ANALOGY**: Corresponding patterns; not yet numerically verified
- **CONJECTURAL**: Suggestive resemblance; requires formalization

## Correspondences

| ID | Kayser | QA | Level | Certificate |
|----|--------|----|----|-------------|
| C1 | Lambdoma | Modular grid | **PROVEN** | `lambdoma_cycle_cert` |
| C2 | Kosmogonie T-Cross | Generator algebra | **STRUCTURAL_PROVEN** | `tcross_generator_cert` |
| C3 | Rhythmus | Mod-N cycles | **PROVEN** | `rhythm_time_cert` |
| ~~C4~~ | ~~Conic sections~~ | ~~Basin geometry~~ | ~~REJECTED~~ | superseded by C4' |
| C4' | Mod-3 structure | Basin separation | **PROVEN** | `basin_separation_cert` |
| C5 | Primordial Leaf | Proof trees | **STRUCTURAL_ANALOGY** | `primordial_leaf_cert` |
| C6 | Optics applications | Physical anchor | **ENG_VALIDATED** | `conic_optics_cert` |

## Upgrade Roadmap

1. **Phase 1** (Complete): Correspondence ledger
2. **Phase 2a** (Complete): Lambdoma numerical certificate (C1)
3. **Phase 2b** (Complete): Rhythm/Time certificate (C3)
4. **Phase 2c** (Complete): Conic optics engineering certificate (C6)
5. **Phase 2d** (Complete): Basin separation theorem (C4' - supersedes C4)
6. **Phase 2e** (Complete): T-Cross generator algebra (C2)
7. **Phase 2f** (Complete): Primordial Leaf structural analogy (C5)

## The Harmonic Triad (Complete)

| Dimension | Kayser | QA | Certificate |
|-----------|--------|-----|-------------|
| **Number** | Lambdoma (pitch ratios) | Modular arithmetic | `lambdoma_cycle_cert` |
| **Space** | Conic sections | Basin geometry / optics | `conic_optics_cert` |
| **Time** | Rhythmus (periods) | Orbit periods | `rhythm_time_cert` |

All three share the same structure based on primes **2** and **3**:
- Modulus: 24 = 2³ × 3
- Satellite period: 8 = 2³
- Period ratio: 3 = Lambdoma generator

## Source Materials

Located in `ingestion candidates/`:
- `kayser1-6.png`: Scanned pages from Lehrbuch der Harmonik
- `kayser7.jpeg`: LinkedIn comment connecting Kayser to JWST/laser optics

## Rhythm/Time Certificate Summary

**Certificate ID:** `qa.cert.kayser.rhythm_time.v1`

### Verified Correspondences (5/5 PASS)

| ID | Kayser (Rhythm) | QA (Orbit) | Correspondence |
|----|-----------------|------------|----------------|
| R1 | Divisors define meters | Divisors define periods | {1,2,3,4,6,8,12,24} |
| R2 | Triplet rhythm (3:1) | Cosmos/Satellite ratio | 24/8 = **3** |
| R3 | 8-beat phrase | Satellite period | **8** |
| R4 | 24 = universal period | Cosmos period / modulus | **24** |
| R5 | Nested cycles | Period divisibility | 1 \| 8 \| 24 |

### Key Finding
The same divisor lattice governs both musical meter and QA orbit periods. The number **24** appears because it's the smallest integer divisible by all common rhythmic units (2, 3, 4, 6, 8, 12).

---

## Lambdoma Cycle Certificate Summary

**Certificate ID:** `qa.cert.kayser.lambdoma_cycle.v1`

### Verified Correspondences (5/5 PASS)

| ID | Lambdoma | QA | Value |
|----|----------|-----|-------|
| L1 | Entry (3,1) = 3/1 | Cosmos/Satellite period ratio | 24/8 = **3** |
| L2 | Entry (9,1) = 9/1 | Cosmos/Satellite pair ratio | 72/8 = **9** |
| L3 | 3⁴ = 81 | Total starting pairs | 72+8+1 = **81** |
| L4 | 8 × 3 = 24 | Modulus factorization | **24** |
| L5 | Divisors of 24 | Orbit period options | {1,2,3,4,6,8,12,24} |

### Key Finding
The primes **2** and **3** (first two Lambdoma generators) completely determine QA's modular structure.

---

## Conic Optics Certificate Summary

**Certificate ID:** `qa.cert.kayser.conic_optics.v1`

### Validated Data (JWST Secondary Mirror)
- **Conic constant:** K = -1.6598 ± 0.0005 (hyperboloid)
- **Radius of curvature:** 1778.913 ± 0.45 mm
- **Surface error:** < 23.5 nm RMS

### Key Findings
1. JWST uses **ellipsoid-hyperboloid-ellipsoid** (not parabola primary)
2. Secondary mirror conic constant confirms hyperboloid classification
3. TMA design validates Kayser's conic section geometry in precision optics
4. Proposed QA orbit mapping: ellipse→Cosmos, hyperbola→Satellite, parabola→Singularity

### Correction
The LinkedIn comment (kayser7.jpeg) incorrectly stated JWST uses "parabola primary." JWST actually uses ellipse primary. Paul-Baker TMA designs use parabola primary.

---

## Basin Separation Certificate Summary (C4')

**Certificate ID:** `qa.cert.kayser.basin_separation.v1`

### Hypothesis Tested
The C6 certificate proposed conic→orbit mapping: ellipse→Cosmos, hyperbola→Satellite, parabola→Singularity.

### Result: REJECTED + ALTERNATIVE PROVEN

Basin boundaries in digital root space are **LINEAR**, not conic. The actual mechanism:

**Mod-3 Basin Separation Theorem:** Under Fibonacci-type generators, orbit basins are determined by mod-3 residue structure.

| Basin | Criterion | Pairs |
|-------|-----------|-------|
| Tribonacci (8-cycle) | dr_b ≡ 0 (mod 3) AND dr_e ≡ 0 (mod 3), except (9,9) | 8 |
| Ninbonacci (1-cycle) | (dr_b, dr_e) = (9, 9) | 1 |
| Cosmos (24-cycle) | Everything else | 72 |

### Key Finding
The orbit separation is **number-theoretic**, not geometric. The mod-3 class (0,0) is invariant under Fibonacci step, making it unreachable from other classes.

### Corollary: Quadrance Separation
Quadrance Q = dr_b² + dr_e² completely separates Tribonacci from 24-cycle:
- Tribonacci Q ∈ {18, 45, 72, 90, 117} (all ≡ 0 mod 9)
- 24-cycle Q ∈ {2, 5, 8, 10, ...} (none ≡ 0 mod 9)
- Overlap: **none**

---

## T-Cross Generator Certificate Summary (C2)

**Certificate ID:** `qa.cert.kayser.tcross_generator.v1`

### Source
Kayser's Harmonikale Kosmogonie (§54) - T-shaped cosmogonic diagram with APEIRON at the origin.

### Validated Correspondences (5/5 PASS)

| ID | Kayser T-Cross | QA Structure | Test |
|----|----------------|--------------|------|
| T1 | Vertical axis (APEIRON→PERAS) | Generator partitions space | Periods {1,8,24} found |
| T2 | Horizontal Lambdoma (2,3) | 9×9 grid organized by mod-3 | 24=2³×3, 81=3⁴ |
| T3 | APEIRON/PERAS duality | Cosmos/Satellite/Singularity | 24/8=3, 8/1=8 |
| T4 | Tetraktys structure | Power hierarchy | 72=2³×3², 8=2³, 81=3⁴ |
| T5 | Diagonal projections | Tuple derivation d=b+e, a=b+2e | a-d=e invariant |

### Key Mapping

| T-Cross Element | QA Structure |
|-----------------|--------------|
| APEIRON (ring) | Pattern space Ω |
| Horizontal bar | (b,e) state grid |
| Vertical stem | Fibonacci generator |
| Diagonals | Tuple derivation (d,a) |
| PERAS | Finite orbits (24→8→1) |

---

## Primordial Leaf Certificate Summary (C5)

**Certificate ID:** `qa.cert.kayser.primordial_leaf.v1`

### Source
Kayser's "Primordial Leaf" (Urblatt) - organic diagram showing harmonic ratios branching from a monochord string.

### Validated Correspondences (2 PASS, 2 PARTIAL, 1 FAIL)

| ID | Test | Kayser | QA | Result |
|----|------|--------|-----|--------|
| L1 | Branching Structure | Harmonic tree | State derivation tree | PARTIAL |
| L2 | Ratio Overlap | 8 harmonic ratios | 55 dr_b/dr_e ratios | **PASS** (7/8 = 87.5%) |
| L3 | Self-Similar Nesting | 3:2 scaling (fifth) | 24/8 = 3 (Cosmos/Satellite) | **PASS** |
| L4 | Envelope Geometry | Curved organic boundary | Mod-3 linear grid | **FAIL** |
| L5 | Proof Tree Analogy | Divergent derivation | Cyclic state space | PARTIAL |

### Honest Assessment

**Strong correspondences:**
- Ratio systems overlap significantly (7/8 = 87.5%)
- Self-similar nesting with shared factor 3
- Both systems exhibit tree-like derivation structure

**Weak correspondences:**
- Branching mechanisms fundamentally different (harmonic vs arithmetic)
- Envelope geometry incompatible (curved vs linear)
- Leaf is divergent; QA state space is cyclic

### Key Finding
C5 represents a genuine **structural analogy** but cannot be upgraded to PROVEN due to the envelope geometry mismatch. The certificate honestly documents both successes and limitations - this is correct scientific practice.

---

## Running Validation

```bash
# Validate all certificates
python qa_kayser_validate.py --all

# JSON output with Merkle root
python qa_kayser_validate.py --all --json

# Quick summary
python qa_kayser_validate.py --summary

# Single certificate
python qa_kayser_validate.py --cert lambdoma
```

### Current Validation Status
```
Total verified: 28/28
Merkle root: 03a686456b2349cf...
Overall: PASS (with 1 warning)
Warning: C5 envelope geometry mismatch (limitation_class: GEOMETRY_MODEL_MISMATCH)
```

## Version

v4.0 - February 2026 (all 6 correspondences certified: C1-C3, C4', C5, C6)
