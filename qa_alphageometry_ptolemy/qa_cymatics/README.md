# QA–Cymatics Correspondence

**One-line synthesis:**
> Cymatics is the experimental study of how lawful resonance generators drive matter into visible, boundary-conditioned geometric states. QA is the formal study of how lawful arithmetic generators drive embedded structures into reachable geometric states.

Both are generator-relative, constraint-driven, produce discrete families, have visible or formal obstructions, and can be organized as certified state transitions.

---

## What this directory contains

| File | Purpose |
|------|---------|
| `qa_cymatics_correspondence_map.json` | Scholarship ledger: 8 scholars, 12 correspondences |
| `qa_cymatics_validate.py` | Deterministic validator for both cert families |
| `mapping_protocol_ref.json` | Gate 0 compliance (canonical mapping protocol reference) |
| `schemas/qa_cymatic_mode_cert.schema.json` | JSON Schema for mode witness certs |
| `schemas/qa_faraday_reachability_cert.schema.json` | JSON Schema for Faraday reachability certs |
| `fixtures/mode_cert_pass.json` | Passing mode cert: circular plate, mode (1,0) |
| `fixtures/mode_cert_fail_off_resonance.json` | Failing mode cert: OFF_RESONANCE |
| `fixtures/faraday_cert_pass.json` | Passing Faraday cert: flat→hexagons reachability |
| `fixtures/faraday_cert_fail_nonlinear_escape.json` | Failing Faraday cert: NONLINEAR_ESCAPE |

---

## Two cert families

### QA_CYMATIC_MODE_CERT.v1 — Chladni mode witness

A certificate that a physical or simulated plate/membrane is in a well-defined eigenmode, and that the resulting Chladni nodal structure maps to a valid QA state pair (b,e).

**State** = realized eigenmode (plate + drive conditions → visible nodal pattern)
**Generator** = control move that changes frequency, amplitude, boundary clamp, medium, or drive point
**Invariant** = nodal symmetry group, Chladni index (m,n), orbit family
**Witness** = nodal graph hash + symmetry group + frequency within tolerance

**Validation checks (7):**

| ID | Check | Fail type if false |
|----|-------|-------------------|
| V1 | Observed frequency within tolerance | OFF_RESONANCE |
| V2 | Nodal counts m,n ≥ 0 | (schema error) |
| V3 | Symmetry group consistent with (m,n) | BOUNDARY_MISMATCH |
| V4 | d_computed = b + e | TUPLE_FORMULA_VIOLATION |
| V5 | a_computed = b + 2e | TUPLE_FORMULA_VIOLATION |
| V6 | a_computed = m + 2n (Chladni formula echo) | TUPLE_FORMULA_VIOLATION |
| V7 | orbit_family consistent with Q(√5) norm mod 3 | ORBIT_CLASS_MISMATCH |

**Chladni formula echo (V6):**
The '+2' in Chladni's f ∝ (m+2n)^k appears verbatim in QA's tuple derivation `a = b + 2e`. V6 verifies this: the tuple's `a` component equals the Chladni mode index `m + 2n`. This is not a coincidence — both encode second-order accumulation of a base displacement (nodal circles accumulate two half-wavelengths; QA's `a` accumulates two generator steps).

**Fail algebra:**

| fail_type | Trigger |
|-----------|---------|
| OFF_RESONANCE | \|obs_freq − drive_freq\| > tolerance; no stable nodal pattern |
| BOUNDARY_MISMATCH | Symmetry group impossible given clamp config + (m,n) |
| MODE_MIXING | Multiple (m,n) needed to describe pattern |
| DAMPING_COLLAPSE | Amplitude too low; sand particles not mobile |
| MEASUREMENT_ALIAS | Apparent symmetry is artifact of photo/capture angle |
| TUPLE_FORMULA_VIOLATION | d/a don't match b+e / b+2e |
| ORBIT_CLASS_MISMATCH | orbit_family inconsistent with norm(b,e) mod 3 |

---

### QA_FARADAY_REACHABILITY_CERT.v1 — Pattern basin reachability

A certificate over a Faraday fluid setup that documents the pattern-basin graph, witnesses legal transitions between pattern classes, and maps pattern families to QA orbit families.

**State** = stable or metastable pattern class (flat, stripes, squares, hexagons, oscillons, quasipattern, disordered)
**Generator** = control move (increase_amplitude, change_frequency, add_surfactant, change_depth, ...)
**Invariant** = spatial symmetry group, Faraday subharmonic ratio (always 1/2), wavelength
**Failure mode** = no stable pattern, mode collision, disorder escape, illegal transition, missing return path

**Three-regime QA mapping:**

| Faraday regime | QA orbit family | Notes |
|---------------|----------------|-------|
| flat (sub-threshold) | singularity | Zero-dimensional fixed point; no spatial structure |
| stripes / squares | satellite | 8-cycle; intermediate organization; mod-3 threshold crossed |
| hexagons / quasipatterns | cosmos | 24-cycle; maximal spatial organization; C6v symmetry |
| disordered | out_of_orbit | No orbit; NONLINEAR_ESCAPE fail type |

**Faraday subharmonic ↔ QA period collapse:**
Faraday: driving at ω → response at ω/2 (period doubling threshold).
QA: cosmos period 24 → satellite period 8 when mod-3 threshold is crossed (period-trisecting).
The shared principle: a parametric threshold selects a subharmonic. The Faraday onset amplitude corresponds to the mod-3 basin boundary condition `dr_b ≡ dr_e ≡ 0 (mod 3)`.

**Validation checks (7):**

| ID | Check | Fail type if false |
|----|-------|-------------------|
| F1 | faraday_subharmonic_ratio = '1/2' | schema error |
| F2 | witnessed_path length = path_length_k | (consistency) |
| F3 | All moves in witnessed_path are legal edges | ILLEGAL_TRANSITION |
| F4 | return_in_k and return_path consistent | RETURN_PATH_NOT_FOUND |
| F5 | current_class is a recognized pattern | PATTERN_CLASS_UNRECOGNIZED |
| F6 | pattern_class_to_orbit_family covers all basin graph nodes | (completeness) |
| F7 | three_regime_correspondence complete | (completeness) |

**Fail algebra:**

| fail_type | Trigger |
|-----------|---------|
| NONLINEAR_ESCAPE | System in disordered regime; no stable basin reachable |
| MODE_MIXING | Multiple spatial modes coexist; no single pattern class |
| DAMPING_COLLAPSE | Amplitude below Faraday threshold; surface stays flat |
| BOUNDARY_MISMATCH | Container geometry prevents target pattern's symmetry |
| ILLEGAL_TRANSITION | Control move not present as legal edge in basin graph |
| RETURN_PATH_NOT_FOUND | return_in_k=true but no valid return path found |
| PATTERN_CLASS_UNRECOGNIZED | Pattern not in recognized taxonomy |

---

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_cymatics

# All fixtures
python qa_cymatics_validate.py --all

# Mode cert: passing
python qa_cymatics_validate.py --mode fixtures/mode_cert_pass.json

# Mode cert: failing (OFF_RESONANCE)
python qa_cymatics_validate.py --mode fixtures/mode_cert_fail_off_resonance.json

# Faraday cert: passing
python qa_cymatics_validate.py --faraday fixtures/faraday_cert_pass.json

# Faraday cert: failing (NONLINEAR_ESCAPE)
python qa_cymatics_validate.py --faraday fixtures/faraday_cert_fail_nonlinear_escape.json

# Demo (auto-detect type from fixtures/)
python qa_cymatics_validate.py --demo
```

---

## Scholarship timeline

| Scholar | Dates | Discovery | QA correspondence |
|---------|-------|-----------|------------------|
| Chladni | 1756–1827 | f ∝ (m+2n)^k; nodal figures | (b,e) pair; a=b+2e; Q(√5) norm |
| Faraday | 1791–1867 | Parametric subharmonic instability | Satellite orbit period collapse |
| Rayleigh | 1842–1919 | Eigenfrequency discreteness; superposition | Orbit period quantization {1,8,24} |
| Watts-Hughes | 1842–1907 | Vowel → Eidophone pattern | Signal → HI fingerprint |
| Waller | 1886–1969 | Mode degeneracy; symmetry group classification | 3-fold cosmos degeneracy; Q(√5) norm |
| Jenny | 1904–1972 | Coined "cymatics"; integer-ratio stability; three-medium hierarchy | Modular stability axiom; three orbit families |
| Lauterwasser | b. 1951 | Biological morphological convergence | Emergent Pythagorean enumeration (Intertwining Theorem) |
| Reid | b. ~1950s | CymaScope digital fingerprinting | HI pipeline as arithmetic CymaScope |

---

## Four cross-cutting principles (proven in both systems independently)

**P1 — Discreteness from boundary conditions.**
Plate shape → discrete eigenfrequencies (Rayleigh). Modulus N → discrete orbit periods {1,8,24} (QA). Same mathematics: eigenvalues of a linear operator on a bounded/finite domain.

**P2 — Quadratic classification.**
Chladni: f ∝ (m+2n)^k — indefinite binary quadratic form on integer pair.
QA: f(b,e) = b²+be−e² — indefinite binary quadratic form. The '+2' coefficient is shared.

**P3 — Rational stability (Jenny's Law = QA's axiom).**
Jenny: rational ratios → stable patterns. QA: modular arithmetic admits only finite orbits → constitutively stable.

**P4 — Three-level hierarchy.**
Jenny: sand/paste/liquid → three organizational regimes.
QA: Singularity (1-cycle) / Satellite (8-cycle) / Cosmos (24-cycle). Both governed by 24 = 2³×3.

---

## What mainstream science says (and why it matters for QA)

The physics literature (Chladni 1787, Rayleigh 1877, Faraday 1831, modern pattern-formation reviews) identifies the primary control variables as: **frequency, amplitude, material properties, boundary conditions, forcing geometry, damping/dissipation, mode competition/symmetry breaking**. The visible form is not arbitrary; it is selected from a constrained modal landscape.

This is the key point for QA: the form is *lawful state selection under generator constraints*, not decoration. Cymatics provides a physically intuitive analogue for QA's core thesis that arithmetic generator moves produce lawfully constrained geometric states.

The non-reduction axiom connection: cymatic patterns are not determined by dimensionless ratios alone — actual scale factors (plate size, fluid depth, stiffness) matter. This supports QA's claim that same reduced ratio ≠ same embedded state.

---

## Open problems

| ID | Statement | Difficulty |
|----|-----------|-----------|
| OP1 | Full (m,n) ↔ (b,e) dictionary: do mode shapes correspond to orbit trajectories? | MEDIUM |
| OP2 | Mathieu equation dictionary: map instability tongue boundary to QA mod-3 basin boundary | HARD |
| OP3 | Build "QA CymaScope": plot orbit state density on (b,e) grid per input signal; compare to CymaScope photos | EASY |
| OP4 | Formalize QA analogue of irrational driving frequency; test | MEDIUM |
| OP5 | Identify symmetry group of QA's 3-fold cosmos degeneracy — conjecture: Gal(Q(√5)/Q) × ℤ/3ℤ | MEDIUM |

---

## Relation to qa_kayser/

Kayser's Harmonik is the 20th-century harmonic-theoretic synthesis of the cymatic tradition — read that directory for the Lambdoma, T-Cross, Rhythmus, and basin separation correspondences. This directory covers the underlying empirical/physical foundations.

```
Chladni → Faraday → Rayleigh → Watts-Hughes → Waller → Jenny
     ↘                                                  ↗
      Kayser (parallel harmonic synthesis) ← qa_kayser/
                                                  ↓
     Lauterwasser → Reid → QA (arithmetic discretization)
```

## Version

v2.0 — 2026-03-21 (full spec stack: 2 schemas, 4 fixtures, validator, mapping protocol ref; README revised to incorporate generator-relative framing and failure algebra)
