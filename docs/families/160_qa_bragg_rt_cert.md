# Family [160] QA_BRAGG_RT_CERT.v1

## One-line summary

Bragg's law of X-ray diffraction IS rational trigonometry: n²Q_λ = 4Q_d·s — no transcendental functions needed.

## Mathematical content

### The algebraic identity

Classical Bragg's law: nλ = 2d·sin(θ)

Square both sides:
- n²λ² = 4d²·sin²(θ)
- n²Q_λ = 4Q_d·s

where Q_λ = λ² (wavelength quadrance), Q_d = d² (lattice spacing quadrance), s = sin²θ (diffraction spread).

This is exact. No approximation. Works over any field (integers, rationals, finite fields).

### Miller index quadrance

For cubic crystals with lattice constant a:
- d² = a² / (h² + k² + l²) = a² / Q(h,k,l)
- Q(h,k,l) = h² + k² + l² is the **quadrance of the Miller index direction vector**

Combined identity: n²Q_λ · Q(h,k,l) = 4a² · s

### Crystal system spread conditions

| System | α spread | β spread | γ spread |
|--------|----------|----------|----------|
| Cubic | 1 | 1 | 1 |
| Hexagonal | 1 | 1 | 3/4 |
| Tetragonal | 1 | 1 | 1 |
| Orthorhombic | 1 | 1 | 1 |

- 90° has spread = sin²(90°) = 1
- 120° has spread = sin²(120°) = 3/4 (Wildberger's max equilateral spread)

### NaCl Cu Kα witnesses

All scaled ×10000 for integer arithmetic: a = 56402, λ = 15406.

| Reflection | Q_miller | Q_d | s (exact rational) | θ (°) |
|------------|----------|-----|---------------------|-------|
| (1,0,0) | 1 | 56402² | 59336209/3181185604 | 7.85 |
| (1,1,0) | 2 | 56402²/2 | 59336209/1590592802 | 11.14 |
| (1,1,1) | 3 | 56402²/3 | 59336209/1060395201 | 13.68 |
| (2,0,0) | 4 | 56402²/4 | 59336209/795296401 | 15.85 |

All four verify n²Q_λ = 4Q_d·s exactly (integer equality).

## QA connection

- Miller index Q(h,k,l) = h²+k²+l² is the SAME concept as G = d²+e² in QA (quadrance of direction vector)
- Tetrahedral bond spread 8/9 = QA satellite fraction in mod-9
- Hexagonal γ=120° spread 3/4 = max equilateral spread (Wildberger Ex 7.4)
- 7²−5² = 24 (QA-4 Iota crystal) — origin of mod-24 = origin of crystal structure

## Tier classification

**Tier 1 — Exact algebraic reformulation.** Squaring both sides of Bragg's law is a trivial algebraic operation. The resulting identity n²Q_λ = 4Q_d·s holds exactly. This is not a claim about physics — it is a restatement in Wildberger's rational trigonometry framework.

## Sources

- W.H. Bragg & W.L. Bragg, Proc. R. Soc. A 88, 428-438 (1913)
- N.J. Wildberger, *Divine Proportions* (2005)
- Ben Iverson, *Pythagoras and the Quantum World*, Vol. 4

## Validator

`qa_alphageometry_ptolemy/qa_bragg_rt_cert_v1/qa_bragg_rt_cert_validate.py --self-test`
