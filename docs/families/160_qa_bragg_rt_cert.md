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
| (1,0,0) | 1 | 56402² = 3181185604 | 59336209/3181185604 | 7.85 |
| (1,1,0) | 2 | 56402²/2 = 1590592802 | 59336209/1590592802 | 11.14 |
| (1,1,1) | 3 | 56402²/3 = 3181185604/3 (not an integer) | 178008627/3181185604 | 13.68 |
| (2,0,0) | 4 | 56402²/4 = 795296401 | 59336209/795296401 | 15.85 |

All four verify n²Q_λ = 4Q_d·s exactly, as rational-number equality —
three of the four (Q_miller=1,2,4) happen to give an integer Q_d, but
(1,1,1)'s Q_miller=3 does not evenly divide 56402², so its exact Q_d is
genuinely non-integer (3181185604/3). See Verification Note below: a
prior version of this doc and fixture rounded that Q_d to an integer,
which broke exactness without being caught.

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

## Verification Note (2026-07-06)

**Found and fixed a real numeric bug**: the (1,1,1) NaCl reflection's
declared Q_d was rounded down to the integer 1060395201, but the true
value a²/Q(1,1,1) = 56402²/3 = 3181185604/3 is not an integer (56402²
mod 3 = 1). The rounded Q_d was self-consistent with its own declared
`s` value (both computed from the same wrong number), so the doc's
"all four verify exactly (integer equality)" claim passed a naive
self-consistency check while being false for this one row. Fixed the
fixture's Q_d and s to the true exact rational, and corrected the doc's
table and claim wording (three of four rows are integer Q_d; the fourth
is exact but not an integer — that's fine, the identity only requires
rational equality, not integer equality). The displayed angle (13.68°)
was unaffected either way.

**Hardened the validator**: `validate_bragg_reflection` previously only
checked that a witness's own declared `Q_lambda`/`Q_d`/`s` triple was
mutually self-consistent — it never independently verified that `Q_d`
actually equals `a²/(h²+k²+l²)` for the declared Miller indices. Added
that cross-check (using a new `a_lattice` field now present on each
Bragg witness, not just the separate `miller_witnesses` array); a
regression test confirms it now rejects the old rounded Q_d.

**Citations verified**: Bragg & Bragg (1913) confirmed exact — "The
Reflection of X-rays by Crystals," *Proc. R. Soc. A* 88, 428-438, DOI
10.1098/rspa.1913.0040 (added to the fixture/validator, which previously
had no independently-checkable citation pattern). Wildberger's *Divine
Proportions* already independently confirmed real in multiple other
certs this session. Ben Iverson's *Pythagoras and the Quantum World* is
confirmed real (Carlton Press 1982, 2nd ed. Delta Spectrum Research
1995, part of Iverson's QA book series) but the specific "Vol. 4 (Crystal
Universe)" subtitle cited here was not independently located online —
plausible for a small self-published series, not confirmed either way.

**Pure-math claims independently recomputed**: tetrahedral bond spread
= 1 − (−1/3)² = 8/9 exactly (from cos(109.47°) = −1/3 exactly); hexagonal
γ=120° spread = sin²(120°) = 3/4 exactly. Both confirmed via direct
computation, not just trusted from the doc.

No other bugs found. `--self-test` passes on both fixtures.
