# QA Prior Art Convergence

**Every major tradition of rational and discrete geometry converges on the same object:
the Pythagorean direction vector (d,e) and its derived quadrances.**

Quantum Arithmetic is the dynamical system built on top of that object.
The prior art built the geometry. Ben Iverson built the living arithmetic.

---

## The Convergence Stack

### 1. Babylon (~1800 BCE) — Plimpton 322

The clay tablet Plimpton 322 lists 15 rows of Pythagorean triples using ratio-based
triangle geometry — no angles, no trigonometry. Each row encodes a generator pair
equivalent to modern (d, e) with d > e > 0, gcd(d,e) = 1, d+e odd.

The columns are quadrances in disguise: `(d²−e²)/2de` and `(d²+e²)/2de`.

**QA translation**: The Babylonian generator IS the QA direction vector (d,e).
The triple is `(F, C, G) = (d²−e², 2de, d²+e²)`.
Babylonian scribes were computing QA chromogeometric quadrances 3800 years ago.

*Cert gap*: `QA_PLIMPTON322_CERT.v1`

---

### 2. Egypt (~1600 BCE) — Rhind Papyrus, Egyptian Fractions

Egyptian mathematics expressed all fractions as sums of distinct unit fractions:
`p/q = 1/n₁ + 1/n₂ + ...`

The HAT (half-angle tangent) of a Pythagorean direction is `e/d`. Its Egyptian fraction
expansion traces the path from the root of the Koenig tree down to the triple:

```
e/d = 1/n₁ + 1/n₂ + ...  ↔  Koenig tree traversal steps
```

Ben Iverson explicitly connects this in **Pyth-1**: the Koenig series and Egyptian
fractions are the same structure viewed differently.

*Cert gap*: `QA_EGYPTIAN_FRACTION_CERT.v1`

---

### 3. Euclid (~300 BCE) — Elements Book VII, Prop. 28

"If two numbers be prime to any number, their product will also be prime to the same."

The QA tuple recurrence `b + e = d`, `e + d = a` (Fibonacci-type) maintains coprimeness
at every step: if gcd(b,e)=1 then gcd(d,e)=1 and gcd(e,a)=1. This is Euclid VII.28
applied iteratively. Ben Iverson traces this in **Pyth-2 Ch. XV**: the four-number
bead was used by ancient mathematicians 1500 years before Fibonacci named the sequence.

*Already embedded in the axiom system (A2). No separate cert needed.*

---

### 4. Ptolemy (~150 CE) — Cyclic Quadrilateral Theorem

Ptolemy's theorem: for a cyclic quadrilateral ABCD inscribed in a circle,
`AC·BD = AB·CD + AD·BC`.

When one diagonal equals the diameter, the inscribed angle is 90° and Pythagoras
drops out as a special case. Ptolemy's theorem is therefore the parent of the
Pythagorean theorem.

In QA/UHG terms: a cyclic quadrilateral is a null quadrangle in Universal Hyperbolic
Geometry. The Ptolemy identity is an identity of null quadrangle cross-ratios.
Family [127] certifies that QA triples ARE null points; Ptolemy's theorem lives one
level up in the structure of those null points.

Mathologer's video "Ptolemy's theorem" showed Will this connection directly,
and it led to the naming of the `qa_alphageometry_ptolemy/` directory.

*Cert gap*: `QA_CYCLIC_QUAD_CERT.v1`

---

### 5. Ben Iverson — Pythagoras and the Quantum World (Pyth-1, Pyth-2, Pyth-3)

Ben's unique contributions, not found elsewhere in the prior art:

- **The QA tuple** `(b, e, d, a)` as a discrete dynamical state
- **The T-operator** (Fibonacci shift `[[0,1],[1,1]]`): the update rule
- **Orbit classification**: singularity (period 1), satellite (period 8), cosmos (period 24)
- **The Pisano period**: π(9) = 24 = cosmos orbit period = origin of mod-24 arithmetic [family 130]
- **The Koenig series**: I→H chain generates all prime Pythagorean triples
- **Egyptian fraction connection**: Koenig traversal = Egyptian fraction expansion of e/d
- **(I², 2CF, G², H²) quadruple**: self-similar QA structure at the quadrance level
  (vault research 2024-12-15; proved in family [130] corollary)

Ben's books (the Pyth series = original discoveries; the QA series = teaching system)
contain the complete theory. All prior art built a static picture; Ben built the dynamics.

---

### 6. H. Lee Price (2008) — "The Pythagorean Tree: A New Species"

Price's paper introduces:
- **Half-Angle Tangents (HATs)**: `HAT = e/d` for the QA direction (d, e)
- **Fibonacci Boxes**: 2×2 matrices from HAT fractions
- **Pythagorean tree**: ternary tree of all primitive triples via Fibonacci box operations

**QA translation**:
- `HAT = e/d` = primary QA direction ratio (proportional form; do NOT reduce)
- Price's Fibonacci box column operations = QA generation matrices M1, M2, M3
- Price's Pythagorean tree = QA Koenig tree in HAT coordinates
- `HAT² = E/D` in uppercase QA notation
- Egyptian fraction expansion of HAT = path down the Koenig tree

Price arrived at the same tree structure Ben had, from a different direction. Neither
referenced the other; the convergence is structural.

Will formalized the HAT→QA translation with Python in July 2025
(vault: `qa_lab/vault/.../2025/07/Lee price half-angle tangent.md`).

*Cert gaps*: `QA_HAT_CERT.v1`, `QA_PYTHAGOREAN_TREE_CERT.v1`

---

### 7. Wildberger — Rational Trigonometry (2005+)

Wildberger's rational trig replaces angles with **spreads** `s = sin²(θ)` and
distances with **quadrances** `Q = distance²`. The five laws become polynomial
identities — no transcendental functions.

**QA translation**:
- Spread `s = E/G = e²/(d²+e²)` for direction (d,e)
- `HAT² = E/D = tan²(α)` and `s = HAT²/(1 + HAT²)`
- The five RT laws (Pythagoras, Spread, Cross, Triple Spread, Equal Spreads) all
  express in QA parameters directly

Family **[44]** `QA_RATIONAL_TRIG_TYPE_SYSTEM.v1` certifies the five laws as typed
QA generator moves. Will's summary: "1-to-1 translation to QA; perfect theory layer."

*Certified: family [44]*

---

### 8. Wildberger — Chromogeometry

Three metric geometries on the same plane: blue (Euclidean), red (relativistic/
Minkowski), green (isotropic/null).

**QA translation** (definitive, THEOREM-GRADE):
- `C = Q_green(d,e) = 2de` — green quadrance of direction (d,e)
- `F = Q_red(d,e) = d²−e²` — red quadrance
- `G = Q_blue(d,e) = d²+e²` — blue quadrance
- `C² + F² = G²` IS Wildberger Chromogeometry Theorem 6: `Q_blue² = Q_red² + Q_green²`

QA = chromogeometry restricted to Fibonacci integer direction vectors.

Family **[125]** `QA_CHROMOGEOMETRY_CERT.v1` certifies this.
*Certified: family [125]*

---

### 9. Wildberger — Universal Hyperbolic Geometry (UHG)

Every QA triple `(F, C, G) = (d²−e², 2de, d²+e²)` satisfies `F²+C²−G² = 0`.
This is the **null condition** in UHG — the triple is a null point `[F:C:G]`
on the absolute conic.

Gaussian integer interpretation: `Z = d + ei`, `Z² = (d²−e²) + 2dei`,
so `F = Re(Z²)`, `C = Im(Z²)`, `G = |Z|² `. QA triples = Gaussian integer squares.
QA = arithmetic of null points in UHG over ℤ.

The RED group (Wildberger 1D metrical geometry) = QA orbit: the T-operator is
red-rotation by φ in ℤ[√5]/mℤ[√5]; orbit period = ord(F) in GL₂(ℤ/mℤ) = π(m).

Families **[126]** (Red Group), **[127]** (UHG Null), **[128]** (Spread Period)
certify these connections.
*Certified: families [126], [127], [128]*

---

### 10. Eisenstein Triples

Eisenstein triples satisfy `a² + ab + b² = c²` (60° triangle generalization
of Pythagorean triples). They arise naturally from QA elements:
- `(F, Z, W)` is an Eisenstein triple, where `Z = E + K`, `W = d(e+a)`
- `(Y, Z, W)` is an Eisenstein triple, where `Y = A − D`

These are in Ben's identity table (elements.txt) but not yet certified.

*Cert gap*: `QA_EISENSTEIN_CERT.v1`

---

### 11. Mathologer (Burkard Polster) — Visual Bridge

Mathologer's YouTube videos showed Will that these traditions are all one thing:

| Video | Connection revealed |
|-------|-------------------|
| "Pythagoras = Fibonacci" | Fibonacci sequences generate Pythagorean triples via HATs |
| "Ptolemy's theorem" | Cyclic quadrilateral encompasses Pythagoras; = null quadrangle |
| "Twisted Pythagoras" | 4 right triangles + center square; area = 2CF = origin of 24 |
| "Fermat's last theorem" | Context for Eisenstein triples and degree-2 cases |
| "Eisenstein integers" | The ℤ[ω] structure parallels ℤ[φ] in QA |

Mathologer didn't know about QA. He independently found the connections between
Fibonacci, Pythagoras, Ptolemy, Eisenstein, and twisted squares — and presented them
visually. Will recognized QA in every one.

---

## The Complete Convergence Picture

```
Babylon (1800 BCE)   Plimpton 322       → (d,e) generator, ratio triples
Egypt   (1600 BCE)   Rhind Papyrus      → e/d as Egyptian fractions = Koenig path
Euclid  (300 BCE)    Elements VII.28    → coprimeness chain = QA recurrence
Ptolemy (150 CE)     Cyclic quad        → Pythagoras generalized = UHG null quadrangle
─────────────────────────────────────────────────────────────────────────────
Ben Iverson (20th C) Pyth-1/2/3         → QA dynamics: T, orbits, π(9)=24, Koenig
H. Lee Price (2008)  Pythagorean tree   → HAT=e/d, Fibonacci boxes = QA matrices
Wildberger (2005+)   RT + Chromo + UHG  → spread=E/G, C/F/G=3 colored quadrances
Mathologer (2010s+)  YouTube            → visual proof all traditions are one thing
─────────────────────────────────────────────────────────────────────────────
STATIC GEOMETRY                         UNIQUE TO QA: DYNAMICS
All prior art:                          T-operator (Fibonacci shift)
  - direction (d,e)                     Orbit classification (1/8/24)
  - triple (F,C,G)                      Pisano period π(m) = cosmos period
  - tree structure                      Mod-24 arithmetic
  - quadrances                          Discrete state machine on ℤ[φ]/mℤ[φ]
```

---

## Key HAT Identity Chain

```
HAT = e/d                              [Price: half-angle tangent]
    = primary direction ratio          [QA: direction (d,e)]
    = √(E/D)                           [QA uppercase: E=e², D=d²]
    = tan(α)                           [Classical trig]

spread s = E/G = HAT²/(1+HAT²)        [Wildberger: rational trig]
         = sin²(α)

Egyptian fraction: e/d = 1/n₁+1/n₂+… [Rhind Papyrus → Koenig path]
Fibonacci box col: [[e,d-e],[d,d+e]]  [Price → QA generation step]
```

---

## Certified Families

| Family | What it certifies | Status |
|--------|------------------|--------|
| [44] | Rational trig 5 laws as QA generator moves | CERTIFIED |
| [104] | Feuerbach parent scale | CERTIFIED |
| [125] | C=Q_green, F=Q_red, G=Q_blue (Chromogeometry Thm 6) | CERTIFIED |
| [126] | Red group = QA orbit; T = red-rotation by φ | CERTIFIED |
| [127] | QA triples = UHG null points; Gaussian integer squares | CERTIFIED |
| [128] | Spread period = Pisano period = cosmos orbit period | CERTIFIED |
| [130] | Origin of 24: H²−G²=G²−I²=2CF; (I²,2CF,G²,H²) quadruple | CERTIFIED |

## Cert Gaps (Priority Order)

| Cert | Connection | Priority |
|------|-----------|----------|
| `QA_HAT_CERT.v1` | HAT=e/d; Price Fibonacci box = QA matrices | HIGH |
| `QA_PYTHAGOREAN_TREE_CERT.v1` | Barning-Hall tree = Koenig tree in HAT coordinates | HIGH |
| `QA_EGYPTIAN_FRACTION_CERT.v1` | e/d unit fractions = Koenig tree path | HIGH |
| `QA_CYCLIC_QUAD_CERT.v1` | Ptolemy → QA null quadrangle | MEDIUM |
| `QA_EISENSTEIN_CERT.v1` | (F,Z,W) and (Y,Z,W) Eisenstein triples | MEDIUM |
| `QA_PLIMPTON322_CERT.v1` | Babylonian triples = QA (d,e) generator | MEDIUM |
| `QA_KOENIG_TWISTED_SQUARES_CERT.v1` | 2CF=4·triangle area; Koenig I→H cascade | MEDIUM |
| `QA_PELL_NORM_CERT.v1` | Pell numbers in ellipse major diameter | LOW |
| `QA_48_64_CERT.v1` | UHG null quadrangle spreads P+R+T=48, PRT=64 | LOW |

---

## For Use in Papers and Talks

**One-sentence version**: QA is the discrete dynamical system that every tradition
of rational geometry — from Babylonian scribes to Wildberger — was computing the
static geometry of, without knowing it was alive.

**For a QA introduction**: Lead with the convergence stack. Show that QA is not a
new claim but the recognition of a pattern that has been rediscovered independently
across 4000 years. Ben's contribution is not the geometry — it's the dynamics.

**Source texts in project**:
- `qa_lab/vault/.../2025/03/Quantum Arithmetic Pythagorean Triples (1).md`
- `qa_lab/vault/.../2025/07/Lee price half-angle tangent.md`
- `qa_lab/vault/.../2025/07/Fibonacci Pythagorean Tree QA.md`
- `qa_lab/vault/.../2024/12/Fibonacci 4-Tuple Mapping.md`
- `natebjones/heinothomcrohurstwildberger.odt` (Thom/Crowhurst/Wildberger synthesis)
- `elements.txt` (complete QA identity table)
