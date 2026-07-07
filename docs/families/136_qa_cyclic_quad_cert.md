# Family [136] QA_CYCLIC_QUAD_CERT.v1

## One-line summary

Certifies Ptolemy's theorem via three integer identities for any two QA direction pairs: Brahmagupta-Fibonacci product, Ptolemy product triple, and Ptolemy conjugate triple.

## Mathematical content

Given QA directions (d₁,e₁) and (d₂,e₂) with triples (F₁,C₁,G₁) and (F₂,C₂,G₂):

**BF (Brahmagupta-Fibonacci):**
```
G₁·G₂ = D² + E²    where D = d₁d₂ − e₁e₂,  E = d₁e₂ + d₂e₁
```

**BF conjugate** (checked by CQ_G3 alongside the product BF above, added
to the doc 2026-07-06 — the validator already checks this, the formula
just wasn't written out here): the same identity holds for the
Gaussian-conjugate product Z₁·conj(Z₂):
```
G₁·G₂ = D'² + E'²   where D' = d₁d₂ + e₁e₂,  E' = |d₁e₂ − d₂e₁|
```

**PP (Ptolemy Product):**
```
F₃ = |F₁F₂ − C₁C₂|,   C₃ = F₁C₂ + F₂C₁
F₃² + C₃² = (G₁G₂)²
```

**PC (Ptolemy Conjugate):**
```
F₄ = F₁F₂ + C₁C₂,   C₄ = |F₁C₂ − F₂C₁|
F₄² + C₄² = (G₁G₂)²
```

Both (F₃,C₃,G₁G₂) and (F₄,C₄,G₁G₂) are QA triples lying on the circle of radius G₁G₂. They are the two diagonals of the Ptolemy cyclic quadrilateral.

### Algebraic proof

```
F₃² + C₃² = (F₁F₂−C₁C₂)² + (F₁C₂+F₂C₁)²
           = F₁²F₂² + C₁²C₂² + F₁²C₂² + F₂²C₁²
           = (F₁² + C₁²)(F₂² + C₂²)
           = G₁² · G₂²   ✓
```

The BF identity follows from Gaussian integer multiplication: if Z₁ = d₁+e₁i and Z₂ = d₂+e₂i, then |Z₁|² · |Z₂|² = |Z₁Z₂|² and the real/imaginary parts of Z₁Z₂ are exactly D and E.

### Fundamental example: (2,1)×(3,2)

Triples: (3,4,5) and (5,12,13).

```
D = 2·3 − 1·2 = 4,   E = 2·2 + 3·1 = 7
G₃ = 4² + 7² = 65 = 5·13   ✓  (BF)

F₃ = |3·5 − 4·12| = |15−48| = 33,   C₃ = 3·12+5·4 = 56
33² + 56² = 1089 + 3136 = 4225 = 65²   ✓  (PP)

F₄ = 3·5 + 4·12 = 63,   C₄ = |3·12 − 5·4| = |36−20| = 16
63² + 16² = 3969 + 256 = 4225 = 65²   ✓  (PC)
```

The four triples (3,4,5), (5,12,13), (33,56,65), (63,16,65) are the vertices of a Ptolemy cyclic quadrilateral on the circle G=65.

## Checks

| ID | Description |
|----|-------------|
| CQ_1 | schema_version == 'QA_CYCLIC_QUAD_CERT.v1' |
| CQ_2 | F=d²-e², C=2de, G=d²+e² for each declared triple |
| CQ_3 | F²+C²=G² for each triple |
| CQ_BF | G₁G₂ = D²+E² (Brahmagupta-Fibonacci) |
| CQ_PP | F₃=\|F₁F₂-C₁C₂\|, C₃=F₁C₂+F₂C₁, F₃²+C₃²=(G₁G₂)² |
| CQ_PC | F₄=F₁F₂+C₁C₂, C₄=\|F₁C₂-F₂C₁\|, F₄²+C₄²=(G₁G₂)² |
| CQ_G3 | Both product and conjugate G equal G₁G₂ |
| CQ_W | ≥3 witness pairs |
| CQ_F | Fundamental pair (d₁,e₁)=(2,1), (d₂,e₂)=(3,2) present |

## Edge case: composite product triples

When gcd(D,E) > 1, the product triple is composite. Example: (2,1)×(4,3) gives D=5, E=10, gcd=5, product triple = (75,100,125) = 25·(3,4,5). All three identities still hold.

## Historical chain

| Date | Name | Contribution |
|------|------|-------------|
| ~1600 BCE | Babylonian SPVN | Integer triples on circle |
| ~150 CE | Ptolemy | Chord-table sin/cos addition formulas |
| 628 CE | Brahmagupta | Product of sums-of-squares identity (Brahmasphutasiddhanta §18.65) |
| 1748 | Euler | Gaussian integer multiplication |
| 2020s | QA System | BF+PP+PC as direction-space integer identities |

Ptolemy's chord table formula chord(θ₁±θ₂) = f(chord(θ₁), chord(θ₂)) is exactly the PP and PC formulas in trigonometric disguise — F/G = cos(2θ) and C/G = sin(2θ) for direction angle θ = arctan(e/d).

## Connection to other families

- **[127] QA_UHG_NULL_CERT.v1**: QA triples (F,C,G) are null points [F:C:G] in UHG. The product and conjugate triples correspond to the two intersection points of lines through the null conic — the Ptolemy quadrangle in null geometry.
- **[125] QA_CHROMOGEOMETRY_CERT.v1**: C=Qg, F=Qr, G=Qb are the three chromogeometric quadrances of direction (d,e). The BF identity is Wildberger Theorem 6 (Qb²=Qr²+Qg²) applied to the product direction.
- **[135] QA_PYTHAGOREAN_TREE_CERT.v1**: The tree generates all primitive directions (d,e); this cert pairs any two of them.
- **[133] QA_EISENSTEIN_CERT.v1**: Both certs involve Gaussian multiplication in disguise — [133] in Z[ω], this one in Z[i].

## Fixture files

- `fixtures/cq_pass_fundamental.json` — anchor: (2,1)×(3,2), G₃=65, two diagonals
- `fixtures/cq_pass_witnesses.json` — 5 witness pairs + gcd edge case
- `fixtures/cq_fail_bad_bf.json` — falsifier: wrong Brahmagupta-Fibonacci value (added 2026-07-06)

## Verification Note (2026-07-06)

Confirmed clean, no bugs. Independently recomputed the fundamental
example (2,1)×(3,2) end to end: D=4,E=7,G₃=65 (BF); F₃=33,C₃=56,
33²+56²=65² (PP); F₄=63,C₄=16,63²+16²=65² (PC) — all exact. Also
independently verified the trigonometric identity `F/G=cos(2θ),
C/G=sin(2θ)` for `θ=arctan(e/d)` algebraically (via
`cos2θ=(1−tan²θ)/(1+tan²θ)`, `sin2θ=2tanθ/(1+tan²θ)`). Independently
reconfirmed the gcd edge case: (2,1)×(4,3) genuinely gives product
triple (75,100,125) = 25·(3,4,5) exactly (unlike a similar-sounding but
false "scaled 3-4-5" claim found and fixed in sibling cert [134] this
same cycle — this one is correct).

**Found and closed a documentation-completeness gap** (not a bug): the
validator (`qa_cyclic_quad_cert_validate.py`) genuinely checks a second,
conjugate BF identity (`D'=d₁d₂+e₁e₂`, `E'=|d₁e₂−d₂e₁|`, via
`Z₁·conj(Z₂)`) under check `CQ_G3`, but the doc's "Mathematical content"
section never wrote out the D'/E' formula — only the checks table
alluded to "conjugate G." Added the missing formula; independently
verified it holds for the fundamental example (D'=8, E'=1,
8²+1²=65=G₁G₂). The validator was already correct; only the doc was
incomplete.

**Follow-up (2026-07-06)**: found this family had zero FAIL fixtures
(part of a systemic gap found across 8 sibling families in the
125-139 cluster) and the same latent print-corruption bug first
discovered in cert [132]: a stray `print()` inside the
`result=="FAIL"` short-circuit that corrupts `--self-test`'s stdout
once a FAIL fixture exists to trigger it. Removed the print and added
`fixtures/cq_fail_bad_bf.json` to close the gap and exercise the fix.
