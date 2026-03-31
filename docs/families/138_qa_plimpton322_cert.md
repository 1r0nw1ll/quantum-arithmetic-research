# Family [138] QA_PLIMPTON322_CERT.v1

## One-line summary

Certifies that Babylonian tablet Plimpton 322 (~1800 BCE) encodes QA chromogeometric triples: each row is a direction (d,e) with both d,e regular (5-smooth), generating exact sexagesimal values F=d²-e², C=2de, G=d²+e².

## Mathematical content

**Core claim**: Each row of Plimpton 322 corresponds to a QA direction (d,e) where:
- Both d and e are **regular** in base-60 (only prime factors 2, 3, 5)
- Triple: F=d²-e² (short side β), C=2de (middle), G=d²+e² (diagonal δ)
- Pythagorean: F²+C²=G²
- **G/C terminates in sexagesimal**: since d,e regular → C=2de regular → denominator of G/C (reduced) is 5-smooth

**Plimpton column 1** = (G/C)² = (d²+e²)²/(2de)², decreasing from Row 1 to Row 15 as the angle increases from ~45° to ~58°.

**SPVN no-zero**: F,C,G > 0 — the Babylonian no-zero convention matches QA axiom A1.

### Why regularity matters

In base-60 (sexagesimal) arithmetic, a fraction terminates if and only if its denominator (in lowest terms) has only prime factors 2, 3, 5 — since 60 = 2²×3×5. The Babylonians could only compute with such "regular" fractions exactly.

Since C=2de, if d and e are regular then C is regular. The denominator of G/C (after cancellation) divides C, so it is also regular. Therefore G/C is exact in base-60. This is why the tablet uses only regular (d,e) pairs — any irregular pair (like (7,3)) produces a non-terminating base-60 expansion.

### Example rows

| Row | (d,e) | (F,C,G) | G/C base-60 |
|-----|-------|---------|-------------|
| 1   | (12,5) | (119,120,169) | 1;24,30 |
| 5   | (9,4) | (65,72,97) | 1;20,50 |
| 6   | (20,9) | (319,360,481) | 1;20,10 |
| 9   | (25,12) | (481,600,769) | 1;16,54 |
| 11  | (2,1) | (3,4,5)×15 | 1;15 |

**Counterexample**: d=7 is not regular (prime 7 ∉ {2,3,5}). Direction (7,3) gives G/C=97/42; denominator 42=2×3×7 is not 5-smooth → does not terminate in base-60 → absent from the tablet.

### QA chromogeometry connection

```
F = Qr(d,e) = d²-e²  (red quadrance of direction)
C = Qg(d,e) = 2de    (green quadrance)
G = Qb(d,e) = d²+e²  (blue quadrance)
```

This is Wildberger Chromogeometry Theorem 6: Qb²=Qr²+Qg², i.e., G²=F²+C². The Babylonians were computing chromogeometric quadrances 3800 years before Wildberger named them — and 3800 years before QA formalized the direction (d,e) structure.

## Checks

| ID | Description |
|----|-------------|
| P322_1 | schema_version == 'QA_PLIMPTON322_CERT.v1' |
| P322_2 | F=d²-e², C=2de, G=d²+e² for declared triple |
| P322_3 | F²+C²=G² (Pythagorean identity) |
| P322_4 | gcd(d,e)=1, d>e, d-e odd (primitive direction) |
| P322_REG | d and e are 5-smooth (regular in base-60) |
| P322_BASE60 | G/C terminates in base-60 (denominator 5-smooth) |
| P322_NOZERO | F,C,G > 0 (SPVN no-zero = QA A1) |
| P322_W | ≥3 witness rows |
| P322_F | Fundamental (d,e)=(2,1) present |

## Historical chain

| ~1800 BCE | Babylonian scribes | Plimpton 322 — exact sexagesimal (cot,csc) table using regular (d,e) pairs |
|-----------|-------------------|-----------------------------------------------------------------------------|
| ~300 BCE | Euclid | Elements: parameterization of Pythagorean triples via (d,e) |
| 1900 CE | Edgar Banks/G.A. Plimpton | Tablet discovered and purchased |
| 1945 | Neugebauer & Sachs | First mathematical analysis (saw triples, missed generating pairs) |
| 2017 | Mansfield & Wildberger | Historia Mathematica 44:395-419 — identified regular generating pairs, argued it is a trigonometric table |
| 2026 | QA | Formal certification: each row = QA chromogeometric triple; F=Qr, C=Qg, G=Qb |

## Academic significance

This cert connects QA to Mansfield & Wildberger's peer-reviewed 2017 paper — the most academically cited work in the Wildberger program. It directly supports the claim in the Pythagorean Families paper that QA recovers and extends the oldest known mathematical tradition. The tablet predates the Greek-Euclidean tradition by 1500 years and independently discovered the same (d,e) generating structure.

## Connection to other families

- **[125] QA_CHROMOGEOMETRY_CERT.v1**: F=Qr, C=Qg, G=Qb is exactly the chromogeometry assignment certified there
- **[135] QA_PYTHAGOREAN_TREE_CERT.v1**: the regular (d,e) pairs appear in the Pythagorean tree; Plimpton 322 is a snapshot of part of that tree
- **[132] QA_HAT_CERT.v1**: HAT₁=e/d = β/(1+δ) in Mansfield notation
- **[127] QA_UHG_NULL_CERT.v1**: each triple (F,C,G) is a null point [F:C:G] in UHG

## Fixture files

- `fixtures/p322_pass_fundamental.json` — Row 1: (12,5), triple (119,120,169), G/C=1;24,30
- `fixtures/p322_pass_witnesses.json` — 5 rows (Rows 1,5,6,9,11) + non-row counterexample (7,3)
